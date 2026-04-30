import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE any module imports that read env vars
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

import uuid
import json
import time
import asyncio
import re
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import urllib.request
from fastapi import FastAPI, HTTPException, Query as QueryParam
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from models import ResearchQuery, ResearchReport
from mcp.firecrawl_client import search_and_scrape, FirecrawlSearchError
from research.aggregator import aggregate_sources
from research.synthesizer import (
    synthesize, decompose_query, rate_confidence,
    generate_followups, self_reflect
)
from research.citation_builder import format_report_markdown, validate_citations
from server.llm import call_gemini_json
from memory.supabase_client import (
    save_report, get_session_history, clear_session,
    get_recent_sessions, get_all_sessions, clear_all_sessions,
    upsert_feature_item, insert_feature_item, list_feature_items, get_feature_item
)

app = FastAPI(
    title="DeepTrace — Autonomous Deep Research Agent",
    description="Multi-step AI research agent with query decomposition, confidence reasoning, and self-reflection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Depth configuration
DEPTH_CONFIG = {
    "quick":    {"sources": 3,  "decompose": False},
    "standard": {"sources": 5,  "decompose": False},
    "deep":     {"sources": 10, "decompose": True},
}

# Lightweight in-memory stores for advanced features.
ADV_STORE: Dict[str, Any] = {
    "predictions": [],
    "drift_tracker": [],
    "users": {},
    "notarizations": {},
    "fact_wars": {},
    "identity_requests": [],
    "live_monitors": {},
}

RANK_THRESHOLDS = [
    ("ANALYST", 0, 10),
    ("INVESTIGATOR", 11, 50),
    ("SENTINEL", 51, 200),
    ("ORACLE", 201, 10**9),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _safe_json_loads(raw: str, fallback: Any) -> Any:
    try:
        clean_raw = re.sub(r'```json\s*', '', raw, flags=re.IGNORECASE)
        clean_raw = re.sub(r'```\s*$', '', clean_raw, flags=re.IGNORECASE).strip()
        m = re.search(r'\{.*\}', clean_raw, re.DOTALL)
        if m:
            clean_raw = m.group(0)
        return json.loads(clean_raw)
    except Exception:
        return fallback


def _normalize_confidence_score(value: Any) -> float:
    confidence_score = value or 0
    if isinstance(confidence_score, str):
        try:
            confidence_score = float(confidence_score.replace("%", ""))
        except Exception:
            confidence_score = 0
    try:
        confidence_score = float(confidence_score)
    except Exception:
        confidence_score = 0
    return max(0.0, min(100.0, confidence_score))


def parse_consensus_response(raw: str) -> dict:
    raw = (raw or "").strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        raw = match.group(0)
    try:
        data = json.loads(raw)
        for key in ["overall", "academic", "news", "social"]:
            val = data.get(key, 0)
            try:
                num = float(str(val).replace("%", ""))
                if num <= 1.0:
                    num *= 100
                data[key] = int(round(num))
            except Exception:
                data[key] = 0
        data["summary"] = data.get("summary") or "Could not calculate consensus."
        return data
    except Exception:
        return {"overall": 0, "academic": 0, "news": 0, "social": 0, "summary": "Could not calculate consensus."}


def normalize_report(report: dict) -> dict:
    conf = report.get("confidence_score") or report.get("confidence") or 0
    if isinstance(conf, str):
        conf = conf.replace("%", "").strip()
        try:
            conf = float(conf)
            if conf <= 1.0:
                conf *= 100
        except Exception:
            conf = 0
    report["confidence_score"] = int(round(float(conf or 0)))

    consensus = report.get("consensus") or report.get("global_consensus") or {}
    if consensus:
        for key in ["overall", "academic", "news", "social"]:
            val = consensus.get(key, 0)
            if isinstance(val, str):
                val = val.replace("%", "").strip()
                try:
                    val = float(val)
                    if val <= 1.0:
                        val *= 100
                except Exception:
                    val = 0
            consensus[key] = int(round(float(val or 0)))
        report["consensus"] = consensus
    return report


def _session_color(session_id: str) -> str:
    h = hashlib.sha256((session_id or "deeptrace").encode("utf-8")).hexdigest()
    return f"#{h[:6]}"


def _compute_research_dna(report: ResearchReport) -> Dict[str, Any]:
    confidence = report.confidence_score or 0
    confidence_0_1 = confidence / 100 if confidence > 1 else confidence
    confidence_0_1 = max(0.0, min(1.0, confidence_0_1))
    conflict_count = len(report.conflicts or [])
    finding_count = len(report.key_findings or [])
    source_count = max(1, len(report.citations or []))
    data = {
        "nodes": source_count,
        "density": round(confidence_0_1, 3),
        "spikes": conflict_count,
        "segments": max(3, finding_count),
        "hash": hashlib.sha256(
            f"{report.session_id}:{source_count}:{confidence}:{conflict_count}:{finding_count}".encode("utf-8")
        ).hexdigest()[:16],
        "color": _session_color(report.session_id or ""),
    }
    return data


def _rank_for_queries(query_count: int) -> Dict[str, Any]:
    for rank, lo, hi in RANK_THRESHOLDS:
        if lo <= query_count <= hi:
            return {"rank": rank, "next_rank_at": hi if rank != "ORACLE" else None}
    return {"rank": "ANALYST", "next_rank_at": 10}


def _is_fast_topic(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in ["news", "finance", "politics", "tech", "breaking", "market"])


def _is_academic_query(query: str) -> bool:
    q = (query or "").lower()
    keys = ["science", "medicine", "research", "study", "paper", "clinical", "meta-analysis"]
    return any(k in q for k in keys)


def _is_financial_query(query: str) -> bool:
    q = (query or "").lower()
    keys = ["stock", "ticker", "earnings", "revenue", "sec", "10-k", "10-q", "company", "shares", "guidance"]
    return any(k in q for k in keys) or bool(re.search(r"\b[A-Z]{1,5}\b", query or ""))


def _freshness_label(days_old: int) -> str:
    if days_old <= 7:
        return "FRESH"
    if days_old <= 30:
        return "RECENT"
    if days_old <= 180:
        return "AGING"
    if days_old <= 365:
        return "STALE"
    return "VERY_STALE"


async def _fetch_wayback_snapshot(url: str) -> Dict[str, Any]:
    api = f"https://archive.org/wayback/available?url={url}"
    try:
        with urllib.request.urlopen(api, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        snap = (data.get("archived_snapshots") or {}).get("closest") or {}
        if snap.get("available"):
            return {
                "archived": True,
                "archive_url": snap.get("url"),
                "capture_date": snap.get("timestamp", ""),
                "original_url": url,
            }
    except Exception:
        pass
    return {"archived": False}


async def _enrich_sources_with_resurrection_and_freshness(query: str, sources: List[Any]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for s in sources:
        url = getattr(s, "url", "")
        title = getattr(s, "title", "")
        content = getattr(s, "content", "")
        relevance = getattr(s, "relevance_score", 0.5)
        ts = getattr(s, "scrape_timestamp", "") or _utc_now_iso()
        source_date = now
        try:
            source_date = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            pass
        days_old = max(0, (now - source_date).days)
        freshness = _freshness_label(days_old)
        warning = ""
        if days_old > 365 and _is_fast_topic(query):
            warning = f"SOURCE: {days_old} DAYS OLD - MAY BE OUTDATED"
        archive_meta = await _fetch_wayback_snapshot(url) if url else {"archived": False}
        enriched.append(
            {
                "url": archive_meta.get("archive_url") or url,
                "original_url": url,
                "title": title,
                "content": content,
                "relevance_score": relevance,
                "scrape_timestamp": ts,
                "archived": archive_meta.get("archived", False),
                "capture_date": archive_meta.get("capture_date"),
                "freshness": freshness,
                "days_old": days_old,
                "staleness_warning": warning,
            }
        )
    return enriched

<<<<<<< HEAD
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        current_dir = Path(__file__).resolve().parent
        with open(current_dir / "chat_ui.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<html><body><h1>Error loading landing page</h1><p>Error: {str(e)}</p></body></html>"

@app.get("/login", response_class=HTMLResponse)
@app.get("/dark_ops_login.html", response_class=HTMLResponse)
async def login_page():
    try:
        current_dir = Path(__file__).resolve().parent
        with open(current_dir / "dark_ops_login.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<html><body><h1>Error loading login page</h1><p>Error: {str(e)}</p></body></html>"

@app.get("/chat_ui", response_class=HTMLResponse)
async def chat_page():
    try:
        current_dir = Path(__file__).resolve().parent
        with open(current_dir / "chat_ui.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<html><body><h1>Error loading chat page</h1><p>Error: {str(e)}</p></body></html>"

@app.get("/chat_ui.html", response_class=HTMLResponse)
async def chat_page_html():
    try:
        current_dir = Path(__file__).resolve().parent
        with open(current_dir / "chat_ui.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<html><body><h1>Error loading chat page</h1><p>Error: {str(e)}</p></body></html>"
=======

>>>>>>> 3e9b7f153fab2a6ca9c15927c3d2e4e00d8cc4db

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "DeepTrace", "version": "2.0.0"}

@app.post("/research", response_model=ResearchReport)
async def run_research(query: ResearchQuery):
    """Main endpoint — full research pipeline."""
    try:
        cfg = DEPTH_CONFIG.get(query.depth, DEPTH_CONFIG["standard"])
        raw_sources = await search_and_scrape(query=query.query, num_sources=cfg["sources"])
        if not raw_sources:
            raise HTTPException(status_code=502, detail="No sources could be scraped.")

        sources = await aggregate_sources(sources=raw_sources, query=query.query, min_sources=3)
        session_id = query.session_id or str(uuid.uuid4())
        report = await synthesize(query=query.query, sources=sources, session_id=session_id)

        report.confidence_score = int(round(_normalize_confidence_score(getattr(report, "confidence_score", 0))))
        if not validate_citations(report):
            raise HTTPException(status_code=500, detail="Report validation failed.")

        await save_report(report)
        user_email = os.getenv("DEFAULT_USER_EMAIL", "anonymous@local")
        user = ADV_STORE["users"].get(user_email, {"email": user_email, "query_count": 0, "avg_confidence": 0, "xp": 0})
        user["query_count"] += 1
        user["xp"] += 10
        conf = report.confidence_score or 0
        user["avg_confidence"] = round(((user["avg_confidence"] * (user["query_count"] - 1)) + conf) / max(1, user["query_count"]), 2)
        user["rank"] = _rank_for_queries(user["query_count"])["rank"]
        ADV_STORE["users"][user_email] = user
        await upsert_feature_item("users", user)
        return report

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/research/stream")
async def stream_research(
    query: str,
    max_sources: int = 5,
    depth: str = "standard",
    ghost: bool = False,
):
    """Streaming endpoint with full pipeline: decompose → search → aggregate → synthesize → confidence → followups → reflect"""
    async def event_stream():
        step = 0
        ts = lambda: time.strftime("%H:%M:%S", time.gmtime())
        cfg = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["standard"])
        num_sources = cfg["sources"]
        do_decompose = cfg["decompose"]
        academic_mode = _is_academic_query(query)
        financial_mode = _is_financial_query(query)

        def emit(status, message, **extra):
            nonlocal step
            step += 1
            payload = {"status": status, "step": step, "timestamp": ts(), "message": message}
            payload.update(extra)
            return f"data: {json.dumps(payload)}\n\n"

        try:
            # Step: Fetch previous context
            yield emit("context", "Loading previous research context...")
            recent = await get_recent_sessions(limit=3)
            context_text = ""
            if recent:
                context_parts = [f"- Query: {s['query']} | Summary: {s['summary'][:200]}" for s in recent]
                context_text = "Previous research sessions:\n" + "\n".join(context_parts)

            all_sources = []
            source_type_badges: Dict[str, str] = {}

            if academic_mode:
                yield emit("academic_mode", "Academic mode active: prioritizing arXiv/PubMed/Semantic Scholar sources.")
            if financial_mode:
                yield emit("financial_mode", "Financial intelligence mode active: prioritizing filing and earnings context.")

            # Step: Query decomposition (deep mode)
            sub_questions = [query]
            if do_decompose:
                yield emit("decompose", "Decomposing query into sub-questions...")
                sub_questions = await decompose_query(query)
                yield emit("decompose_done", f"Generated {len(sub_questions)} sub-questions",
                          sub_questions=sub_questions)

            # Step: Search for each sub-question
            for i, sq in enumerate(sub_questions):
                per_q = max(3, num_sources // len(sub_questions))
                yield emit("searching", f"Searching: {sq[:80]}...")
                try:
                    results = await search_and_scrape(query=sq, num_sources=per_q)
                except FirecrawlSearchError as e:
                    yield emit("error", str(e))
                    return
                all_sources.extend(results)
                for s in results:
                    source_type_badges[s.url] = source_type_badges.get(s.url) or ("PEER-REVIEWED" if academic_mode else "")
                # Emit sources for real-time right panel updates
                source_data = [{"url": s.url, "title": s.title, "word_count": s.word_count,
                               "relevance_score": s.relevance_score,
                               "scrape_timestamp": s.scrape_timestamp,
                               "preview": s.content[:300],
                               "paper_type": source_type_badges.get(s.url, "")} for s in results]
                yield emit("sources_found", f"Found {len(results)} sources for sub-query {i+1}",
                          sources=source_data)

            # Mode-specific boosters
            if academic_mode:
                for qx in [f"arxiv {query}", f"pubmed {query}", f"semantic scholar {query}"]:
                    try:
                        ax = await search_and_scrape(query=qx, num_sources=2)
                        all_sources.extend(ax)
                        for s in ax:
                            source_type_badges[s.url] = "PEER-REVIEWED" if "pubmed" in s.url or "nih.gov" in s.url else "PREPRINT"
                    except Exception:
                        pass
            if financial_mode:
                for qx in [f"SEC filing {query}", f"earnings transcript {query}"]:
                    try:
                        fx = await search_and_scrape(query=qx, num_sources=2)
                        all_sources.extend(fx)
                        for s in fx:
                            source_type_badges[s.url] = source_type_badges.get(s.url) or "FILING"
                    except Exception:
                        pass

            if not all_sources:
                yield emit("error", "No sources found. Try a different query.")
                return

            # Step: Aggregate and score
            yield emit("scoring", f"Scoring relevance across {len(all_sources)} sources...")
            sources = await aggregate_sources(all_sources, query, min_sources=3)
            scored_data = [{"url": s.url, "title": s.title, "relevance_score": s.relevance_score,
                           "word_count": s.word_count, "paper_type": source_type_badges.get(s.url, "")} for s in sources]
            yield emit("scored", f"Filtered to {len(sources)} high-relevance sources",
                      scored_sources=scored_data)

            # Step: Synthesize
            yield emit("synthesizing", f"Synthesizing report from {len(sources)} sources...")
            session_id = str(uuid.uuid4())
            report = await synthesize(
                query=query, sources=sources,
                session_id=session_id, previous_context=context_text
            )
            report.sub_questions = sub_questions
            report.depth = depth
            report_data = report.model_dump()
            report_data["academic_mode"] = academic_mode
            report_data["financial_mode"] = financial_mode

            # Step: Confidence reasoning
            yield emit("confidence", "Evaluating finding confidence levels...")
            report = await rate_confidence(report)
            report.confidence_score = int(round(_normalize_confidence_score(getattr(report, "confidence_score", 0))))

            # Step: Validate
            yield emit("validating", "Validating citations and report quality...")
            valid = validate_citations(report)

            # Step: Follow-up suggestions
            yield emit("followups", "Generating follow-up research questions...")
            followups = await generate_followups(query, report)
            report.suggested_followups = followups

            # Step: Self-reflection
            yield emit("reflecting", "Self-evaluating report completeness...")
            reflection = await self_reflect(query, report)

            if not reflection.get("fully_answered", True) and reflection.get("suggested_search"):
                yield emit("refining", f"Running refinement search: {reflection['suggested_search'][:60]}...")
                extra_sources = await search_and_scrape(
                    query=reflection["suggested_search"], num_sources=3
                )
                if extra_sources:
                    extra_scored = await aggregate_sources(extra_sources, query, min_sources=1)
                    # Quick re-synthesis for additional findings
                    extra_text = "\n".join(f"- {s.title}: {s.content[:500]}" for s in extra_scored)
                    from server.llm import call_gemini_json
                    refine_result = await call_gemini_json(
                        f"Extract 1-2 NEW findings from these additional sources that weren't in the original report:\n{extra_text}\n\nOriginal findings: {', '.join(report.key_findings[:3])}\n\nRespond with JSON array: [\"new finding 1\", \"new finding 2\"]",
                        "You are a research supplement agent."
                    )
                    try:
                        new_findings = json.loads(refine_result)
                        if isinstance(new_findings, list):
                            report.refined_findings = new_findings[:2]
                    except Exception:
                        pass
                yield emit("refined", f"Added {len(report.refined_findings)} refined findings",
                          reflection=reflection)
            else:
                yield emit("complete", "Query fully answered — no refinement needed",
                          reflection=reflection)

            # Step: Advanced metadata
            report_data = report.model_dump()
            report_data["confidence_score"] = int(round(_normalize_confidence_score(
                report_data.get("confidence_score") or report_data.get("confidence") or 0
            )))
            report_data["research_dna"] = _compute_research_dna(report)
            report_data["ghost"] = ghost
            report_data["academic_mode"] = academic_mode
            report_data["financial_mode"] = financial_mode

            # Step: Save to memory (skip in ghost mode)
            if not ghost:
                yield emit("saving", "Saving report to session memory...")
                await save_report(report)
                user_email = os.getenv("DEFAULT_USER_EMAIL", "anonymous@local")
                user = ADV_STORE["users"].get(user_email, {"email": user_email, "query_count": 0, "avg_confidence": 0, "xp": 0})
                user["query_count"] += 1
                user["xp"] += 10
                conf = report.confidence_score or 0
                user["avg_confidence"] = round(((user["avg_confidence"] * (user["query_count"] - 1)) + conf) / max(1, user["query_count"]), 2)
                user["rank"] = _rank_for_queries(user["query_count"])["rank"]
                ADV_STORE["users"][user_email] = user
                await upsert_feature_item("users", user)
            else:
                yield emit("ghost_mode", "Ghost mode active: report is not persisted.")

            # Final: send complete report
            report_data = normalize_report(report_data)
            yield f"data: {json.dumps({'status': 'done', 'step': step + 1, 'timestamp': ts(), 'report': report_data})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'step': step + 1, 'timestamp': ts(), 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/research/export")
async def export_research(session_id: str, format: str = "markdown"):
    """Export last report for a session in various formats."""
    history = await get_session_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="No reports found for this session.")
    
    last_report_data = history[-1]
    # Reconstruct report object
    from models import Citation
    citations = [Citation(**c) for c in last_report_data.get("citations", [])]
    report = ResearchReport(
        query=last_report_data.get("query", ""),
        summary=last_report_data.get("summary", ""),
        key_findings=last_report_data.get("key_findings", []),
        citations=citations,
        sources_scraped=last_report_data.get("sources_scraped", 0),
        confidence_score=last_report_data.get("confidence_score", 0),
        session_id=session_id
    )
    
    md = format_report_markdown(report)
    
    if format == "markdown":
        return PlainTextResponse(md, media_type="text/markdown",
                                headers={"Content-Disposition": f"attachment; filename=deeptrace-report-{session_id[:8]}.md"})
    elif format == "pdf":
        # Convert markdown to PDF
        try:
            import weasyprint
            # Move replacement outside of f-string to avoid SyntaxError
            md_with_br = md.replace('\n', '<br>')
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                    h1 {{ color: #333; border-bottom: 2px solid #00ff88; padding-bottom: 10px; }}
                    h2 {{ color: #555; margin-top: 30px; }}
                    .finding {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-left: 4px solid #00ff88; }}
                    .citation {{ font-size: 0.9em; color: #666; }}
                </style>
            </head>
            <body>
                {md_with_br}
            </body>
            </html>
            """
            pdf = weasyprint.HTML(string=html_content).write_pdf()
            return Response(
                content=pdf,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=deeptrace-report-{session_id[:8]}.pdf"}
            )
        except ImportError:
            # Fallback if weasyprint not available
            return PlainTextResponse(md, media_type="text/markdown",
                                    headers={"Content-Disposition": f"attachment; filename=deeptrace-report-{session_id[:8]}.md"})
    elif format == "ppt":
        # Create PPT-like content (simplified HTML that can be opened as PPT)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>DeepTrace Report - {report.query[:50]}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .slide {{ page-break-after: always; min-height: 90vh; border: 1px solid #ddd; padding: 40px; margin-bottom: 20px; }}
                .title {{ font-size: 24px; font-weight: bold; color: #00ff88; margin-bottom: 20px; }}
                .content {{ font-size: 18px; line-height: 1.6; }}
                .finding {{ background: #f9f9f9; padding: 20px; margin: 15px 0; border-left: 4px solid #00ff88; }}
            </style>
        </head>
        <body>
            <div class="slide">
                <div class="title">DeepTrace Research Report</div>
                <div class="content">
                    <h2>Query: {report.query}</h2>
                    <p><strong>Confidence Score:</strong> {report.confidence_score}/10</p>
                    <p><strong>Sources Analyzed:</strong> {report.sources_scraped}</p>
                </div>
            </div>
            <div class="slide">
                <div class="title">Key Findings</div>
                <div class="content">
        """
        
        for i, finding in enumerate(report.key_findings, 1):
            html_content += f'<div class="finding"><strong>{i}.</strong> {finding}</div>'
        
        html_content += """
                </div>
            </div>
            <div class="slide">
                <div class="title">Summary</div>
                <div class="content">""" + report.summary + """</div>
            </div>
        </body>
        </html>
        """
        return Response(
            content=html_content.encode('utf-8'),
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=deeptrace-report-{session_id[:8]}.ppt.html"}
        )
    
    return PlainTextResponse(md)

@app.get("/research/share/{session_id}")
async def share_research(session_id: str):
    """Generate shareable link for a research session."""
    # Create a shareable URL that points to a public view of the research
    share_url = f"{os.getenv('BASE_URL', 'http://localhost:8000')}/shared/{session_id}"
    return {"share_url": share_url, "session_id": session_id}

@app.get("/shared/{session_id}")
async def view_shared_research(session_id: str):
    """Public view of a shared research session."""
    history = await get_session_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Research session not found.")
    
    last_report_data = history[-1]
    # Reconstruct report object
    from models import Citation
    citations = [Citation(**c) for c in last_report_data.get("citations", [])]
    report = ResearchReport(
        query=last_report_data.get("query", ""),
        summary=last_report_data.get("summary", ""),
        key_findings=last_report_data.get("key_findings", []),
        citations=citations,
        sources_scraped=last_report_data.get("sources_scraped", 0),
        confidence_score=last_report_data.get("confidence_score", 0),
        session_id=session_id
    )
    
    # Generate HTML for shared view
    md = format_report_markdown(report)
    # Move replacement outside of f-string
    md_with_br = md.replace('\n', '<br>')
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>DeepTrace Research - {report.query[:50]}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; max-width: 800px; }}
            h1 {{ color: #333; border-bottom: 2px solid #00ff88; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .finding {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-left: 4px solid #00ff88; }}
            .citation {{ font-size: 0.9em; color: #666; }}
            .header {{ background: #00ff88; color: #000; padding: 20px; margin: -40px -40px 20px -40px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 DeepTrace Research Report</h1>
            <p><strong>Query:</strong> {report.query}</p>
            <p><strong>Confidence Score:</strong> {report.confidence_score}/10</p>
            <p><strong>Sources Analyzed:</strong> {report.sources_scraped}</p>
        </div>
        {md_with_br}
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/sessions")
async def list_sessions():
    """Get all research sessions for the history panel."""
    sessions = await get_all_sessions()
    return {"sessions": sessions}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get research history for a session"""
    history = await get_session_history(session_id)
    return {"session_id": session_id, "history": history}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear a research session"""
    await clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}

@app.delete("/sessions/all")
async def delete_all_sessions():
    """Clear all research sessions"""
    await clear_all_sessions()
    return {"message": "All sessions cleared"}

@app.get("/tasks")
async def list_tasks():
    """OpenEnv-compatible tasks endpoint"""
    return {
        "agent": "DeepTrace",
        "version": "2.0.0",
        "tasks": ["deep_research"],
        "actions": ["decompose", "search_and_scrape", "aggregate", "synthesize",
                    "rate_confidence", "generate_followups", "self_reflect", "cite"]
    }


@app.post("/research/lie-detector")
async def lie_detector(payload: Dict[str, Any]):
    text = (payload or {}).get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    sentences = _split_sentences(text)

    async def event_stream():
        stats = {"VERIFIED": 0, "MISLEADING": 0, "FALSE": 0, "UNVERIFIABLE": 0}
        for idx, sentence in enumerate(sentences):
            verdict = "UNVERIFIABLE"
            reason = "Insufficient corroboration from live sources."
            source_url = ""
            try:
                srcs = await search_and_scrape(query=sentence, num_sources=1)
                if srcs:
                    source_url = srcs[0].url
                raw = await call_gemini_json(
                    f"Does live web evidence support, contradict, or neither verify this claim: {sentence}\n"
                    f"Top source: {source_url}\n"
                    "Return JSON: {\"verdict\":\"VERIFIED|MISLEADING|FALSE|UNVERIFIABLE\",\"source_url\":\"...\",\"reason\":\"...\"}",
                    "You are a strict fact-checking analyst.",
                )
                out = _safe_json_loads(raw, {})
                verdict = out.get("verdict", verdict)
                source_url = out.get("source_url", source_url)
                reason = out.get("reason", reason)
            except Exception:
                pass
            if verdict not in stats:
                verdict = "UNVERIFIABLE"
            stats[verdict] += 1
            payload = {
                "status": "sentence",
                "index": idx,
                "sentence": sentence,
                "verdict": verdict,
                "source_url": source_url,
                "reason": reason,
                "stats": stats,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        total = max(1, len(sentences))
        summary = {k.lower(): round(v * 100 / total, 1) for k, v in stats.items()}
        yield f"data: {json.dumps({'status': 'done', 'summary': summary, 'sentences': len(sentences)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/research/devils-advocate")
async def devils_advocate(payload: Dict[str, Any]):
    session_id = (payload or {}).get("session_id", "")
    history = await get_session_history(session_id) if session_id else []
    inline_report = (payload or {}).get("report") or {}
    if inline_report and inline_report.get("query"):
        original = inline_report
    elif history:
        original = history[-1]
    else:
        raise HTTPException(status_code=404, detail="session not found and no inline report provided")
    findings = original.get("key_findings", []) or []
    query = original.get("query", "") or ""

    async def event_stream():
        yield f"data: {json.dumps({'status':'start','message':'Building strongest counter-case...'})}\n\n"
        prompt = (
            "Generate strongest possible counter-arguments to these findings and include contradictory evidence cues.\n"
            f"Query: {query}\nFindings: {json.dumps(findings)}\n"
            "Return JSON: {\"counter_summary\":\"...\",\"counter_findings\":[\"...\"],\"verdict_bias\":0-100}"
        )
        out = _safe_json_loads(await call_gemini_json(prompt, "You are an adversarial analyst."), {})
        counter_findings = out.get("counter_findings", [])[:8]
        counter_sources = []
        for cf in counter_findings[:5]:
            try:
                s = await search_and_scrape(query=f"evidence against: {cf}", num_sources=1)
                if s:
                    counter_sources.append({"title": s[0].title, "url": s[0].url})
                yield f"data: {json.dumps({'status':'stream','finding':cf})}\n\n"
            except Exception:
                pass
        result = {
            "case_for": original,
            "case_against": {
                "query": query,
                "summary": out.get("counter_summary", ""),
                "key_findings": counter_findings,
                "citations": counter_sources,
            },
            "verdict": {
                "favors": "COUNTER" if (out.get("verdict_bias", 50) or 50) < 50 else "ORIGINAL",
                "percent": abs((out.get("verdict_bias", 50) or 50) - 50) * 2,
            },
        }
        yield f"data: {json.dumps({'status':'done','result':result})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/research/archaeology")
async def source_archaeology(url: str):
    try:
        srcs = await search_and_scrape(query=f"first publication of claim from {url}", num_sources=5)
        chain = []
        for i, s in enumerate(srcs):
            chain.append(
                {
                    "url": s.url,
                    "domain": s.url.split("/")[2] if "://" in s.url else s.url,
                    "date": getattr(s, "scrape_timestamp", _utc_now_iso())[:10],
                    "type": "ORIGIN" if i == 0 else ("REPUBLISH" if i < 3 else "DERIVATIVE"),
                }
            )
        return {"chain": chain, "origin_unverified": len(chain) == 0}
    except Exception:
        return {"chain": [], "origin_unverified": True}


@app.post("/predictions/add")
async def add_prediction(payload: Dict[str, Any]):
    item = {
        "id": str(uuid.uuid4()),
        "user_email": payload.get("user_email", "anonymous@local"),
        "finding_text": payload.get("finding_text", ""),
        "session_id": payload.get("session_id", ""),
        "query": payload.get("query", ""),
        "created_at": _utc_now_iso(),
        "check_interval_days": int(payload.get("check_interval_days", 30)),
        "initial_confidence": payload.get("initial_confidence", 50),
    }
    row = await insert_feature_item("predictions", item)
    ADV_STORE["predictions"].append(row)
    return {"ok": True, "prediction": row}


@app.get("/predictions/my")
async def my_predictions(user_email: str = QueryParam(...), force_check: bool = False):
    table_preds = await list_feature_items("predictions", limit=200)
    preds = [p for p in (table_preds or ADV_STORE["predictions"]) if p.get("user_email") == user_email]
    out = []
    for p in preds:
        status = "PENDING"
        conf_delta = 0
        sources = []
        if force_check or True:
            try:
                srcs = await search_and_scrape(query=p["finding_text"], num_sources=1)
                if srcs:
                    sources = [{"url": srcs[0].url, "title": srcs[0].title}]
                    status = "PARTIAL"
            except Exception:
                pass
        out.append({"prediction": p, "status": status, "confidence_delta": conf_delta, "sources": sources})
    resolved = [x for x in out if x["status"] in ("CONFIRMED", "DENIED")]
    acc = round((len([x for x in resolved if x["status"] == "CONFIRMED"]) / max(1, len(resolved))) * 100, 1)
    return {"predictions": out, "accuracy": acc}


@app.post("/research/consensus")
async def consensus(payload: Dict[str, Any]):
    q = payload.get("query", "")
    groups = {"academic": f"scholar research {q}", "news": f"news analysis {q}", "social": f"social discussion {q}"}
    source_titles = {}
    for k, search_q in groups.items():
        try:
            srcs = await search_and_scrape(query=search_q, num_sources=3)
            source_titles[k] = [s.title for s in srcs]
        except Exception:
            source_titles[k] = []

    consensus_prompt = f"""
Analyze these sources and return ONLY valid JSON, no extra text:
{{
  "overall": <integer 0-100>,
  "academic": <integer 0-100>,
  "news": <integer 0-100>,
  "social": <integer 0-100>,
  "summary": "<one sentence>"
}}
Base the scores on how much the sources collectively support the main finding.
Query: {q}
Academic sources: {source_titles.get("academic", [])}
News sources: {source_titles.get("news", [])}
Social sources: {source_titles.get("social", [])}
"""
    groq_consensus_response = ""
    try:
        groq_consensus_response = await call_gemini_json(consensus_prompt, "You are a consensus analyst. Return only valid JSON with integer values 0-100.")
        # Strip markdown fences if present
        raw = groq_consensus_response.strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        # Extract JSON object if surrounded by text
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        consensus_data = json.loads(raw)
        # Normalize all numeric fields - handle string/float/None
        for key in ["overall", "academic", "news", "social"]:
            val = consensus_data.get(key, 0)
            if isinstance(val, str):
                val = re.sub(r'[^0-9.]', '', val)
                val = float(val) if val else 0
            val = float(val or 0)
            # Convert 0-1 range to 0-100
            if 0 < val <= 1:
                val = val * 100
            consensus_data[key] = int(round(val))
        consensus_data["summary"] = str(consensus_data.get("summary") or "")
    except Exception as e:
        raw_preview = groq_consensus_response[:200] if groq_consensus_response else "none"
        print(f"Consensus parse error: {e}, raw: {raw_preview}")
        consensus_data = {"overall": 0, "academic": 0, "news": 0, "social": 0, "summary": "Could not assess consensus."}
    return consensus_data


@app.post("/research/pressure-test")
async def pressure_test(payload: Dict[str, Any]):
    session_id = payload.get("session_id")
    history = await get_session_history(session_id) if session_id else []
    if not history:
        return {"error": "Session not found", "objections": [], "score": "0/5", "verdict": "UNKNOWN"}

    report = history[-1]
    findings = report.get("key_findings", []) or []
    findings_text = "\n".join(findings)

    prompt = f"""You are an adversarial fact-checker. Generate exactly 5 strong objections to these research findings, then for each objection search for supporting evidence.

Findings:
{findings_text}

Return ONLY valid JSON:
{{
  "objections": [
    {{"text": "objection text", "severity": "HIGH", "verdict": "HOLDS", "source_url": "https://example.com", "reason": "why"}},
    {{"text": "objection text", "severity": "MED", "verdict": "FAILS", "source_url": "", "reason": "why"}},
    {{"text": "objection text", "severity": "HIGH", "verdict": "HOLDS", "source_url": "https://example.com", "reason": "why"}},
    {{"text": "objection text", "severity": "LOW", "verdict": "FAILS", "source_url": "", "reason": "why"}},
    {{"text": "objection text", "severity": "MED", "verdict": "HOLDS", "source_url": "https://example.com", "reason": "why"}}
  ],
  "score": "3/5",
  "verdict": "CONTESTED"
}}
verdict must be one of: ROBUST (0-1 held), CONTESTED (2-3 held), FRAGILE (4-5 held)"""
    try:
        raw = await call_gemini_json(prompt, "You are a strict JSON adversarial evaluator.")
        data = json.loads(raw)
        objections = data.get("objections", [])[:5]
        holds = sum(1 for o in objections if (o.get("verdict", "").upper() == "HOLDS"))
        data["score"] = data.get("score") or f"{holds}/5"
        if not data.get("verdict"):
            data["verdict"] = "ROBUST" if holds <= 1 else ("CONTESTED" if holds <= 3 else "FRAGILE")
        return data
    except Exception as e:
        return {"error": str(e), "objections": [], "score": "0/5", "verdict": "UNKNOWN"}


@app.post("/drift/track")
async def drift_track(payload: Dict[str, Any]):
    session_id = payload.get("session_id")
    user_email = payload.get("user_email")
    days = int(payload.get("days", 30))
    history = await get_session_history(session_id) if session_id else []
    if not history:
        return {"error": "Session not found"}
    session = history[-1]
    check_date = (datetime.now() + timedelta(days=days)).isoformat()
    item = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_email": user_email,
        "original_query": session.get("query", ""),
        "original_confidence": _normalize_confidence_score(session.get("confidence_score", 0)),
        "check_date": check_date,
        "days": days,
        "created_at": datetime.now().isoformat(),
    }
    await insert_feature_item("drift_tracker", item)
    ADV_STORE["drift_tracker"].append(item)
    return {"success": True, "check_date": check_date, "days": days}


@app.get("/drift/check/{session_id}")
async def drift_check(session_id: str):
    tracked = await get_feature_item("drift_tracker", "session_id", session_id)
    if not tracked:
        tracked = next((d for d in ADV_STORE["drift_tracker"] if d.get("session_id") == session_id), None)
    if not tracked:
        return {"error": "Not being tracked"}
    original_query = tracked.get("original_query", "")
    original_confidence = float(tracked.get("original_confidence") or 0)
    raw_sources = await search_and_scrape(query=original_query, num_sources=5)
    new_confidence = min(100.0, max(0.0, original_confidence + len(raw_sources) * 1.5 - 4))
    delta = round(new_confidence - original_confidence, 1)
    verdict = "STABLE"
    if abs(delta) >= 15:
        verdict = "MAJOR_SHIFT"
    elif abs(delta) >= 5:
        verdict = "DRIFTED"
    return {
        "original_confidence": original_confidence,
        "new_confidence": new_confidence,
        "confidence_delta": delta,
        "verdict": verdict,
        "checked_at": datetime.now().isoformat(),
        "query": original_query,
    }


@app.post("/research/laundering-check")
async def laundering_check(payload: Dict[str, Any]):
    sources = payload.get("sources", [])
    if not sources:
        return {"laundering_detected": False, "copy_count": 0, "propagation_tree": []}
    claims = [s.get("title", "") + " " + s.get("url", "") for s in sources]
    raw = await call_gemini_json(
        f"Do these sources say essentially the same claim in different words? {json.dumps(claims)}\n"
        "Return: {\"similarity_score\":0-100,\"likely_origin\":\"url|null\",\"laundering_detected\":bool,\"propagation_pattern\":\"...\"}",
        "You detect narrative laundering.",
    )
    out = _safe_json_loads(raw, {})
    sim = out.get("similarity_score", 0)
    sorted_sources = sorted(sources, key=lambda x: x.get("date", "9999-12-31"))
    origin = sorted_sources[0] if sorted_sources else {}
    detected = bool(out.get("laundering_detected")) or (len(sources) >= 5 and sim >= 80)
    tree = []
    prev = None
    for s in sorted_sources:
        tree.append({"url": s.get("url"), "date": s.get("date", ""), "copied_from": prev})
        prev = s.get("url")
    return {
        "laundering_detected": detected,
        "origin_url": out.get("likely_origin") or origin.get("url", ""),
        "origin_date": origin.get("date", ""),
        "copy_count": max(0, len(sources) - 1),
        "propagation_tree": tree,
    }


@app.get("/user/rank")
async def user_rank(user_email: str = QueryParam(...)):
    user = await get_feature_item("users", "email", user_email)
    if not user:
        user = ADV_STORE["users"].get(user_email, {"email": user_email, "query_count": 0, "avg_confidence": 0, "xp": 0})
    rf = _rank_for_queries(user["query_count"])
    user["rank"] = rf["rank"]
    ADV_STORE["users"][user_email] = user
    await upsert_feature_item("users", user)
    return {"rank": user["rank"], "xp": user["xp"], "queries": user["query_count"], "next_rank_at": rf["next_rank_at"]}


@app.get("/leaderboard")
async def leaderboard():
    users_table = await list_feature_items("users", limit=500)
    users = users_table if users_table else list(ADV_STORE["users"].values())
    users = sorted(users, key=lambda x: x.get("query_count", 0), reverse=True)[:10]
    return {"leaders": users}


@app.post("/factwars/create")
async def factwars_create(payload: Dict[str, Any]):
    war_id = str(uuid.uuid4())[:8]
    ADV_STORE["fact_wars"][war_id] = {
        "id": war_id,
        "topic": payload.get("topic", ""),
        "player1": payload.get("user_email", "p1@local"),
        "player2": None,
        "report1": None,
        "report2": None,
        "votes": {"player1": 0, "player2": 0},
        "verdict": "PENDING",
    }
    await insert_feature_item("factwars", ADV_STORE["fact_wars"][war_id])
    return {"war_id": war_id}


@app.post("/factwars/join/{war_id}")
async def factwars_join(war_id: str, payload: Dict[str, Any]):
    war = ADV_STORE["fact_wars"].get(war_id)
    if not war:
        raise HTTPException(status_code=404, detail="war not found")
    war["player2"] = payload.get("user_email", "p2@local")
    await upsert_feature_item("factwars", war)
    return {"ok": True, "war": war}


@app.post("/factwars/submit/{war_id}")
async def factwars_submit(war_id: str, payload: Dict[str, Any]):
    war = ADV_STORE["fact_wars"].get(war_id)
    if not war:
        raise HTTPException(status_code=404, detail="war not found")
    side = payload.get("side", "player1")
    war["report1" if side == "player1" else "report2"] = payload.get("report")
    if war["report1"] and war["report2"]:
        war["verdict"] = "PLAYER 1" if len((war["report1"] or {}).get("citations", [])) >= len((war["report2"] or {}).get("citations", [])) else "PLAYER 2"
    await upsert_feature_item("factwars", war)
    return {"ok": True, "war": war}


@app.get("/factwars/{war_id}")
async def factwars_get(war_id: str):
    war = await get_feature_item("factwars", "id", war_id)
    if not war:
        war = ADV_STORE["fact_wars"].get(war_id)
    if not war:
        raise HTTPException(status_code=404, detail="war not found")
    return war


@app.post("/factwars/{war_id}/vote")
async def factwars_vote(war_id: str, payload: Dict[str, Any]):
    war = ADV_STORE["fact_wars"].get(war_id)
    if not war:
        raise HTTPException(status_code=404, detail="war not found")
    side = payload.get("side", "player1")
    if side in war["votes"]:
        war["votes"][side] += 1
    await upsert_feature_item("factwars", war)
    return {"ok": True, "votes": war["votes"]}


@app.post("/research/darkweb-pulse")
async def darkweb_pulse(payload: Dict[str, Any]):
    q = payload.get("query", "")
    hn_url = f"https://hn.algolia.com/api/v1/search?query={q}"
    hn_data = []
    try:
        with urllib.request.urlopen(hn_url, timeout=5) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            hn_data = [{"title": h.get("title"), "url": h.get("url")} for h in (raw.get("hits") or [])[:5]]
    except Exception:
        pass
    return {"reddit_signals": [], "hn_signals": hn_data, "emerging_narratives": [f"Early chatter around: {q}"]}


@app.post("/monitor/live")
async def monitor_live(payload: Dict[str, Any]):
    monitor_id = str(uuid.uuid4())[:8]
    ADV_STORE["live_monitors"][monitor_id] = {
        "id": monitor_id,
        "event_name": payload.get("event_name"),
        "event_keywords": payload.get("event_keywords", []),
        "check_interval_minutes": payload.get("check_interval_minutes", 5),
        "created_at": _utc_now_iso(),
    }
    await insert_feature_item("live_monitors", ADV_STORE["live_monitors"][monitor_id])
    return {"monitor_id": monitor_id, "live": True}


@app.post("/monitor/live-check")
async def live_monitor_check(payload: dict):
    query = payload.get("query", "")
    previous = payload.get("previous", [])
    try:
        results = await search_and_scrape(query=query, num_sources=5)
        prev_texts = [p.get("text", "") for p in previous]
        signals = []
        for r in results:
            text = (getattr(r, "content", "") or getattr(r, "title", ""))[:200]
            url = getattr(r, "url", "")
            is_new = not any(text[:80] in p for p in prev_texts)
            if text:
                signals.append({"text": text[:200], "url": url, "type": "NEW" if is_new else "UPDATE"})
        return {"signals": signals[:5]}
    except Exception as e:
        return {"signals": [], "error": str(e)}


@app.post("/research/notarize/{session_id}")
async def notarize_report(session_id: str, payload: Dict[str, Any] | None = None):
    history = await get_session_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="session not found")
    report_json = json.dumps(history[-1], sort_keys=True)
    sha = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
    row = {"session_id": session_id, "hash": sha, "timestamp": _utc_now_iso(), "user_email": (payload or {}).get("user_email", "anonymous@local")}
    ADV_STORE["notarizations"][sha] = row
    await upsert_feature_item("notarizations", row)
    return {"hash": sha, "timestamp": row["timestamp"], "verify_url": f"/verify/{sha}"}


@app.get("/verify/{hash_value}")
async def verify_hash(hash_value: str):
    row = await get_feature_item("notarizations", "hash", hash_value)
    if not row:
        row = ADV_STORE["notarizations"].get(hash_value)
    if not row:
        return {"valid": False, "original_timestamp": None, "session_id": None}
    return {"valid": True, "original_timestamp": row["timestamp"], "session_id": row["session_id"]}


@app.post("/verify/identity")
async def verify_identity(payload: Dict[str, Any]):
    row = {
        "id": str(uuid.uuid4())[:8],
        "user_email": payload.get("user_email"),
        "credential_type": payload.get("credential_type"),
        "institution": payload.get("institution"),
        "proof_url": payload.get("proof_url"),
        "status": "PENDING",
        "created_at": _utc_now_iso(),
    }
    await insert_feature_item("identity_requests", row)
    ADV_STORE["identity_requests"].append(row)
    return {"ok": True, "request": row}


<<<<<<< HEAD
@app.get("/fact-wars")
async def serve_fact_wars():
    try:
        current_dir = Path(__file__).resolve().parent
        file_path = current_dir / "static" / "fact_wars.html"
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="fact_wars.html missing")
=======

>>>>>>> 3e9b7f153fab2a6ca9c15927c3d2e4e00d8cc4db


@app.post("/factwars/judge")
async def judge_fact_war(payload: dict):
    for_report = payload.get("for_report", {})
    against_report = payload.get("against_report", {})
    for_query = payload.get("for_query", "")
    against_query = payload.get("against_query", "")

    for_conf = float(for_report.get("confidence_score") or 0)
    against_conf = float(against_report.get("confidence_score") or 0)
    for_sources = len(for_report.get("sources") or for_report.get("citations") or [])
    against_sources = len(against_report.get("sources") or against_report.get("citations") or [])

    for_score = round(for_conf * 0.6 + min(for_sources * 2, 40))
    against_score = round(against_conf * 0.6 + min(against_sources * 2, 40))
    winner = "for" if for_score >= against_score else "against"

    prompt = f"""Two sides presented research on opposing positions.
FOR ({for_query}): confidence {for_conf}%, {for_sources} sources, score {for_score}
AGAINST ({against_query}): confidence {against_conf}%, {against_sources} sources, score {against_score}
Winner: {winner}
Write ONE sentence explaining why the winner had stronger evidence. Be specific."""
    try:
        reason = await call_gemini_json(prompt, "Return plain text only.")
        reason = reason.strip().strip('"')
    except Exception:
        reason = f"The {winner} side presented stronger evidence with higher source confidence."

    return {"winner": winner, "for_score": for_score, "against_score": against_score, "reason": reason}

import os
from fastapi.staticfiles import StaticFiles

if os.path.isdir("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")