(function () {
  "use strict";

  var canvas = document.getElementById("neural-bg");
  if (!canvas) return;

  Object.assign(canvas.style, {
    position: "fixed",
    top: "0",
    left: "0",
    width: "100%",
    height: "100%",
    zIndex: "0",
    pointerEvents: "none"
  });

  var ctx = canvas.getContext("2d");
  if (!ctx) return;

  var w = 0;
  var h = 0;
  var dpr = 1;
  var t0 = performance.now();

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].x = Math.min(Math.max(nodes[i].x, 0), w);
      nodes[i].y = Math.min(Math.max(nodes[i].y, 0), h);
    }
    for (var j = 0; j < nodesFar.length; j++) {
      nodesFar[j].x = Math.min(Math.max(nodesFar[j].x, 0), w);
      nodesFar[j].y = Math.min(Math.max(nodesFar[j].y, 0), h);
    }
  }

  function makeNodes(count, speed, rBase) {
    var arr = [];
    var i;
    for (i = 0; i < count; i++) {
      arr.push({
        x: Math.random() * Math.max(w, 1),
        y: Math.random() * Math.max(h, 1),
        vx: (Math.random() - 0.5) * speed,
        vy: (Math.random() - 0.5) * speed,
        r: rBase * (0.55 + Math.random() * 0.95)
      });
    }
    return arr;
  }

  var nodes = [];
  var nodesFar = [];

  function initNodes() {
    nodes = makeNodes(78, 0.52, 2.4);
    nodesFar = makeNodes(32, 0.22, 1.1);
  }

  resize();
  initNodes();
  window.addEventListener("resize", resize);

  function bounce(n) {
    n.x += n.vx;
    n.y += n.vy;
    if (n.x < 0) {
      n.x = 0;
      n.vx *= -1;
    } else if (n.x > w) {
      n.x = w;
      n.vx *= -1;
    }
    if (n.y < 0) {
      n.y = 0;
      n.vy *= -1;
    } else if (n.y > h) {
      n.y = h;
      n.vy *= -1;
    }
  }

  function drawEdges(arr, maxD, baseA, r, g, b) {
    var i;
    var j;
    var dx;
    var dy;
    var dist;
    var a;
    for (i = 0; i < arr.length; i++) {
      for (j = i + 1; j < arr.length; j++) {
        dx = arr[i].x - arr[j].x;
        dy = arr[i].y - arr[j].y;
        dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxD) {
          a = (1 - dist / maxD) * baseA;
          ctx.strokeStyle = "rgba(" + r + "," + g + "," + b + "," + a + ")";
          ctx.lineWidth = dist < maxD * 0.35 ? 1.25 : 0.85;
          ctx.beginPath();
          ctx.moveTo(arr[i].x, arr[i].y);
          ctx.lineTo(arr[j].x, arr[j].y);
          ctx.stroke();
        }
      }
    }
  }

  function drawNodes(arr, fill, glow) {
    var i;
    var n;
    for (i = 0; i < arr.length; i++) {
      n = arr[i];
      if (glow) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r * 3.2, 0, Math.PI * 2);
        var grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 3.2);
        grad.addColorStop(0, glow);
        grad.addColorStop(1, "rgba(56,139,253,0)");
        ctx.fillStyle = grad;
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
    }
  }

  function frame(now) {
    var t = (now - t0) / 1000;
    var pulse = 0.72 + 0.28 * Math.sin(t * 1.15);
    var pulseFast = 0.85 + 0.15 * Math.sin(t * 2.4);

    ctx.fillStyle = "#080c14";
    ctx.fillRect(0, 0, w, h);

    var k;
    for (k = 0; k < nodesFar.length; k++) bounce(nodesFar[k]);
    for (k = 0; k < nodes.length; k++) bounce(nodes[k]);

    drawEdges(nodesFar, 95, 0.1 * pulse, 56, 100, 180);
    drawNodes(nodesFar, "rgba(88,140,220,0.45)", null);

    drawEdges(nodes, 128, 0.26 * pulse * pulseFast, 56, 166, 253);
    drawNodes(nodes, "rgba(200,230,255,0.95)", "rgba(56,139,253,0.22)");

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
})();
