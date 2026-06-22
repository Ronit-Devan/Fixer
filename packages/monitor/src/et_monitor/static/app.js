"use strict";

// Dependency-free dashboard: polls /api/state and renders the UI + canvas
// charts. No CDN, no build; works fully offline next to llama-server.

const POLL_MS = 1000;
// Restrained palette for a light theme; readable on white, not neon.
const COLORS = { ok: "#15803d", info: "#111827", warn: "#b45309", crit: "#b91c1c" };

function $(id) { return document.getElementById(id); }
function fmt(v, digits = 0, suffix = "") {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return Number(v).toFixed(digits) + suffix;
}
function money(v) {
  if (v === null || v === undefined) return "-";
  return "$" + Number(v).toLocaleString(undefined, { maximumFractionDigits: v < 100 ? 2 : 0 });
}

// ---- tiny canvas line chart with gradient area fill (offline, no libs) ----
function drawChart(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  // Use the CSS layout size (stable). Reading canvas.height here would feed
  // back the value we set last frame and the canvas would grow every redraw.
  const cssW = canvas.clientWidth || 480;
  const cssH = canvas.clientHeight || 150;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 32, r: 10, t: 10, b: 16 };
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;
  const clean = series.filter((v) => v != null);
  const max = opts.max != null ? opts.max : Math.max(1, ...clean) * 1.15;
  const color = opts.color || COLORS.info;

  // grid + y labels
  ctx.strokeStyle = "#eef0f2";
  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px ui-sans-serif, sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 2; i++) {
    const y = pad.t + (h * i) / 2;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + w, y); ctx.stroke();
    ctx.fillText(fmt(max * (1 - i / 2), 0), 3, y + 3);
  }
  if (series.length < 2) return;

  const n = series.length;
  const xAt = (i) => pad.l + (w * i) / (n - 1);
  const yAt = (v) => pad.t + h * (1 - Math.min(v, max) / max);

  // build path points
  const pts = [];
  for (let i = 0; i < n; i++) if (series[i] != null) pts.push([xAt(i), yAt(series[i]), i]);
  if (pts.length < 2) return;

  // area fill
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + h);
  grad.addColorStop(0, color + "1f");
  grad.addColorStop(1, color + "00");
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pad.t + h);
  pts.forEach((p) => ctx.lineTo(p[0], p[1]));
  ctx.lineTo(pts[pts.length - 1][0], pad.t + h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // line
  ctx.beginPath();
  pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();

  // last dot
  const last = pts[pts.length - 1];
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(last[0], last[1], 3, 0, Math.PI * 2); ctx.fill();
}

let started = false;

function render(state) {
  const { snapshot, diagnosis, history } = state;
  const snap = snapshot.latest;
  started = true;

  // connection + header pills
  $("conn-dot").className = "dot live";
  $("conn-text").textContent = "live";
  $("pill-conn").classList.add("on");
  $("pill-backend").textContent = "backend: " + snapshot.backend;
  $("pill-backend").classList.add("on");
  const llamaOn = snap && snap.llama_reachable;
  $("pill-llama").textContent = "llama-server: " + (llamaOn ? "connected" : "not found");
  $("pill-llama").classList.toggle("on", !!llamaOn);
  if (snap) $("gpu-name").textContent = snap.gpu_name || "-";

  // verdict
  const card = $("verdict-card");
  card.className = "card verdict " + (diagnosis.severity || "info");
  $("verdict-badge").textContent = (diagnosis.verdict || "").replace(/_/g, " ");
  $("verdict-conf").textContent = diagnosis.confidence
    ? `${Math.round(diagnosis.confidence * 100)}% confidence` : "";
  $("verdict-title").textContent = diagnosis.title || "-";
  $("verdict-summary").textContent = diagnosis.summary || "";
  const recs = diagnosis.recommendations || [];
  $("recs-wrap").hidden = recs.length === 0;
  const ul = $("verdict-recs");
  ul.innerHTML = "";
  recs.forEach((r) => { const li = document.createElement("li"); li.textContent = r; ul.appendChild(li); });

  // cost panel
  const s = snapshot.session;
  const idlePct = (s.idle_fraction || 0) * 100;
  $("idle-frac").textContent = fmt(idlePct, 0, "%");
  $("idle-bar").style.width = Math.min(100, idlePct) + "%";
  if (s.gpu_hourly_usd > 0) {
    $("wasted-so-far").textContent = money(s.wasted_usd_so_far);
    $("proj-monthly").textContent = money(s.projected_monthly_idle_usd);
    $("cost-note").textContent = `based on $${s.gpu_hourly_usd}/GPU-hr · estimate, not a bill`;
  } else {
    $("wasted-so-far").textContent = "-";
    $("proj-monthly").textContent = "-";
    $("cost-note").textContent = "restart with --gpu-price <$/hr> to estimate idle cost";
  }
  $("uptime").textContent = `session ${fmt((s.uptime_s || 0) / 60, 1)} min · idle ${fmt(s.idle_seconds || 0, 0)}s`;

  // tiles
  if (snap) {
    $("t-util").textContent = fmt(snap.util_pct, 0, "%");
    $("t-mem").textContent = snap.mem_used_ratio != null ? fmt(snap.mem_used_ratio * 100, 0, "%") : "-";
    $("t-mem-sub").textContent = (snap.mem_used_mb != null && snap.mem_total_mb)
      ? `${fmt(snap.mem_used_mb / 1024, 1)} / ${fmt(snap.mem_total_mb / 1024, 0)} GB` : "";
    $("t-power").textContent = fmt(snap.power_w, 0, " W");
    $("t-tps").textContent = fmt(snap.gen_tokens_per_s, 0);
    $("t-req").textContent = snap.requests_processing != null ? fmt(snap.requests_processing, 0) : "-";
    $("t-kv").textContent = snap.kv_cache_usage_ratio != null ? fmt(snap.kv_cache_usage_ratio * 100, 0, "%") : "-";
  }

  // per-GPU strip (only when the box has more than one card)
  const gpus = (snapshot.gpus || []);
  const strip = $("gpu-strip");
  if (gpus.length > 1) {
    strip.hidden = false;
    const host = snapshot.host_label ? snapshot.host_label + " · " : "";
    $("gpu-cards").innerHTML = gpus.map((g) => {
      const u = g.latest && g.latest.util_pct != null ? Math.round(g.latest.util_pct) + "%" : "-";
      const d = g.diagnosis || {};
      const sev = d.severity || "info";
      const v = (d.verdict || "").replace(/_/g, " ");
      const pred = d.predicted ? " ⚠︎" : "";
      return `<div class="gpu-chip ${sev}"><b>${host}GPU ${g.index}</b>` +
             `<span>${u}</span><em>${v}${pred}</em></div>`;
    }).join("");
  } else {
    strip.hidden = true;
  }

  // charts
  const tail = history.slice(-180);
  drawChart($("chart-util"), tail.map((x) => x.util_pct), { max: 100, color: COLORS.info });
  drawChart($("chart-mem"), tail.map((x) => x.mem_used_ratio != null ? x.mem_used_ratio * 100 : null), { max: 100, color: COLORS.ok });
  drawChart($("chart-tps"), tail.map((x) => x.gen_tokens_per_s), { color: COLORS.warn });
  drawChart($("chart-kv"), tail.map((x) => x.kv_cache_usage_ratio != null ? x.kv_cache_usage_ratio * 100 : null), { max: 100, color: COLORS.crit });
}

async function poll() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    render(await r.json());
  } catch (e) {
    $("conn-dot").className = "dot stale";
    $("conn-text").textContent = started ? "reconnecting…" : "starting…";
  }
}

// ---- remediation panel (only renders when the layer is enabled) ----------
let remModeDirty = false;
async function pollRemediation() {
  let state;
  try {
    state = await (await fetch("/api/remediation/state", { cache: "no-store" })).json();
  } catch { return; }
  const panel = $("remediation-panel");
  if (!state || !state.enabled) { panel.hidden = true; return; }
  panel.hidden = false;

  const sel = $("rem-mode-select");
  if (!remModeDirty && document.activeElement !== sel) sel.value = state.mode;
  $("rem-state").textContent = "state: " + state.state;
  $("rem-breaker").textContent = "breaker: " + state.breaker_state;
  $("rem-breaker").classList.toggle("warn-pill", state.breaker_state !== "closed");

  // pending approvals
  const ap = state.pending_approvals || [];
  $("rem-approvals").hidden = ap.length === 0;
  const al = $("rem-approval-list");
  al.innerHTML = "";
  ap.forEach((a) => {
    const li = document.createElement("li");
    li.innerHTML =
      `<div><strong>${a.root_cause.replace(/_/g, " ")}</strong> — ${a.summary}` +
      `<code>${a.command_preview || ""}</code></div>`;
    const ok = document.createElement("button");
    ok.textContent = "Approve"; ok.className = "rem-approve";
    ok.onclick = () => act(`/api/remediation/approvals/${a.id}/approve`);
    const no = document.createElement("button");
    no.textContent = "Reject"; no.className = "rem-reject";
    no.onclick = () => act(`/api/remediation/approvals/${a.id}/reject`);
    li.appendChild(ok); li.appendChild(no);
    al.appendChild(li);
  });

  // audit tail
  try {
    const audit = await (await fetch("/api/remediation/audit", { cache: "no-store" })).json();
    const list = $("rem-audit-list");
    list.innerHTML = "";
    (audit.records || []).slice(-8).reverse().forEach((r) => {
      const li = document.createElement("li");
      li.textContent = `${r.phase} · ${r.root_cause} · ${r.action_kind || ""} · ${r.decision}`;
      list.appendChild(li);
    });
  } catch { /* ignore */ }
}

async function act(url) {
  try { await fetch(url, { method: "POST" }); } catch { /* ignore */ }
  pollRemediation();
}

$("rem-mode-select").addEventListener("change", async (e) => {
  remModeDirty = true;
  await fetch("/api/remediation/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: e.target.value }),
  }).catch(() => {});
  remModeDirty = false;
  pollRemediation();
});

poll();
pollRemediation();
setInterval(poll, POLL_MS);
setInterval(pollRemediation, 2000);
window.addEventListener("resize", () => { if (started) poll(); });
