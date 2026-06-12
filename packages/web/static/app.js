const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const verdictCard = document.getElementById("verdict-card");
const verdictText = document.getElementById("verdict-text");
const confidenceEl = document.getElementById("confidence");
const summaryEl = document.getElementById("summary");
const evidenceList = document.getElementById("evidence-list");
const metricsDl = document.getElementById("metrics-dl");
const actionsPanel = document.getElementById("actions-panel");
const actionsList = document.getElementById("actions-list");
const attribBody = document.getElementById("attrib-body");
const decisionsBody = document.getElementById("decisions-body");
const traceMeta = document.getElementById("trace-meta");

function setStatus(message, isError = false) {
  statusEl.hidden = !message;
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function verdictClass(verdict) {
  if (verdict === "healthy") return "healthy";
  if (verdict === "unknown") return "unknown";
  return "bound";
}

function formatVerdict(verdict) {
  return verdict.replace(/_/g, " ").toUpperCase();
}

function formatMetricValue(key, value) {
  if (typeof value !== "number") return String(value);
  if (key.endsWith("_us") || key.includes("_us")) {
    return value >= 1000 ? `${(value / 1000).toFixed(1)} ms` : `${value.toFixed(0)} µs`;
  }
  if (key.includes("util") || key.includes("share") || key.includes("ratio") || key.includes("fraction")) {
    return value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

function renderExplainStats(stats) {
  attribBody.innerHTML = "";
  const idleUs = stats.idle_us || 0;
  const rows = [
    ["GPU idle (no activity)", `${(idleUs / 1000).toFixed(1)} ms`],
    [
      "DataLoader overlap with idle",
      `${((stats.dataloader_us || 0) / 1000).toFixed(1)} ms (${(((stats.dataloader_us || 0) / Math.max(idleUs, 1)) * 100).toFixed(0)}% of idle)`,
    ],
    [
      "NCCL overlap with idle",
      `${((stats.nccl_us || 0) / 1000).toFixed(1)} ms (${(((stats.nccl_us || 0) / Math.max(idleUs, 1)) * 100).toFixed(0)}% of idle)`,
    ],
    [
      "Memcpy overlap with idle",
      `${((stats.memcpy_us || 0) / 1000).toFixed(1)} ms (${(((stats.memcpy_us || 0) / Math.max(idleUs, 1)) * 100).toFixed(0)}% of idle)`,
    ],
    [
      "Checkpoint overlap with idle",
      `${((stats.checkpoint_us || 0) / 1000).toFixed(1)} ms (${(((stats.checkpoint_us || 0) / Math.max(idleUs, 1)) * 100).toFixed(0)}% of idle)`,
    ],
    [
      "Sync overlap with idle",
      `${((stats.sync_us || 0) / 1000).toFixed(1)} ms (${(((stats.sync_us || 0) / Math.max(idleUs, 1)) * 100).toFixed(0)}% of idle)`,
    ],
    ["Average kernel duration", `${(stats.avg_kernel_dur || 0).toFixed(0)} µs`],
    ["Tiny kernel ratio (<50µs)", `${((stats.tiny_kernel_ratio || 0) * 100).toFixed(0)}%`],
    ["Rule fired", stats.rule || "—"],
  ];
  for (const [metric, value] of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<th scope="row">${metric}</th><td>${value}</td>`;
    attribBody.appendChild(tr);
  }
}

function renderDecisions(decisions) {
  decisionsBody.innerHTML = "";
  if (!decisions || !decisions.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="5">No decision log available.</td>`;
    decisionsBody.appendChild(tr);
    return;
  }
  for (const d of decisions) {
    let statusClass = "status-skipped";
    let statusText = "skipped";
    if (d.fired) {
      statusClass = "status-fired";
      statusText = "FIRED (won)";
    } else if (d.passed) {
      statusClass = "status-passed";
      statusText = "passed, lost competition";
    }
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${d.rule}</code></td>
      <td>${Number(d.value).toFixed(2)}</td>
      <td>${Number(d.threshold).toFixed(2)}</td>
      <td class="${statusClass}">${statusText}</td>
      <td>${d.note || ""}</td>
    `;
    decisionsBody.appendChild(tr);
  }
}

function renderResults(data) {
  const verdict = data.verdict;
  verdictCard.className = `verdict-card ${verdictClass(verdict)}`;
  verdictText.textContent = formatVerdict(verdict);
  if (data.confidence == null) {
    confidenceEl.hidden = true;
    confidenceEl.textContent = "";
  } else {
    confidenceEl.hidden = false;
    confidenceEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(0)}%`;
  }
  summaryEl.textContent = data.summary || "";

  evidenceList.innerHTML = "";
  for (const line of data.evidence || []) {
    const li = document.createElement("li");
    li.textContent = line;
    evidenceList.appendChild(li);
  }

  metricsDl.innerHTML = "";
  const metrics = data.metrics || {};
  for (const [key, value] of Object.entries(metrics)) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = formatMetricValue(key, value);
    metricsDl.appendChild(dt);
    metricsDl.appendChild(dd);
  }

  const actions = data.recommended_actions || [];
  actionsPanel.hidden = actions.length === 0;
  actionsList.innerHTML = "";
  for (const action of actions) {
    const li = document.createElement("li");
    li.textContent = action;
    actionsList.appendChild(li);
  }

  renderExplainStats(data.stats || {});
  renderDecisions((data.stats || {}).decisions);

  const info = data.trace_info || {};
  traceMeta.textContent = `${info.filename || "trace"} · ${info.event_count ?? "?"} events · ${(info.duration_ms ?? 0).toFixed(0)} ms`;

  resultsEl.hidden = false;
}

async function analyzeFile(file) {
  if (!file.name.endsWith(".json") && !file.name.endsWith(".json.gz")) {
    setStatus("Please upload a .json or .json.gz trace file.", true);
    return;
  }

  resultsEl.hidden = true;
  setStatus(`Analyzing ${file.name}…`);

  const form = new FormData();
  form.append("file", file, file.name);

  try {
    const res = await fetch("/analyze", { method: "POST", body: form });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    setStatus("");
    renderResults(body);
  } catch (err) {
    setStatus(err.message || "Analysis failed.", true);
  }
}

function onFiles(files) {
  if (files && files.length) analyzeFile(files[0]);
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => onFiles(fileInput.files));

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  onFiles(e.dataTransfer.files);
});
