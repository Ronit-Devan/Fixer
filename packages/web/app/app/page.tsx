"use client";

import {
  useState,
  useRef,
  useCallback,
  DragEvent,
  ChangeEvent,
} from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Verdict =
  | "healthy"
  | "dataloader_bound"
  | "pcie_bound"
  | "kernel_launch_bound"
  | "nccl_bound"
  | "checkpoint_bound"
  | "sync_bound"
  | "unknown";

interface Decision {
  rule: string;
  fired: boolean;
  value: number | string;
  threshold: number | string;
  note?: string;
}

interface AnalysisResult {
  verdict: Verdict;
  confidence: number;
  summary: string;
  evidence: string[];
  recommended_actions: string[];
  metrics: Record<string, number | undefined>;
  stats: {
    decisions: Decision[];
    engine_version: string;
    [key: string]: unknown;
  };
  trace_info: {
    event_count: number;
    duration_ms: number;
    filename: string;
  };
}

const VERDICT_COLOR: Record<Verdict, string> = {
  healthy: "text-emerald-400",
  dataloader_bound: "text-amber-400",
  pcie_bound: "text-amber-400",
  kernel_launch_bound: "text-amber-400",
  nccl_bound: "text-amber-400",
  checkpoint_bound: "text-amber-400",
  sync_bound: "text-amber-400",
  unknown: "text-slate-400",
};

const VERDICT_LABEL: Record<Verdict, string> = {
  healthy: "HEALTHY",
  dataloader_bound: "DATALOADER_BOUND",
  pcie_bound: "PCIE_BOUND",
  kernel_launch_bound: "KERNEL_LAUNCH_BOUND",
  nccl_bound: "NCCL_BOUND",
  checkpoint_bound: "CHECKPOINT_BOUND",
  sync_bound: "SYNC_BOUND",
  unknown: "UNKNOWN",
};

function validateFile(f: File): string | null {
  if (!f.name.endsWith(".json") && !f.name.endsWith(".json.gz")) {
    return "File must be a .json or .json.gz PyTorch Profiler trace.";
  }
  if (f.size > 50 * 1024 * 1024) {
    return "File exceeds the 50 MB limit.";
  }
  return null;
}

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [decisionsOpen, setDecisionsOpen] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (f: File) => {
    const err = validateFile(f);
    if (err) {
      setValidationError(err);
      return;
    }
    setValidationError(null);
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", f);
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error ${res.status}: ${text}`);
      }
      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const reset = () => {
    setLoading(false);
    setResult(null);
    setError(null);
    setValidationError(null);
    setDecisionsOpen(false);
    setMetricsOpen(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const idle = !loading && !result && !error;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* ── Header ── */}
      <header className="sticky top-0 z-10 bg-slate-950/95 border-b border-slate-800 px-6 py-3 flex items-center justify-between backdrop-blur-sm">
        <span className="font-semibold text-slate-100 text-sm tracking-tight">
          ET — diagnose why your GPU is idle
        </span>
        <a
          href="https://github.com/devan-p/ET"
          target="_blank"
          rel="noopener noreferrer"
          className="text-slate-500 text-xs hover:text-slate-300 transition-opacity duration-300"
        >
          v0.3 · github.com/devan-p/ET
        </a>
      </header>

      {/* ── Main ── */}
      <main className="flex-1 flex flex-col items-center px-6 py-10 w-full max-w-3xl mx-auto">
        {/* ── Idle: upload zone ── */}
        {idle && (
          <div className="w-full opacity-100 transition-opacity duration-300">
            <div
              className={`w-full border-2 border-dashed rounded-xl flex flex-col items-center justify-center cursor-pointer transition-colors duration-200 ${
                isDragging
                  ? "border-sky-400 bg-slate-900/80"
                  : "border-slate-700 bg-slate-900 hover:border-slate-500"
              }`}
              style={{ minHeight: "40vh" }}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <span className="text-slate-300 text-lg font-medium text-center px-6">
                Drop a PyTorch Profiler trace (.json or .json.gz)
              </span>
              <span className="text-slate-500 text-sm mt-2">
                or click to browse · max 50 MB
              </span>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.json.gz"
              className="hidden"
              onChange={handleInputChange}
            />
            {validationError && (
              <p className="mt-3 text-red-400 text-sm">{validationError}</p>
            )}
          </div>
        )}

        {/* ── Loading ── */}
        {loading && (
          <div className="flex flex-col items-center justify-center gap-4 py-32 opacity-100 transition-opacity duration-300">
            <div className="w-8 h-8 border-2 border-slate-700 border-t-sky-400 rounded-full animate-spin" />
            <span className="text-slate-400 text-sm">Analyzing trace...</span>
          </div>
        )}

        {/* ── Error ── */}
        {error && !loading && (
          <div className="w-full border border-red-400/40 bg-red-400/5 rounded-xl p-5 opacity-100 transition-opacity duration-300">
            <p className="text-red-400 text-sm font-semibold mb-1">
              Analysis failed
            </p>
            <p className="text-slate-300 text-sm font-mono break-all">
              {error}
            </p>
            <button
              onClick={reset}
              className="mt-4 text-sm text-slate-400 hover:text-slate-200 transition-colors duration-200"
            >
              Try again →
            </button>
          </div>
        )}

        {/* ── Results ── */}
        {result && !loading && (
          <div className="w-full flex flex-col gap-4 opacity-100 transition-opacity duration-300">
            {/* Reset */}
            <button
              onClick={reset}
              className="self-start text-xs text-slate-500 hover:text-slate-300 transition-colors duration-200 border border-slate-800 rounded-md px-3 py-1.5"
            >
              Analyze another trace →
            </button>

            {/* Card 1 — Verdict */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <div
                className={`text-4xl font-bold tracking-tight font-mono ${VERDICT_COLOR[result.verdict]}`}
              >
                {VERDICT_LABEL[result.verdict]}
              </div>
              <div className="text-slate-300 text-sm mt-2">
                Confidence: {Math.round(result.confidence * 100)}%
              </div>
              <p className="text-slate-200 text-sm mt-3 leading-relaxed">
                {result.summary}
              </p>
            </div>

            {/* Card 2 — Evidence */}
            {result.evidence.length > 0 && (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                  Evidence
                </h2>
                <ul className="flex flex-col gap-2">
                  {result.evidence.map((item, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-sm text-slate-200"
                    >
                      <span className="text-slate-500 mt-0.5 select-none">
                        ·
                      </span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Card 3 — Recommended actions */}
            {result.recommended_actions.length > 0 && (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                  Recommended Actions
                </h2>
                <ol className="flex flex-col gap-3">
                  {result.recommended_actions.map((action, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-3 text-sm text-slate-200"
                    >
                      <span className="text-slate-500 tabular-nums w-4 shrink-0 pt-0.5">
                        {i + 1}.
                      </span>
                      <span className="flex-1">{action}</span>
                      {i === 0 && (
                        <span className="shrink-0 text-[10px] font-semibold uppercase tracking-wider text-amber-400 border border-amber-400/30 rounded px-1.5 py-0.5 mt-0.5">
                          Top fix
                        </span>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {/* Card 4 — Trace info */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl px-6 py-4">
              <p className="text-slate-400 text-sm">
                {result.trace_info.filename}
                {" · "}
                {result.trace_info.event_count.toLocaleString()} events
                {" · "}
                {result.trace_info.duration_ms.toLocaleString()} ms
                {" · "}
                engine {result.stats.engine_version}
              </p>
            </div>

            {/* Card 5 — Detector decisions (collapsible) */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
              <button
                onClick={() => setDecisionsOpen((v) => !v)}
                className="w-full flex items-center justify-between px-6 py-4 text-sm text-slate-400 hover:text-slate-200 transition-colors duration-200"
              >
                <span>Show engine reasoning</span>
                <span
                  className={`text-xs transition-transform duration-200 ${
                    decisionsOpen ? "rotate-180" : ""
                  }`}
                >
                  ▼
                </span>
              </button>
              {decisionsOpen && (
                <div className="px-6 pb-5 overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="text-slate-500 text-left border-b border-slate-800">
                        <th className="pb-2 pr-6 font-medium">Rule</th>
                        <th className="pb-2 pr-6 font-medium">Value</th>
                        <th className="pb-2 pr-6 font-medium">Threshold</th>
                        <th className="pb-2 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.stats.decisions.map((d, i) => (
                        <tr
                          key={i}
                          className="border-b border-slate-800/50 last:border-0"
                        >
                          <td className="py-2 pr-6 text-slate-300">
                            {d.rule}
                          </td>
                          <td className="py-2 pr-6 text-slate-300">
                            {String(d.value)}
                          </td>
                          <td className="py-2 pr-6 text-slate-300">
                            {String(d.threshold)}
                          </td>
                          <td className="py-2">
                            {d.fired ? (
                              <span className="text-emerald-400">✓</span>
                            ) : (
                              <span className="text-slate-500">✗</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Card 6 — Raw metrics (collapsible) */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
              <button
                onClick={() => setMetricsOpen((v) => !v)}
                className="w-full flex items-center justify-between px-6 py-4 text-sm text-slate-400 hover:text-slate-200 transition-colors duration-200"
              >
                <span>Show raw metrics</span>
                <span
                  className={`text-xs transition-transform duration-200 ${
                    metricsOpen ? "rotate-180" : ""
                  }`}
                >
                  ▼
                </span>
              </button>
              {metricsOpen && (
                <div className="px-6 pb-5">
                  <pre className="text-xs font-mono text-slate-300 overflow-auto max-h-72 whitespace-pre-wrap break-all">
                    {JSON.stringify(
                      { metrics: result.metrics, stats: result.stats },
                      null,
                      2
                    )}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
