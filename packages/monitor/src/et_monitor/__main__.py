"""``et-monitor`` entrypoint; one command, opens a dashboard in the browser.

    et-monitor                         # auto-detect GPU, look for llama-server on :8080
    et-monitor --llama-url http://localhost:8081
    et-monitor --gpu-price 0.50        # show $ wasted to idle at $0.50/GPU-hr
    et-monitor --demo                  # scripted timeline, no GPU/model needed
    et-monitor --port 7070 --no-browser
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
import webbrowser
from pathlib import Path

from et_monitor.alerts import (
    AlertConfig,
    AlertManager,
    SlackNotifier,
    WebhookNotifier,
)
from et_monitor.demo import DemoGpuSampler, DemoLlamaScraper, DemoTimeline
from et_monitor.gpu import get_gpu_sampler
from et_monitor.llama import LlamaScraper
from et_monitor.server import create_app
from et_monitor.state import Monitor, MonitorConfig


def _build_alert_manager(args: argparse.Namespace) -> AlertManager | None:
    notifiers = []
    if args.slack_webhook:
        notifiers.append(SlackNotifier(args.slack_webhook))
    if args.webhook:
        notifiers.append(WebhookNotifier(args.webhook))
    if not notifiers:
        return None
    cfg = AlertConfig(
        idle_alert_after_s=args.alert_idle_min * 60.0,
        cooldown_s=args.alert_cooldown_min * 60.0,
        notify_recovery=not args.no_alert_recovery,
        host_label=args.host_label,
    )
    kinds = ", ".join(type(n).__name__ for n in notifiers)
    print(f"  alerts on → {kinds} (idle>{args.alert_idle_min}m, cooldown {args.alert_cooldown_min}m)")
    return AlertManager(notifiers, cfg)


def _build_remediation_factory(args: argparse.Namespace, monitor: Monitor):
    """Build a PER-GPU remediation manager factory, if installed and requested.

    Returns ``callable(gpu_index) -> RemediationManager`` so each card on a
    multi-GPU box gets its own manager (independent breaker / verify / approvals),
    while a shared ``FleetCoordinator`` caps how many GPUs remediate at once
    (blast radius) and a shared config gives one box-wide kill-switch. Optional
    dependency: if ``et_remediation`` isn't installed the monitor runs advise-only.
    """
    if args.remediation_mode == "disabled" and not args.remediation_setup:
        return None
    try:
        from et_remediation import (
            AuditLog,
            CommandRunner,
            DataCenterActuator,
            FleetCoordinator,
            LlamaCppActuator,
            ProtectedWorkload,
            RemediationConfig,
            RemediationManager,
            RemediationMode,
            default_config_path,
            default_registry,
        )
        from et_remediation.setup_ui import run_setup
    except ImportError:
        print("  remediation: et-remediation not installed; running advise-only.")
        return None

    cfg_path = args.remediation_config or str(default_config_path())
    cfg = RemediationConfig.load_or_default(cfg_path)
    if args.remediation_setup:
        cfg = run_setup(path=cfg_path, existing=cfg)
    if args.remediation_mode and args.remediation_mode != "disabled":
        cfg.mode = RemediationMode(args.remediation_mode)
    if args.remediation_audit_log:
        cfg.audit_path = args.remediation_audit_log
    # Production safety: debounce a one-tick verdict (predictive detection gives
    # the lead time to afford it). Honor a higher user-configured value.
    cfg.trigger_debounce = max(cfg.trigger_debounce, 3)

    host = args.host_label or socket.gethostname()

    def _inflight() -> float | None:
        latest = (monitor.snapshot() or {}).get("latest") or {}
        return latest.get("requests_processing")

    # Actuators are stateless command-builders keyed by the request's target, so
    # one set is shared across GPUs; the shared FleetCoordinator bounds blast radius.
    actuators = [
        DataCenterActuator(CommandRunner(execute=True)),
        LlamaCppActuator(CommandRunner(execute=True), requests_inflight=_inflight),
    ]
    fleet = FleetCoordinator(max_concurrent=cfg.caps.max_concurrent_per_node)
    protected = ProtectedWorkload(
        pids=frozenset(cfg.protected_pids), label=cfg.protected_label
    )
    registry = default_registry()

    def _audit_for(gpu_index: int):
        # Each GPU gets its OWN audit sink (distinct JSONL file when a path is
        # set) so concurrent per-GPU appends never interleave/corrupt.
        if not cfg.audit_path:
            return AuditLog()
        p = Path(cfg.audit_path)
        return AuditLog(jsonl_path=str(p.with_name(f"{p.stem}.gpu{gpu_index}{p.suffix}")))

    def factory(gpu_index: int):
        return RemediationManager(
            registry, cfg, actuators,
            audit=_audit_for(gpu_index),
            protected=protected, fleet=fleet,
            node_id=f"{host}:gpu{gpu_index}", gpu_index=gpu_index,
        )

    print(
        f"  remediation -> mode={cfg.mode.value} on {host} "
        f"(per-GPU, fleet cap {fleet.max_concurrent}; kill-switch: et-remediation mode advise)"
    )
    return factory


def _build_monitor(args: argparse.Namespace) -> Monitor:
    config = MonitorConfig(
        interval_s=args.interval,
        gpu_hourly_usd=args.gpu_price,
    )
    alerts = _build_alert_manager(args)
    host_label = args.host_label or socket.gethostname()
    if args.demo:
        # Shorter window so each scripted phase dominates its own verdict -
        # otherwise the trailing 30s mixes phases and sticky signals (KV, throttle)
        # mask the others. Real boxes keep the 30s default.
        config.window_seconds = 8
        timeline = DemoTimeline()
        monitor = Monitor(
            DemoGpuSampler(timeline),
            DemoLlamaScraper(timeline),  # type: ignore[arg-type]
            config,
            alert_manager=alerts,
            host_label=host_label,
        )
    else:
        gpu = get_gpu_sampler(force_mock=args.mock)
        llama = None if args.no_llama else LlamaScraper(args.llama_url)
        monitor = Monitor(gpu, llama, config, alert_manager=alerts, host_label=host_label)
    monitor.remediation_factory = _build_remediation_factory(args, monitor)
    return monitor


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="et-monitor",
        description="Live GPU idleness + llama.cpp inference monitor (local web app).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7070)
    parser.add_argument(
        "--llama-url",
        default="http://localhost:8080",
        help="Base URL of llama-server (expects /metrics). Default :8080.",
    )
    parser.add_argument(
        "--no-llama",
        action="store_true",
        help="Don't scrape llama-server; GPU-only mode.",
    )
    parser.add_argument(
        "--gpu-price",
        type=float,
        default=0.0,
        help="GPU cost in $/hour, to estimate dollars wasted on idle. 0 = off.",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--mock", action="store_true", help="Force mock GPU data (no NVIDIA card)."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Scripted inference timeline; no GPU or model required.",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    # --- alerting ---
    parser.add_argument(
        "--slack-webhook",
        default="",
        help="Slack Incoming Webhook URL; get pinged when the GPU needs attention.",
    )
    parser.add_argument(
        "--webhook",
        default="",
        help="Generic webhook URL; receives the full diagnosis as JSON.",
    )
    parser.add_argument(
        "--alert-idle-min",
        type=float,
        default=10.0,
        help="Alert after the GPU is idle this many minutes (default 10).",
    )
    parser.add_argument(
        "--alert-cooldown-min",
        type=float,
        default=15.0,
        help="Minimum minutes between repeat alerts for the same issue (default 15).",
    )
    parser.add_argument(
        "--no-alert-recovery",
        action="store_true",
        help="Don't send a message when the GPU recovers to healthy.",
    )
    parser.add_argument(
        "--host-label",
        default="",
        help="Label shown in alerts to identify this box (e.g. 'sf-blackwell-01').",
    )
    # --- remediation (auto-actuation) layer ---
    parser.add_argument(
        "--remediation-mode",
        choices=["disabled", "off", "advise", "dry_run", "auto"],
        default="disabled",
        help="Enable the remediation layer in this mode. 'disabled' (default) "
        "leaves it off entirely; 'advise' recommends only; 'auto' applies "
        "non-disruptive fixes through the guarded verify/rollback path.",
    )
    parser.add_argument(
        "--remediation-config",
        default="",
        help="Path to the remediation config JSON (default ~/.et/remediation.json).",
    )
    parser.add_argument(
        "--remediation-setup",
        action="store_true",
        help="Run the interactive remediation setup wizard before starting.",
    )
    parser.add_argument(
        "--remediation-audit-log",
        default="",
        help="Append-only JSONL path for the remediation audit trail.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import uvicorn

    monitor = _build_monitor(args)
    app = create_app(monitor)

    url = f"http://{args.host}:{args.port}/"
    print(f"\n  ET monitor → {url}")
    print("  (Ctrl-C to stop)\n")
    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(1.2), webbrowser.open(url)),
            daemon=True,
        ).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
