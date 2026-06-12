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
import threading
import time
import webbrowser

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


def _build_monitor(args: argparse.Namespace) -> Monitor:
    config = MonitorConfig(
        interval_s=args.interval,
        gpu_hourly_usd=args.gpu_price,
    )
    alerts = _build_alert_manager(args)
    if args.demo:
        # Shorter window so each scripted phase dominates its own verdict -
        # otherwise the trailing 30s mixes phases and sticky signals (KV, throttle)
        # mask the others. Real boxes keep the 30s default.
        config.window_seconds = 8
        timeline = DemoTimeline()
        return Monitor(
            DemoGpuSampler(timeline),
            DemoLlamaScraper(timeline),  # type: ignore[arg-type]
            config,
            alert_manager=alerts,
        )
    gpu = get_gpu_sampler(force_mock=args.mock)
    llama = None if args.no_llama else LlamaScraper(args.llama_url)
    return Monitor(gpu, llama, config, alert_manager=alerts)


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
