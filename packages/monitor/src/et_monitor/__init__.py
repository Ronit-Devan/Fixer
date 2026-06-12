"""ET inference monitor; live GPU idleness + llama.cpp attribution as a local web app."""

from et_monitor.alerts import (
    AlertConfig,
    AlertManager,
    SlackNotifier,
    WebhookNotifier,
)
from et_monitor.analyzer import Thresholds, analyze
from et_monitor.state import Monitor, MonitorConfig
from et_monitor.types import Diagnosis, Snapshot, Verdict

__all__ = [
    "analyze",
    "Thresholds",
    "Monitor",
    "MonitorConfig",
    "Diagnosis",
    "Snapshot",
    "Verdict",
    "AlertManager",
    "AlertConfig",
    "SlackNotifier",
    "WebhookNotifier",
]

__version__ = "0.1.0"
