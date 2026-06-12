"""ET engine: diagnose GPU idleness from PyTorch Profiler traces."""

from gpu_doctor_engine.diagnose import diagnose, diagnose_with_stats
from gpu_doctor_engine.ingest import load_trace
from gpu_doctor_engine.types import Diagnosis, Event, Trace, Verdict

__version__ = "0.3.0"
__all__ = [
    "diagnose",
    "diagnose_with_stats",
    "load_trace",
    "Diagnosis",
    "Event",
    "Trace",
    "Verdict",
]
