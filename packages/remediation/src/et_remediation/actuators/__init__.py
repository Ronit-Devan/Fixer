"""Backend actuators behind one common interface.

  * ``DataCenterActuator`` — nvidia-smi / NVML / DCGM tuning, MPS/MIG, freeing
    stale memory, killing orphaned PIDs; K8s/Slurm drain for the disruptive path.
  * ``LlamaCppActuator`` — runtime-safe nudges, and (disruptive, approval-gated)
    restart of llama-server with tuned flags after a request-drain.
  * ``FakeActuator`` — records calls and drives a fake telemetry model so the
    whole guarded loop is testable without a GPU.
"""

from et_remediation.actuators.base import (
    Actuator,
    ActuationState,
    CommandRunner,
)
from et_remediation.actuators.datacenter import DataCenterActuator
from et_remediation.actuators.fake import FakeActuator, FakeTelemetryModel
from et_remediation.actuators.llamacpp import LlamaCppActuator

__all__ = [
    "Actuator",
    "ActuationState",
    "CommandRunner",
    "DataCenterActuator",
    "FakeActuator",
    "FakeTelemetryModel",
    "LlamaCppActuator",
]
