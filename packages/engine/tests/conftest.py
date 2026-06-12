"""Shared pytest fixtures for the gpu-doctor-engine test suite."""

from __future__ import annotations

import pytest

from gpu_doctor_engine.types import Trace

from tests.helpers import make_kernel_event, make_trace


@pytest.fixture
def healthy_trace() -> Trace:
    """A Trace where the GPU is busy 95% of the time."""
    return make_trace(
        events=[make_kernel_event(ts=0, dur=950)],
        duration_us=1000,
        gpu_kernel_time_us=950,
    )
