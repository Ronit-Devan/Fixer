"""gpu-doctor-agent: always-on sampling spine for ET's live GPU agent (Product A)."""

from gpu_doctor_agent.buffer import RingBuffer
from gpu_doctor_agent.config import AgentConfig, ConfigError
from gpu_doctor_agent.detector import IdleDetector, IdleEvent, IdleState
from gpu_doctor_agent.sampler import (
    MockNvmlSampler,
    NvmlSampler,
    NvmlUnavailable,
    Sample,
    Sampler,
    get_sampler,
)
from gpu_doctor_agent.torch_source import (
    TorchHookEventSource,
    TorchUnavailable,
    convert_function_events,
    map_category,
)

__version__ = "0.1.0"
__all__ = [
    "AgentConfig",
    "ConfigError",
    "IdleDetector",
    "IdleEvent",
    "IdleState",
    "MockNvmlSampler",
    "NvmlSampler",
    "NvmlUnavailable",
    "RingBuffer",
    "Sample",
    "Sampler",
    "TorchHookEventSource",
    "TorchUnavailable",
    "convert_function_events",
    "get_sampler",
    "map_category",
]
