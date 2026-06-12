# Calibration lessons from real PyTorch Profiler traces

## What we learned generating real fixtures

### Lesson 1: DataLoader is a wrapper that contains everything else

PyTorch's DataLoader code path is on the CPU thread during every batch
fetch, even when the actual bottleneck is the cuMemcpy inside the
DataLoader's `__iter__`. Naive pattern-matching on "DataLoader" in event
names will incorrectly attribute Memcpy-bound, NCCL-bound, and IO-bound
traces to DATALOADER_BOUND.

**Fix:** specific causes beat generic causes. Memcpy and NCCL get a 30%
threshold of idle-window share; if either crosses it, they win over
DataLoader. DataLoader is the fallback when nothing more specific fires.

**Research connection:** this matches the MinatoLoader paper's finding
(arXiv 2509.10712) that 76% of dataloader-attributed idleness on its
benchmark was actually head-of-line blocking on individual slow samples,
not the dataloader infrastructure itself.

### Lesson 2: gpu_memcpy events are GPU-busy time, not GPU-idle time

PyTorch Profiler classifies `gpu_memcpy` events as GPU activity. Our
merged-intervals math counts them as busy, so they never appear in
idle windows. The overlap-with-idle approach was structurally blind
to Memcpy bottlenecks.

**Fix:** a direct ratio rule. If `gpu_memcpy_time / (gpu_kernel_time +
gpu_memcpy_time) >= 50%`, we declare PCIE_BOUND regardless of what
the idle-window analysis says.

### Lesson 3: Production training sits at 70-85% util, not 95%+

The original 85% HEALTHY threshold was set assuming an idealized
training loop. Real well-configured runs sit at 70-85% with normal
dataloader activity between batches. Treating that as broken is the
fastest way to lose customer trust.

**Fix:** two-tier HEALTHY. ≥85% util is unambiguous healthy. ≥70% util
with no dominant suspect (no bottleneck > 50% of idle) is also healthy
at lower confidence.

## What this means for the product

Diagnostic accuracy on real traces is the entire wedge. Every false
positive we ship is a customer who uninstalls. Future detectors must
be validated on real fixtures before merging, not just synthetic ones.

A trace fixture per verdict is now part of the test suite. Adding a
new detector requires adding a new fixture that triggers it correctly.