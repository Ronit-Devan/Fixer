# ET agent — Kubernetes deploy

Artifacts for running the ET sampling spine as a Kubernetes DaemonSet —
one pod per node, low overhead, no host perturbation.

The validation profile today is **Tier-1 sampling + mock attribution**.
It needs no GPU, no NVML on the host, and no privileged capabilities.
The point is to prove the rollout shape works end-to-end (DS topology,
resource ceilings, lifecycle, security context, signal handling) on
kind / CI before live attribution is wired in.

> Today this DaemonSet runs **Tier-1 + mock attribution**, NOT live
> attribution. The "On a real GPU cluster" section below lists every
> change required to flip it to live.

## Prerequisites

- Docker 23+ (BuildKit is on by default)
- [`kind`](https://kind.sigs.k8s.io/) and `kubectl`

## Build the image

From the repo root:

```bash
docker buildx build \
  --build-context engine=packages/engine \
  -t et-agent:dev \
  -f packages/agent/Dockerfile \
  packages/agent
```

`--build-context engine=...` is how the agent's `../engine` path
dependency gets pulled into a build context rooted at `packages/agent/`
(which is also where the `.dockerignore` lives). Resulting image is
~230-280 MB uncompressed — see the size breakdown comment at the top
of [packages/agent/Dockerfile](../Dockerfile).

## Load it into a kind cluster

```bash
# One-time: create a cluster.
kind create cluster --name et

# Push the locally-built image into the cluster's image store.
kind load docker-image et-agent:dev --name et
```

The manifest sets `imagePullPolicy: IfNotPresent` so kubelet uses the
loaded image instead of trying to pull from a registry that doesn't
have it.

## Deploy

```bash
kubectl apply -f packages/agent/deploy/daemonset.yaml
```

This creates the `et-system` namespace, a service account, and the
DaemonSet. On a 1-node kind cluster you'll see one agent pod.

## Verify it's running

```bash
kubectl -n et-system get ds et-agent
kubectl -n et-system get pods -l app.kubernetes.io/name=et-agent -o wide
```

Tail logs (the agent prints alerts as the mock `idle` scenario sustains
long enough to trip the detector):

```bash
kubectl -n et-system logs -l app.kubernetes.io/name=et-agent -f --tail=20
```

You should see:

- A startup banner with the resolved sampler config (interval,
  thresholds, sustain, recovery).
- An `ALERT` line each time the mock scenario sustains idle long enough
  for the detector to confirm — roughly every ~10 ticks at the default
  1.0s interval, modulated by the attribution verdict.

## Tear down

```bash
kubectl delete -f packages/agent/deploy/daemonset.yaml

# Optional: drop the kind cluster entirely.
kind delete cluster --name et
```

## What this manifest enforces today

- **Footprint contract.** requests `cpu=25m memory=64Mi`, limits
  `cpu=100m memory=128Mi`. This is the agent's overhead guarantee per
  node — any sustained breach is a regression to investigate, not a
  knob to raise.
- **Hardened pod spec.** Non-root (UID 10001), read-only root
  filesystem, `ALL` capabilities dropped, no privilege escalation,
  RuntimeDefault seccomp. /tmp and /home/agent are tiny pod-scoped
  emptyDirs.
- **Graceful shutdown.** `terminationGracePeriodSeconds: 30` so the
  agent's SIGTERM handler — which trips a shutdown flag the loop
  checks once per sample tick — has time to flush its summary line
  and exit cleanly.
- **No host perturbation.** No hostPID, no hostNetwork, no hostPath
  mounts, no privileged caps. The pod is effectively a regular
  workload that happens to be scheduled per-node.

## On a real GPU cluster

The kind profile above is deliberately inert on real GPUs. To run
against actual NVML, **every** item below has to be done — none is
"just flip a flag":

1. **Schedule onto GPU nodes.** Uncomment the `nodeSelector`
   (`nvidia.com/gpu.present: "true"`) and the matching `tolerations`
   block in `daemonset.yaml`. Assumes either the NVIDIA GPU Operator
   or Node Feature Discovery is labeling the nodes.
2. **Drop `--mock`, pick a real attribution source.** Override `args`
   in the container spec, e.g.:
   ```yaml
   args: ["run", "--no-mock", "--attribution-source", "none", "--interval", "1.0"]
   ```
   for plain Tier-1, or `--attribution-source file --trace-file ...`
   (with the trace mounted via ConfigMap/PVC) for replay. The
   eBPF/CUPTI live source from 3c plugs in here too.
3. **Mount NVML.** Add a hostPath for
   `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1` (or rely on the
   NVIDIA container runtime's automatic injection if the cluster uses
   it). `pynvml` opens the library lazily and raises
   `NvmlUnavailable` otherwise.
4. **Device access.** Either request a GPU via the NVIDIA device
   plugin (`resources.limits."nvidia.com/gpu": 1`) or — if just
   observing — hostPath-mount `/dev/nvidiactl` and `/dev/nvidia0`
   with a securityContext that permits device access.
5. **3c eBPF live tap.** When the host-side eBPF sampler lands the
   pod additionally needs `hostPID: true`, `capabilities.add:
   [BPF, PERFMON, SYS_RESOURCE]`, and a `/sys/kernel/debug` mount.
   That profile belongs in a separate, reviewed manifest, not as a
   delta on this one.
