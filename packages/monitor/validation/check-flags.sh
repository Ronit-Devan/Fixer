#!/usr/bin/env bash
# Verify the tuned flags ET's remediation can emit actually exist in THIS
# llama-server build. llama.cpp renames speculative-decoding flags across
# versions (e.g. --draft vs --draft-max vs --spec-draft-n-max), so a tune built
# for the wrong spelling fails the restart. Run this BEFORE trusting auto-apply
# on any box (especially Zane's) — it's the single check that kills version skew.
#
# Usage:
#   ./check-flags.sh                       # uses 'llama-server' on PATH
#   SERVER=/path/to/llama-server ./check-flags.sh
set -euo pipefail

SERVER="${SERVER:-llama-server}"
if ! command -v "$SERVER" >/dev/null 2>&1 && [ ! -x "$SERVER" ]; then
  echo "ERROR: llama-server not found (set SERVER=/path/to/llama-server)" >&2
  exit 1
fi

HELP="$("$SERVER" --help 2>&1 || true)"
[ -n "$HELP" ] || { echo "ERROR: '$SERVER --help' produced no output" >&2; exit 1; }

have() { printf '%s' "$HELP" | grep -qE -- "$1"; }

# capability label | regex of acceptable spellings (any one present = supported)
ROWS=(
  "full GPU offload (-ngl)            |(^|[^a-z])(-ngl|--n-gpu-layers|--gpu-layers)([^a-z]|$)"
  "flash attention (--flash-attn)     |(-fa|--flash-attn)"
  "KV cache type K (--cache-type-k)   |(-ctk|--cache-type-k)"
  "KV cache type V (--cache-type-v)   |(-ctv|--cache-type-v)"
  "continuous batching                |--cont-batching|--no-cont-batching"
  "parallel slots (--parallel)        |(-np|--parallel)"
  "micro-batch (--ubatch-size)        |(-ub|--ubatch-size)"
  "mlock                              |--mlock"
  "draft model (-md/--model-draft)    |(-md|--model-draft|--spec-draft-model)"
  "draft GPU layers (-ngld) [REQUIRED for spec decode]|(-ngld|--n-gpu-layers-draft|--gpu-layers-draft)"
  "draft tokens/step (--draft)        |(--draft($|[^-])|--draft-max|--spec-draft-n-max|-nd|--draft-n)"
  "draft min                          |(--draft-min|--spec-draft-n-min)"
  "draft p-min/split                  |(--draft-p-min|--spec-draft-p-split|--draft-p-split)"
)

echo "==> llama-server flag compatibility ($SERVER)"
miss_core=0; miss_spec=0
for row in "${ROWS[@]}"; do
  label="${row%%|*}"; rx="${row#*|}"
  if have "$rx"; then
    printf "  [ OK ] %s\n" "$label"
  else
    printf "  [MISS] %s\n" "$label"
    case "$label" in
      *REQUIRED*|*"draft "*|*"draft model"*|*"draft tokens"*) miss_spec=$((miss_spec+1));;
      *ubatch*|*"draft min"*|*"p-min"*) : ;;  # optional / nice-to-have
      *) miss_core=$((miss_core+1));;
    esac
  fi
done

echo "------------------------------------------------------------"
if [ "$miss_core" -gt 0 ]; then
  echo "RESULT: $miss_core CORE flag(s) missing — do NOT auto-apply on this build."
  exit 1
fi
if [ "$miss_spec" -gt 0 ]; then
  echo "RESULT: core OK, but speculative-decoding flags differ on this build."
  echo "        The -ngl / flash-attn / KV-quant fixes are safe; set the draft flag"
  echo "        names that DO appear above via remediation knobs before enabling spec decode."
  exit 0
fi
echo "RESULT: all flags present — every ET tune renders correctly on this build."
