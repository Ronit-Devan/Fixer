#!/usr/bin/env bash
# Measure REAL decode tokens/sec from a running llama-server.
#
# This is the ground-truth measurement everything else compares against: it asks
# the server to generate N tokens and reads llama-server's own timing report
# (timings.predicted_per_second), median over a few runs to smooth noise.
#
# Usage:
#   ./bench-tps.sh                                  # localhost:8080, defaults
#   URL=http://host:8080 N=256 RUNS=3 ./bench-tps.sh
#   ./bench-tps.sh --url http://host:8080 --n 256 --runs 3
#
# Prints a single line:  TPS_DECODE=<median tok/s>   (also prompt tok/s)
set -euo pipefail

URL="${URL:-http://localhost:8080}"
N="${N:-256}"          # tokens to generate per run
RUNS="${RUNS:-3}"      # measured runs (a warmup run is always done first)
PROMPT="${PROMPT:-Write a detailed technical explanation of how GPUs execute matrix multiplication.}"

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --runs) RUNS="$2"; shift 2;;
    --prompt) PROMPT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

command -v curl >/dev/null || { echo "curl required" >&2; exit 1; }
PY="${PYTHON:-python3}"

# One request -> prints "<decode_tps> <prompt_tps>" parsed from llama-server timings.
_one() {
  local npred="$1"
  local body
  body=$(curl -fsS "$URL/completion" \
    -H 'Content-Type: application/json' \
    -d "$($PY - "$PROMPT" "$npred" <<'PYEOF'
import json,sys
print(json.dumps({"prompt": sys.argv[1], "n_predict": int(sys.argv[2]),
                  "cache_prompt": False, "temperature": 0}))
PYEOF
)") || return 1
  printf '%s' "$body" | $PY - <<'PYEOF'
import json,sys
try:
    d=json.load(sys.stdin); t=d.get("timings",{}) or {}
    dec=t.get("predicted_per_second"); pre=t.get("prompt_per_second")
    if dec is None:
        # Fallback: derive from counts/ms if per_second fields absent on this build.
        n=t.get("predicted_n"); ms=t.get("predicted_ms")
        dec=(n/(ms/1000.0)) if (n and ms) else 0.0
    print(f"{dec or 0:.2f} {pre or 0:.2f}")
except Exception as e:
    print("0 0")
PYEOF
}

echo "==> benchmarking $URL  (n_predict=$N, runs=$RUNS)"
echo "    warmup..."
_one 32 >/dev/null 2>&1 || { echo "ERROR: server at $URL did not respond to /completion" >&2; exit 1; }

decs=(); pres=()
for i in $(seq 1 "$RUNS"); do
  read -r d p < <(_one "$N")
  decs+=("$d"); pres+=("$p")
  printf "    run %d/%d:  decode %s tok/s   prompt %s tok/s\n" "$i" "$RUNS" "$d" "$p"
done

# Median (robust to one slow run).
median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
MED_DEC=$(median "${decs[@]}")
MED_PRE=$(median "${pres[@]}")

echo "------------------------------------------------------------"
echo "TPS_DECODE=$MED_DEC"
echo "TPS_PROMPT=$MED_PRE"
