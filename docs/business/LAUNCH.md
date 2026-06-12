# Launch posts: ET Inference Monitor

Copy-paste drafts for the product-led launch. Lead with the demo GIF
(`et-monitor-demo.gif`) everywhere. Post the Show HN and the Reddit post the
same day; reply to every comment for the first 48 hours.

> Before launching: push the repo public, put the GIF at the top of the README,
> and make sure `./run.sh --demo` works from a clean clone. The whole pitch is
> "try it in 60 seconds"; the first-run experience IS the marketing.

---

## Show HN

**Title:**
`Show HN: ET, see why your llama.cpp GPU is idle (local web app)`

**Body:**

> I run llama.cpp on my own GPU and could never answer a basic question: when the
> card *isn't* busy, why? Is it idle because there's no traffic, or memory-bandwidth
> bound on decode, or is the KV cache full and requests are queueing? `nvidia-smi`
> shows a utilization number but not the *reason*.
>
> ET is a small local web app that answers that. You run one command on the box
> serving the model; it samples the GPU (NVML / nvidia-smi) and scrapes
> llama-server's `/metrics`, then tells you the root cause in plain language -
> idle / decode-bandwidth-bound / memory-headroom / KV-cache-pressure /
> throttling / healthy; plus an estimate of the dollars you're burning on idle.
>
> It's a single `pip install` + one command, opens in your browser, runs fully
> locally (nothing leaves the machine), no account. There's a `--demo` mode that
> works with no GPU if you just want to see it.
>
> Stack: Python (FastAPI) + a dependency-free vanilla-JS dashboard, ~no build
> step. GPU backend falls back NVML → nvidia-smi → mock so it runs anywhere.
> Optional Slack/webhook alerts when the GPU sits idle or throttles.
>
> Repo: <link> · 30-sec demo: <gif/video link>
>
> Would love feedback on the detection heuristics; especially from people
> running multi-slot or batched llama.cpp setups.

**First comment (post it yourself):** the verdict list + the one-paragraph
explanation of how idle attribution works (NVML idle windows × llama metrics).

---

## r/LocalLLaMA

**Title:**
`I built a free local tool that tells you WHY your inference GPU is idle (not just that it is)`

**Body:**

> Like a lot of you I self-host with llama.cpp, and I kept wanting to know what my
> GPU was actually doing; `nvidia-smi` gives a % but not the *why*.
>
> So I made **ET**: a local web-app monitor for a llama.cpp box. It watches the
> GPU + `llama-server` and gives you a plain-English verdict:
>
> - 🟦 **Idle: no requests** (and roughly what that idle is costing you)
> - 🟦 **Decode bandwidth-bound**: low-concurrency generation, batch for more throughput
> - 🟦 **Memory headroom**: VRAM free, you could run a bigger model / more context / more slots
> - ⚠️ **KV cache pressure**: cache full / requests deferred
> - 🚨 **Throttling**: clocks dragged down under load
> - ✅ **Healthy**
>
> One command, opens in the browser, 100% local, no signup. Has a `--demo` mode
> so you can see it without pointing it at anything. Optional Slack alerts.
>
> [GIF]
>
> Repo + setup: <link>
>
> Genuinely want feedback: what other inference bottlenecks should it detect?
> Anyone running batched / multi-slot; does the KV-pressure call match what you
> see?

*(Read the subreddit's self-promo rules first; lead as a builder sharing a tool,
not an ad. Engage in comments; that's what gets it upvoted.)*

---

## X / Twitter (thread)

1/ Your inference GPU is probably idle more than you think; and you can't see
why. I built a tiny local tool that tells you. 🧵 [GIF]

2/ ET watches your llama.cpp box (GPU + llama-server metrics) and gives a
plain-English verdict: idle / decode-bound / memory-headroom / KV-pressure /
throttling / healthy. Plus the $ you're burning on idle.

3/ One command, runs locally in your browser, no account. `--demo` mode works
with no GPU. Slack alerts optional. Open source: <link>

4/ Built it because `nvidia-smi` tells you the GPU is 12% busy but not *why*.
The "why" is where the money is. Feedback welcome; esp. from batched/multi-slot
setups.

---

## After launch; capture the proof

- Watch which commenters are *running it* (they'll mention their setup). Those
  are warm leads; DM them about a fleet/team version, not a cold pitch.
- Screenshot any "oh wow it found X% idle = $Y" reactions. That's social proof
  for the investor deck.
- Add a (consent-based, opt-in) install counter so you can quote installs/week.
