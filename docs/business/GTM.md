# ET go-to-market: getting clients (and why cold email isn't working)

Written for the founders. Blunt, concrete, prioritized. The thesis: stop
selling, start distributing. ET's natural motion is **product-led**: a free
local tool that gives instant, screenshot-able value, not outbound cold email.

---

## 1. Why the cold emails are getting silence

Cold email fails for *developer infrastructure* almost every time, for
specific reasons:

- **Wrong buyer behavior.** Engineers don't buy infra tools from a cold inbox.
  They discover tools (HN, Reddit, GitHub, X, a teammate), *try them in 5
  minutes*, and adopt if they work. The purchase comes after adoption, not
  before.
- **No proof, no trust.** You're a two-person team with one logo. A cold email
  asking for a meeting is all ask, no give. There's nothing to react to.
- **No specificity.** "We detect GPU idleness" is a category, not a pain. The
  reader can't tell if it applies to them without doing work.
- **The medium fights you.** A meeting request is high-friction. A link to a
  tool that tells them "your GPU was idle 68% of last week = ~$900/mo" is
  low-friction and self-proving.

Cold email *can* work later as a complement, but only once you have proof
(a case study, a number, a free tool people already use). Leading with it now
is pushing on a string.

---

## 2. Who actually buys this (ICP)

Not the frontier labs; they have in-house observability. Your buyer is the
team that **owns or reserves GPUs for inference and feels the cost**:

**Tier-1 (start here; exactly your current client's shape):**
- Startups self-hosting LLM inference on **their own / reserved GPUs**
  (`llama.cpp`, Ollama, vLLM, TGI); on-prem boxes, a workstation card, or a
  handful of cloud GPUs they pay for 24/7.
- They're GPU-rich but ops-poor: no Grafana/DCGM stack, no one watching
  utilization. Idle GPU = burning money they can see on the invoice.

**Tier-2 (expand later):**
- Neoclouds / GPU-rental providers who want to *prove* utilization to their own
  customers (white-label the dashboard).
- AI agencies / consultancies running inference for multiple clients.
- Anyone with a sticker-shock GPU bill and no utilization data.

**Disqualifiers:** teams fully on serverless inference APIs (no GPU to watch),
or already running mature DCGM + Grafana fleets (you'd be a feature, not a
product; for now).

---

## 3. The motion that fits: product-led, free tool first

The `packages/monitor` web app **is** your top-of-funnel. It is:
`pip install` → one command → opens a browser → within 60s tells them how much
GPU they're wasting in dollars. That is a *self-proving* artifact. Lead with it
everywhere.

**Funnel:**
1. **Discover**; they see it on HN / r/LocalLLaMA / X / GitHub.
2. **Try**; `./run.sh --gpu-price 0.50`, instant value, no signup.
3. **Share**; the "$X wasted/mo" readout is screenshot-bait; that's your viral loop.
4. **Expand**; the free tool is single-box, point-in-time. Paid is the layer
   they need once they care: multi-node fleet view, historical trends,
   alerting (Slack/PagerDuty on sustained idle or KV pressure), and
   recommendations they can act on. That's the wedge from free → paid.

**Where to distribute (ranked):**
1. **r/LocalLLaMA**; your exact users, `llama.cpp` natives. A genuine
   "I built a tool that shows you what your inference GPU is actually doing"
   post with a GIF of the dashboard cycling verdicts. Be a participant, not an ad.
2. **Show HN**; "Show HN: ET; see why your llama.cpp GPU is idle (local web app)".
   Ship with the demo GIF and the open-source repo. Respond to every comment.
3. **X / Twitter ML-infra**; short clip of the dashboard + the wasted-$ number.
   Tag/reply into `llama.cpp`, Ollama, local-LLM threads.
4. **GitHub**; the repo itself is distribution. Clean README, GIF at top, one
   command. Get it llama.cpp-adjacent (a "tools" list PR, awesome-llama lists).
5. **Discords**; llama.cpp, Ollama, LocalLLaMA, GPU/homelab servers.
6. **dev.to / a short blog**; "We measured 30 self-hosted inference GPUs.
   Median utilization was X%." Data posts travel.

---

## 4. Turn your one client into the asset that unlocks everything

One paying client is not "small"; it's your **proof engine**. Extract:

- **A number.** Run the monitor on their Blackwell box for a week. Get the
  headline: "ET found GPU idle N% of the time ≈ $X/month on a single card."
- **A quote.** One sentence from them on what it surfaced that they couldn't see.
- **A mini case study** (one page): before (no visibility) → after (number +
  what they changed). This single page does more than 500 cold emails.

Even though they're "a bit off your scope," they are *on-scope for the new
inference product*; that's the point of building the `llama.cpp` monitor. Make
them successful, then let their result sell for you.

---

## 5. The investor narrative

Reframe from "we have a tool" to a market wedge:

> Self-hosted LLM inference is exploding, and almost none of those GPUs are
> instrumented for utilization. Teams are paying for cards that sit idle or
> run un-tuned. ET is the utilization-and-cost layer for self-hosted
> inference. We land with a free local monitor (instant "here's your wasted
> $"), and expand to fleet observability + tuning. We have a paying design
> partner and N free installs growing W%/week.

What investors want to see next, in order:
1. **Pull**, not push: installs/week, week-over-week retention, organic shares.
2. The **design-partner number** (the case study above).
3. A believable **free → paid** path (fleet, alerting, history).

Your current chicken-and-egg ("investors want product, product needs clients")
breaks the moment you have a free tool people install on their own. Installs are
the proof that de-risks the round; and they don't require permission from a
cold-email gatekeeper.

---

## 6. Next 30 / 60 / 90 days

**Days 0-30: proof + launch-ready**
- Run the monitor on the client's box; capture the wasted-$ number + quote.
- Record a 20-second GIF of the dashboard cycling verdicts (`et-monitor --demo`).
- Polish the repo README (GIF on top, one-command quickstart). ✅ scaffolded.

**Days 30-60: distribution**
- Show HN + r/LocalLLaMA + X launch, same week, with the GIF and the case study.
- Be present in every thread/comment for 48h. Ship fixes live.
- Add Slack/webhook alerting (the first obvious paid-adjacent feature).

**Days 60-90: convert pull into pipeline**
- Instrument installs (opt-in, privacy-respecting) so you can quote growth.
- Reach out to the *engaged* free users (not cold; warm, they already use it)
  about a fleet/team version.
- Take the installs + design-partner number to investors.

---

## 7. Pitfalls to avoid

- Don't gate the free tool behind a signup. Friction kills the loop.
- Don't over-broaden the pitch ("observability platform"). Win one wedge:
  *"see and cut wasted inference-GPU spend."*
- Don't bury the money. The wasted-$ number is the whole hook; keep it the
  first thing the dashboard shows.
- Don't ship telemetry without consent; this audience will roast you for it.
  Opt-in only, and say so loudly.
