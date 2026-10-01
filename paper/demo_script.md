# Demo script — results dashboard

For the Summer School of Astronomy & Astrophysics 2026 recorded explanation
video. This is the walkthrough for the **DEMO** cut in `presentation.pptx`
(slide 11, "Live demo: the dashboard") — record it separately and edit it
into the deck at that cut point, or screen-record it live during a single
take that pauses the slides. Target length: **4–5 minutes**, five stops,
in order: **Overview → Results → Objects → Anomalies → Live**.

Every number this script tells you to say **out loud as a fixed figure** is
also printed on the actual dashboard, sourced live from the same JSON files
`paper/numbers.tex` was generated from (`reports/gate_c/gate_c_report.json`,
`outputs/anomaly/anomaly_results.json`,
`outputs/anomaly/persisted_scorers_meta.json`). **Read the number that is
actually on your screen, not the one printed in this script** — they should
match (both come from the same on-disk artifact, untouched since the paper
was written), but the screen is the source of truth at recording time. If a
number on screen ever disagrees with this script, trust the screen and note
the discrepancy afterward rather than reading the wrong figure aloud.

**On the Live tab specifically: say plainly, on camera, that you are
replaying archived Tier B objects through the live pipeline, not receiving
real-time Fink alerts.** This was a deliberate choice for this recording
(reliability over live-network risk — see the session report), and the
dashboard's own live-Fink code path is verified working separately
(`GUIDE.md` §5, Day 13). Don't let a viewer assume otherwise; Step 5 below
gives you the exact line to say.

---

## Pre-demo checklist

Do this **before** you start recording, in order. Budget 5–10 minutes the
first time; under 2 minutes on a rehearsed machine.

1. **Docker infra.**
   ```bash
   make infra-up
   # or: docker compose -f infra/docker-compose.yml up -d
   ```
   Confirm all three services are healthy before moving on:
   ```bash
   docker compose -f infra/docker-compose.yml ps
   ```
   You want `sn-classifier-kafka`, `sn-classifier-postgres`, and
   `sn-classifier-redis` all showing `healthy` in the STATUS column. This
   can take 10–20 seconds after `up -d` returns — re-run `ps` if any service
   still shows `starting`.

2. **The API.**
   ```bash
   make api
   # or: .venv/bin/uvicorn src.serve.api:app --host 0.0.0.0 --port 8010
   ```
   Leave this running in its own terminal. Confirm it's up:
   ```bash
   curl http://localhost:8010/health
   ```
   Should return a JSON `{"status": "ok", ...}`-shaped response, not a
   connection error.

3. **The dashboard.**
   ```bash
   make frontend
   # or: npm --prefix frontend run dev
   ```
   Leave this running in its own terminal too — **two long-running
   processes, two terminals, plus Docker**. Open `http://localhost:3000/`
   in the browser window you'll be recording. If port 8010 is already in
   use on your machine, change both the API's `--port` and
   `frontend/.env.local`'s `NEXT_PUBLIC_API_BASE_URL` together, or nothing
   on the frontend will load.

4. **Seed it with data (replay mode — see the note above on why).** In a
   third terminal:
   ```bash
   python -m src.serve.inference_worker --idle-stop-seconds 30 &
   python -m src.serve.replay --split test
   ```
   (`make worker` / `make replay` also work, but omit the idle-stop flag —
   the worker will then run indefinitely rather than exiting cleanly once
   the replay finishes, which is harmless but means you'll `Ctrl+C` it
   yourself once Step 5 is done recording.) Within a few seconds you should
   see the Overview page's throughput sparkline start climbing. This
   replays the full Tier B test split (1,994 objects, 49,587 alerts) — more
   than you need for the recording, but it's what reproduces the paper's
   own end-to-end validation number (§ below), so let it run in the
   background through the whole demo rather than stopping it early.

5. **Confirm object detail will actually load.** Before recording, open any
   object's card once (Objects tab, click any card) and check that its
   light curve renders rather than showing "No live rolling-state history."
   If you see that message, Redis's rolling state isn't populated yet —
   re-run step 4's two commands and wait ~30 seconds before trying again.

6. **Close everything unrelated** in the browser window/tab you'll record
   — other tabs, notification banners, your OS's dark-mode toggle if you
   want to match the deck's light theme. Resize the browser window now,
   not mid-recording.

**If any of steps 1–3 fails outright** (Docker won't start, a port is
occupied, `npm run dev` errors), don't troubleshoot on camera — cut to
slide 11 in the deck, narrate over it using the "what to say" text below for
each of the five stops, and come back to the live dashboard afterward for
your own use. That's what the fallback column is for.

---

## Step 1 — Overview tab (`/`)

**~45 seconds**

| | |
|---|---|
| **Click** | Land on `http://localhost:3000/` (the Overview tab is the default landing page — no click needed if you're already there). |
| **Say** | "Here's the dashboard's front page. These five numbers are read live from the same report files behind the paper. Tier A macro-F1 is our simulated-data ceiling — [read the value on screen]. Tier B is the deployed, fine-tuned model on real ZTF data — [read the value]. The sim-to-real gap between them is the drop we talked about a moment ago. This ROC-AUC is the anomaly branch's three-component score on real, confirmed non-supernova objects. And this last card is genuinely live — serving latency, computed from whatever's been scored in the last few minutes, not a number we pre-recorded." Then: "Below that are our four validation gates — three passed, and Gate D is marked deferred, which we'll come back to." |
| **Timing** | ~45s. Don't linger — this tab is a dashboard cover page, not where the depth is. |
| **If it doesn't load** | Cut to slide 11 in the deck (the demo handover slide) and say the same lines, substituting the fixed values from `paper/numbers.tex`: Tier A macro-F1 **0.7169**, Tier B macro-F1 **0.6716**, sim-to-real gap **−0.0453**, anomaly ROC-AUC (Tier B, 3-component) **0.7616**. Say explicitly that you're showing the report figures because the live dashboard didn't come up for this recording. |

---

## Step 2 — Results tab (`/results`)

**~90 seconds — the most important stop before Objects.**

| | |
|---|---|
| **Click** | Sidebar → **Results**. |
| **Say (phase curve, ~35s)** | "This is macro-F1 as a function of how many days of the light curve the model has seen — our primary reported metric throughout this project, not a single end-of-curve number. Both curves rise monotonically as more photometry arrives — that shape is itself a sanity check: a model that got *worse* with more data would indicate a bug, not a subtlety. Tier B has no point before day zero because real BTSbot alerts carry no pre-detection forced photometry at all — we don't plot a point we don't have." |
| **Click** | Scroll to the confusion matrix card. Click the **Tier B (deployed, real ZTF)** tab if it isn't already selected. |
| **Say (confusion matrix, ~40s)** | "Now watch the diagonal." **Drag the cutoff slider from the leftmost position to `full`, slowly, narrating as you go:** "At the earliest cutoff, the model has almost nothing to work with and collapses onto predicting the majority class for nearly everything — you can see the off-diagonal is dominant. As I drag this forward, accumulating more of each object's photometry, watch the diagonal strengthen." Let it settle on `full`. "This is a better explanation of what 'the model works' actually means than any single static figure — it's the model's confidence sharpening in real time as the same kind of evidence a live system would accumulate arrives." |
| **Timing** | ~90s total (35s phase curve + ~10s transition + 40s confusion matrix drag + ~5s close). If you're short on time, the confusion-matrix slider drag is the one moment in the whole demo not to cut. |
| **If it doesn't load** | Fall back to `paper/figures/fig4_phase_curve.pdf` and `fig5_confusion_grid.pdf` (both already in the deck's source figures, presentation-resolution PNGs in the session scratch dir) — describe the slider verbally instead of showing it: "the confusion grid figure shows this same progression at fixed cutoffs — 0, 5, 15, 30 days, and full." |

---

## Step 3 — Objects tab (`/objects`) — the centrepiece

**~110 seconds**

| | |
|---|---|
| **Click** | Sidebar → **Objects**. |
| **Click** | Set the **Label** filter to **Correct**. Optionally also set **True class** to `Ia` (or any class) to land on a clean example faster. Click any card. |
| **Say (correct example, ~50s)** | "This is the centrepiece of the dashboard — every object we've scored, with its full detail available. I've filtered to a correct classification." [Click a card.] "Here's the light curve exactly as the model saw it, colored by wavelength. Here are the real science, template, and difference cutout images — not simulated, these are the actual postage stamps from the alert. On the right, the model's own class-probability bars, and below that, the anomaly-score breakdown for this same object, by component." Point out the true/predicted class badges and the checkmark in the header. |
| **Click** | Close the detail view. Change **Label** to **Incorrect**, and if you can, **True class** to `Ib/c` (the project's own recorded dominant confusion is true-Ib/c-predicted-as-Ia — `paper/sections/results.tex`, Figure 6 caption). Click the most confident wrong call you can find. |
| **Say (incorrect example, ~50s)** | "Now the honest part. This object is truly Ib/c, spectroscopically, but the model called it something else with real confidence — you can see that in the probability bars. Look at the light curve: it's sparse, unevenly sampled compared to the last one. Ib/c is both the rarest class in the real sky and the class our own real test split supports the least — this is exactly the kind of case that number is summarizing. We're not hiding this; it's the expected shape of the sim-to-real gap we just showed you on the Results tab." |
| **Timing** | ~110s (50 + 10 transition + 50). This is worth going slightly over budget elsewhere to protect. |
| **If it doesn't load** | Fall back to `paper/figures/fig6_prediction_panels.pdf` (four real Tier B objects already laid out exactly this way: light curve, cutout triplet, probability bars — including the misclassified Ib/c-as-Ia panel described in the same terms as above). Narrate over it with the same two paragraphs. |

---

## Step 4 — Anomalies tab (`/anomalies`)

**~60 seconds**

| | |
|---|---|
| **Click** | Sidebar → **Anomalies**. |
| **Say (~40s)** | "Three components feed the anomaly score. Reconstruction error asks whether a causal decoder can predict this object's next flux point from its own recent history — the component that actually held up best on real data. Mahalanobis distance and an isolation forest both measure distance in the classifier's own learned space — strong on simulated data, but they don't transfer well to real ZTF, because that geometry was learned entirely from simulation. There was a fourth candidate, classifier confidence — we tried it and dropped it, because it does the opposite of what you'd want: it gets *more* confident, not less, on objects unlike anything it was trained on." Point at the Gate D badge. "And this is Gate D, marked deferred — every object in this evaluation set turned out to already be a confirmed real astrophysical object, so there was never a bogus or artifact candidate in the pool for the ranker to have found. We're stating that directly rather than calling it a pass." |
| **Click** | Scroll to the top-ranked objects table; click the top row. |
| **Say (~15s)** | "This is the actual top-ranked object right now, with its full detail available exactly like the Objects tab." |
| **Timing** | ~60s. |
| **If it doesn't load** | Fall back to `paper/figures/fig7_anomaly_slope.pdf` and describe the four components verbally from the table above, then read the Gate D deferral line from `RESULTS.md`'s "What the gates showed" section. |

---

## Step 5 — Live tab (`/live`)

**~45 seconds — say the disclosure line on camera.**

| | |
|---|---|
| **Click** | Sidebar → **Live monitor**. |
| **Say (disclosure — say this verbatim, early in this segment)** | "One honest note before we look at this: what's streaming through here right now is **replayed data from our archived Tier B test set**, not a live connection to Fink's real alert broker. We verified live Fink ingestion separately during development — it connects and streams real ZTF alerts within seconds — but for this recording we're using the same replay pipeline you just saw feed the rest of the dashboard, because it's deterministic and doesn't depend on network conditions at the moment we're recording. The consumer code on the other side of this — Kafka to Redis to the model to Postgres — is the exact same code either way; only the producer changes." |
| **Say (the rest, ~20s)** | "You can see alerts arriving in the bar chart, and the table below fills in as each one is scored — object, predicted and true class, confidence, anomaly score, and end-to-end latency, all real numbers from this actual run." Click one row to show the detail view opens the same way as the other tabs. |
| **Timing** | ~45s. |
| **If it doesn't load** | Cut to slide 11 (or a static screenshot if you captured one during rehearsal) and say: "This is what the live monitor looks like once alerts are streaming — again, this particular capture is replayed Tier B data, not a live broker connection." Do **not** imply the fallback image is live if it visibly isn't (e.g. a static screenshot with a stopped timestamp) — say so. |

---

## After the demo

Cut back to the deck at slide 12 ("What the sim-to-real gap means"). Stop
the replay/worker processes (`Ctrl+C` both, or let the worker's
`--idle-stop-seconds 30` exit on its own once replay finishes) before you
stop recording, so the terminal doesn't sit there mid-shot if it's in frame.

## Total demo timing

45s + 90s + 110s + 60s + 45s ≈ **350s (~5:50)**, above the 4–5 minute
target once every "if it loads cleanly" segment runs at its stated length.
To pull it back under 5 minutes: trim the Overview stop to ~30s (drop the
Gate-by-gate detail, just say "three passed, one deferred"), and hold the
Anomalies stop to ~45s by skipping the top-row click-through. The Results
slider drag and the Objects correct/incorrect pair are the two segments
this script recommends protecting from cuts, since they're the ones the
brief itself calls out as the strongest evidence ("spend time here").
