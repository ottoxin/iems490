# IEMS Assignment 1: Movie Recommendation (BBH) — Baselines, CoT, and OPRO

This repo evaluates a multiple-choice **movie recommendation** task from BIG-Bench Hard (BBH).
Given movies a user liked and four options (A–E), the model must output **only a single letter** like `(C)`.

It includes:

* **Baseline** (zero-shot)
* **CoT (silent)** — think internally, still output `(A–E)`
* **OPRO (paper-aligned)** — automated prompt engineering (“Optimization by PROmpting”), which searches short **instructions** to append to the baseline prompt and improves accuracy on a held-out test split.

---

## Repo layout

```
movie-rec-bbh/
├─ README.md
├─ requirements.txt
│
├─ data/
│  └─ movie_recommendation.json      
│
├─ prompts/
│  ├─ baseline_movie_rec.txt         # Baseline prompt (must output only (A–E))
│  ├─ cot_silent_movie_rec.txt       # CoT prompt
│  └─ opro_generate.txt              # OPRO optimizer meta-prompt
│
└─ src/
   ├─ __init__.py
   ├─ model_api.py                   # OpenAI-compatible client; Qwen (DashScope) preferred
   ├─ load_bbh_movie_rec.py          # Robust loader; title→letter mapping
   ├─ run_baseline.py                # Baseline runner
   ├─ run_manual.py                  # CoT runner
   ├─ auto_search.py                 # OPRO utilities
   └─ run_auto.py                    # OPRO driver with progress output
```

---

## Requirements

```
python 3.11+
requests==2.32.3
tqdm==4.66.5
pandas==2.2.3
```

Install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Model & environment

This repo calls an **OpenAI-compatible** `/chat/completions` API.

Preferred: **Qwen-Flash** via DashScope (Alibaba Model Studio)

```bash
export DASHSCOPE_API_KEY=sk-...        # required
export DASHSCOPE_MODEL=qwen-flash      # default is qwen-flash
# optional region/base:
# export DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Fallbacks (optional):

* Kimi (Moonshot): `MOONSHOT_API_KEY`, `MOONSHOT_MODEL`
* OpenAI-compatible: `OPENAI_API_KEY`, `OPENAI_MODEL`

If no key is set, the code **raises** with a clear error (no dummy predictions).

---

## Data

Place **`data/movie_recommendation.json`** in the repo (committed).
The loader handles the BBH file format and maps any title-style gold answers (e.g., `Monsters, Inc`) to the correct option letter.

> **Note:** The full dataset contains **250** items.

---

## Usage

### 1) Baseline (from a small sample test)

```bash
python src/run_baseline.py data/movie_recommendation.json outputs/baseline_2.json prompts/baseline_movie_rec.txt 2
```

**Example output:**

```
Baseline: 100%|██████████████████████████████████████████████████████| 10/10 [00:09<00:00, 1.02it/s]
Baseline accuracy: 0.600  (N=10)
Wrote: outputs/baseline_10_qwen.json
```

---

### 2) CoT (fro a small sample test)

```bash
python src/run_manual.py data/movie_recommendation.json outputs/manual_cot_2.json prompts/cot_silent_movie_rec.txt 2
```

**Example output:**

```
CoT (silent): 100%|██████████████████████████████████████████████████| 10/10 [00:09<00:00, 1.06it/s]
CoT accuracy: 0.700  (N=10)
Wrote: outputs/manual_cot_10.json
```

---

### 3) OPRO 

OPRO searches short **instructions** (3–5 rules) to append to the baseline prompt.

**Paper guidance:** use ≈**20%** of BBH for tuning, **80%** for testing.
For a 250-item set:

* **Validation** (tuning): 50 items
* **Test** (final): 200 items
* **Exemplars per step** (shown to the optimizer): 3
* **Candidates per step (K)**: 8
* **Steps**: 1 iteration

Run:

```bash
python src/run_auto.py \
  --data data/movie_recommendation.json \
  --outdir outputs/auto \
  --K 8 --steps 1 \
  --val-size 50 --test-size 200 --exemplars 3 \
  --seed 42 \
  --optimizer-temp 1.0 \
  --save minimal
```

What you’ll see:

* Progress for baseline scoring, candidate generation, and validation scoring
* A small leaderboard of top instructions per step
* Final summary with:

  * **Baseline (test) accuracy**
  * **Best instruction (val) accuracy**
  * **OPRO (test) accuracy**
  * **Δ vs Baseline (test)**
  * The **best instruction** (single line)

If `--save minimal`, files written:

* `outputs/auto/best_instruction.txt`
* `outputs/auto/summary.txt`
* `outputs/auto/test_preds.json`

(Use `--save full` to also dump meta-prompts, candidates, and validation score CSVs; `--save none` prints only.)

**Example output (OPRO smoke test with tiny split; no files saved):**

```bash
python src/run_auto.py \
  --data data/movie_recommendation.json \
  --outdir outputs/auto_debug \
  --K 4 --steps 1 \
  --val-size 20 --test-size 20 --exemplars 3 \
  --seed 123 \
  --save none
```

```
=== OPRO summary ===
Baseline (test) acc = 0.600
Best rubric (val) acc = 0.650 | comp = 1.000
OPRO (test) acc = 0.650 | comp = 1.000
Δ vs Baseline (test) = +0.050

Best rubric:
Favor strongest genre alignment; then closest time period; then comparable mood; prefer widely recognized titles
```

---

## Reproducibility

* Fixed random seed for splits and exemplar selection (`--seed`).
* Paper-aligned defaults (`K=8`, `steps=1`, `exemplars=3`, optimizer temp=1.0).
* Optional artifacts: meta-prompts, candidate lists, validation score tables.
* Progress printing shows step sizes, provider/model, and compact per-instruction stats.

---

Absolutely—here’s a **ready-to-paste “Docker usage” section** for your README, plus where I’d place it.

## Where to put it

Add this **right after your “Usage” section** and before “Reproducibility/Troubleshooting.” That way readers first see local runs, then the containerized alternative.

---

## Docker

The provided `Dockerfile` builds a slim image with all deps and your project files. You’ll pass API keys at runtime and mount `outputs/` so results persist on your host.

> **Do not** hardcode real API keys in README or scripts. Use environment variables.

### Optional: clone fresh

```bash
git clone https://github.com/ottoxin/iems490.git
cd iems490
```

### Build image

```bash
docker build -t iems490 .
```

### Baseline (small sample test)

Mount `outputs/` so files land on your machine; pass your DashScope key as an env var.

```bash
docker run --rm -it \
  -e DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
  -e DASHSCOPE_MODEL="qwen-flash" \
  -v "$PWD/outputs:/app/outputs" \
  iems490 bash -lc \
  "python src/run_baseline.py data/movie_recommendation.json outputs/baseline_2.json prompts/baseline_movie_rec.txt 2"
```

### CoT (Chain of Thought on small sample)

```bash
docker run --rm -it \
  -e DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
  -e DASHSCOPE_MODEL="qwen-flash" \
  -v "$PWD/outputs:/app/outputs" \
  iems490 bash -lc \
  "python src/run_manual.py data/movie_recommendation.json \
     outputs/manual_cot_2.json \
     prompts/cot_silent_movie_rec.txt \
     2"
```

### OPRO (small sample test)

Quick end-to-end check with a very small split; prints results and writes minimal artifacts.

```bash
docker run --rm -it \
  -e DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
  -e DASHSCOPE_MODEL="qwen-flash" \
  -v "$PWD/outputs:/app/outputs" \
  iems490 bash -lc \
  "python src/run_auto.py --data data/movie_recommendation.json --outdir outputs/auto \
   --K 8 --steps 1 --val-size 5 --test-size 5 --exemplars 3 --seed 42 \
   --optimizer-temp 1.0 --save minimal"
```

### OPRO

Uses ~20% validation, 80% test (for 250 items: 50/200), `K=8`, one step—matching the OPRO paper defaults.

```bash
docker run --rm -it \
  -e DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" \
  -e DASHSCOPE_MODEL="qwen-flash" \
  -v "$PWD/outputs:/app/outputs" \
  iems490 bash -lc \
  "python src/run_auto.py --data data/movie_recommendation.json --outdir outputs/auto \
   --K 8 --steps 1 --val-size 50 --test-size 200 --exemplars 3 --seed 42 \
   --optimizer-temp 1.0 --save minimal"
```

**Notes**

* The `Dockerfile` copies `data/movie_recommendation.json` into the image; if you update the dataset, rebuild the image or bind-mount `data/` as well.

---

## Results

Here’s the cleaned table:

| Method                         | Split / N         | OPRO config (K / Steps / Exemplars) |           Val Acc |  Test Acc | Baseline Acc (same split) | Δ vs Baseline |
| ------------------------------ | ----------------- | ----------------------------------- | ----------------: | --------: | ------------------------: | ------------: |
| **Baseline**                   | 250 items         | —                                   |                 — | **0.548** |                         — |             — |
| **CoT**                        | 250 items         | —                                   |                 — | **0.584** |                     0.548 |    **+0.036** |
| **OPRO** (paper-aligned small) | val=50 / test=200  | K=8 / Steps=1 / Ex=3                | 0.660 (comp 1.00) | **0.660** |                     0.530 |    **+0.130** |
| **OPRO** (larger test)         | val≈50 / test=200 | K=8 / Steps=2 / Ex=3                     | 0.700 (comp 1.00) | **0.710** |                     0.535 |    **+0.175** |

---

## Troubleshooting

* **`RuntimeError: No API key configured`**
  Export `DASHSCOPE_API_KEY` (or another supported key).
* **Module import errors**
  Run commands from the **repo root** (the folder containing `src/`, `prompts/`, `data/`).
* **Model outputs more than `(A–E)`**
  Ensure you’re using the provided prompts; OPRO also tracks **compliance** and down-ranks bad instructions.
* **Data not found / format error**
  Confirm `data/movie_recommendation.json` exists and is valid JSON. The loader prints a clear error with a small snippet if the format is off.
---

