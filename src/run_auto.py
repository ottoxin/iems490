from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from auto_search import (
    load_bbh_movie_recommendation, read_baseline_template, build_inference_prompt,
    render_meta_prompt, parse_candidates, dedup_candidates, post_chat_for_generation,
    format_examples_for_calibration, format_prior_block_from_history, eval_instruction_on_items,
    rank_scored_instructions, random_split, pick_exemplars, extract_letter, _detect_provider
)
from model_api import call_model  # solver

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="OPRO (paper-aligned) with progress output")
    ap.add_argument("--data", required=True, help="Path to BBH movie_recommendation.json")
    ap.add_argument("--outdir", default="outputs/auto", help="Output directory")
    ap.add_argument("--K", type=int, default=8, help="Candidate INSTRUCTIONS per optimization step (paper: 8)")
    ap.add_argument("--steps", type=int, default=1, choices=[1,2], help="Optimization steps")
    ap.add_argument("--val-size", type=int, default=50, help="Validation size (~20% for BBH)")
    ap.add_argument("--test-size", type=int, default=200, help="Test size")
    ap.add_argument("--exemplars", type=int, default=3, help="Exemplars per step for optimizer (paper: 3)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap.add_argument("--optimizer-temp", type=float, default=1.0, help="Optimizer LLM temperature (paper: 1.0)")
    ap.add_argument("--save", choices=["none","minimal","full"], default="none",
                    help="Artifacts to save: none | minimal | full")
    ap.add_argument("--meta-template", default="prompts/opro_generate.txt", help="Meta-prompt template path")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output (keeps progress bars)")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    # Detect provider to print a quick banner
    provider = _detect_provider()
    print(f"[setup] Optimizer provider: {provider['name']} | model={provider['model']}")
    print(f"[setup] Steps={args.steps} | K={args.K} | val={args.val_size} | test={args.test_size} | exemplars={args.exemplars} | seed={args.seed}")

    # Load and split
    start_total = time.time()
    items = load_bbh_movie_recommendation(args.data)
    val_items, test_items, split_info = random_split(items, args.val_size, args.test_size, args.seed)
    if not args.quiet:
        print(f"[data] Loaded {len(items)} items → val={len(val_items)} test={len(test_items)}")

    # Baseline (test) acc with no instruction
    baseline_t = read_baseline_template()
    base_ok = 0
    if not args.quiet:
        print("[baseline] Scoring baseline on TEST …")
    iterator = test_items
    if _HAS_TQDM:
        iterator = tqdm(test_items, desc="Baseline (test)", leave=False)
    for ex in iterator:
        p = baseline_t.replace("{{question}}", ex["prompt"])
        r = call_model(p)
        base_ok += int(extract_letter(r) == ex["gold"])
    base_acc_test = base_ok / len(test_items) if test_items else 0.0
    print(f"[baseline] Test accuracy = {base_acc_test:.3f}")

    # Exemplars (paper: 3)
    ex_items, ex_idx = pick_exemplars(val_items, args.exemplars, args.seed)
    calib_block = format_examples_for_calibration(ex_items)

    best_history: List[Tuple[str, float]] = []  # [(instruction, val_acc)]
    all_scored: List[Tuple[str, float, float]] = []  # [(instruction, acc, compliance)]
    best_overall = None  # (instruction, val_acc, val_comp)

    for step in range(1, args.steps + 1):
        print(f"\n[step {step}/{args.steps}] Building meta-prompt …")
        prior_block = format_prior_block_from_history(best_history, cap=20)
        meta_prompt = render_meta_prompt(args.meta_template, calib_block, prior_block, args.K)

        if args.save == "full":
            save_text(outdir / f"meta_prompt_step{step}.txt", meta_prompt)

        print(f"[step {step}] Generating {args.K} instruction candidates …")
        raw = post_chat_for_generation(meta_prompt, max_tokens=512, temperature=args.optimizer_temp)
        cands = parse_candidates(raw, args.K)
        before = len(cands)
        cands = dedup_candidates(cands, jacc_thresh=0.85)
        after = len(cands)
        print(f"[step {step}] Candidates received={before}, after de-dup={after}")

        if args.save == "full":
            (outdir / f"candidates_step{step}.json").write_text(json.dumps(cands, ensure_ascii=False, indent=2), encoding="utf-8")

        # Evaluate on validation
        print(f"[step {step}] Scoring candidates on VALIDATION ({len(val_items)} items) …")
        rows: List[Dict[str, Any]] = []
        scored_step: List[Tuple[str, float, float]] = []

        # Outer loop over instructions with a progress bar for instruction count
        instr_iter = enumerate(cands, 1)
        if _HAS_TQDM:
            instr_iter = tqdm(list(instr_iter), desc=f"Instructions (step {step})", leave=False)

        for idx, instr in instr_iter:
            label = f"instr {idx}/{len(cands)}"
            acc, comp, n, _preds = eval_instruction_on_items(instr, val_items, baseline_t, show_progress=True, progress_label=label)
            scored_step.append((instr, acc, comp))
            rows.append({"instruction": instr, "acc": acc, "compliance": comp, "n": n, "len": len(instr)})
            if not args.quiet:
                print(f"  - [{label}] acc={acc:.3f} comp={comp:.3f} | {instr[:80]}")

        # Save round scores
        if args.save in ("minimal","full"):
            import pandas as pd
            df = pd.DataFrame(rows).sort_values(by=["acc","compliance","len","instruction"], ascending=[False,False,True,True])
            outdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(outdir / f"val_scores_step{step}.csv", index=False)

        # Update histories
        all_scored.extend(scored_step)
        ranked_all = rank_scored_instructions([(i,a,c) for (i,a,c) in all_scored])
        best_overall = ranked_all[0]
        # Keep best-20 (instruction, val_acc) in history
        best_history = [(i, a) for (i, a, _c) in ranked_all[:20]]

        # Print a small leaderboard
        print(f"[step {step}] Top instructions this step:")
        for i, (ins, acc, comp) in enumerate(ranked_all[:5], 1):
            print(f"    {i:>2}. acc={acc:.3f} comp={comp:.3f} | {ins}")

        # Simple early stop: if no improvement vs previous step
        if step > 1:
            prev_best_acc = ranked_all[1][1] if len(ranked_all) > 1 else ranked_all[0][1]
            if best_overall[1] <= prev_best_acc:
                print(f"[step {step}] No improvement vs previous step; stopping early.")
                break

    # Final chosen instruction
    best_instruction, best_val_acc, best_val_comp = best_overall
    print("\n[final] Chosen instruction:")
    print("       " + best_instruction)

    # Evaluate on TEST
    print(f"[final] Scoring chosen instruction on TEST ({len(test_items)} items) …")
    test_acc, test_comp, _, test_preds = eval_instruction_on_items(best_instruction, test_items, baseline_t, show_progress=True, progress_label="TEST")

    # Save artifacts
    if args.save in ("minimal","full"):
        save_text(outdir / "best_instruction.txt", best_instruction)
        (outdir / "test_preds.json").write_text(json.dumps(test_preds, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = (
            f"OPRO (paper-aligned) for Movie Recommendation\n"
            f"Seed: {args.seed} | steps: {args.steps} | K: {args.K} | optimizer_temp: {args.optimizer_temp}\n"
            f"Splits: val={len(val_items)} test={len(test_items)} | exemplars per step: {args.exemplars}\n\n"
            f"Baseline (test) accuracy: {base_acc_test:.3f}\n"
            f"Best instruction (val) accuracy: {best_val_acc:.3f} (compliance {best_val_comp:.3f})\n"
            f"OPRO (test) accuracy: {test_acc:.3f} (compliance {test_comp:.3f})\n"
            f"Δ vs Baseline (test): {test_acc - base_acc_test:+.3f}\n\n"
            f"Best instruction:\n{best_instruction}\n"
        )
        save_text(outdir / "summary.txt", summary)

    # Console summary
    print("\n=== OPRO (paper-aligned) summary ===")
    print(f"Baseline (test) acc = {base_acc_test:.3f}")
    print(f"Best instruction (val) acc = {best_val_acc:.3f} | comp = {best_val_comp:.3f}")
    print(f"OPRO (test) acc = {test_acc:.3f} | comp = {test_comp:.3f}")
    print(f"Δ vs Baseline (test) = {test_acc - base_acc_test:+.3f}")
    print(f"Total wall time: {time.time() - start_total:.1f}s")

if __name__ == "__main__":
    main()