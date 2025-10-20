import re, json, sys
from pathlib import Path
from tqdm import tqdm
from load_bbh_movie_rec import load_bbh_movie_recommendation
from model_api import call_model

LETTER = re.compile(r"\(([A-E])\)")

def extract_letter(text: str) -> str:
    m = LETTER.search(text.upper())
    return m.group(1) if m else ""

def main(data_path: str, out_path: str, prompt_path: str = "prompts/cot_silent_movie_rec.txt",
         limit: int | None = None):
    items = load_bbh_movie_recommendation(data_path, limit=limit)
    template = Path(prompt_path).read_text(encoding="utf-8")
    rows = []
    for ex in tqdm(items, desc="CoT (silent)"):
        prompt = template.replace("{{question}}", ex["prompt"])
        resp = call_model(prompt)
        pred = extract_letter(resp)
        rows.append({"pred": pred, "gold": ex["gold"], "ok": int(pred == ex["gold"]), "raw": resp})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    acc = sum(r["ok"] for r in rows) / len(rows)
    print(f"CoT accuracy: {acc:.3f}  (N={len(rows)})")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    """
    Usage:
      python src/run_manual.py data/movie_recommendation.json outputs/manual_cot.json
      # limit to first N items:
      python src/run_manual.py data/movie_recommendation.json outputs/manual_cot_2.json prompts/cot_silent_movie_rec.txt 2
    """
    data_path = sys.argv[1]
    out_path = sys.argv[2]
    prompt_path = sys.argv[3] if len(sys.argv) > 3 else "prompts/cot_silent_movie_rec.txt"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    main(data_path, out_path, prompt_path, limit)
