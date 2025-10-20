from __future__ import annotations
import os, json, re, string, random, time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Reuse existing modules
from load_bbh_movie_rec import load_bbh_movie_recommendation
from model_api import call_model  # solver calls (short answers)

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

LETTER = re.compile(r"\(([A-E])\)")

# ---------------------------
# Basic text helpers
# ---------------------------
def extract_letter(text: str) -> str:
    m = LETTER.search(str(text).upper())
    return m.group(1) if m else ""

def normalize_spaces(s: str) -> str:
    return " ".join(s.strip().split())

def instruction_canonical(s: str) -> str:
    s = s.lower().strip().replace("—", "-")
    return " ".join(s.split())

def instruction_bigrams(s: str) -> set:
    toks = [t for t in instruction_canonical(s).translate(str.maketrans("", "", string.punctuation)).split() if t]
    return set(zip(toks, toks[1:])) if len(toks) >= 2 else set()

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

# ---------------------------
# Build inference prompts (solver)
# ---------------------------
def read_baseline_template(path: str = "prompts/baseline_movie_rec.txt") -> str:
    return Path(path).read_text(encoding="utf-8")

def build_inference_prompt(question: str, instruction_lines: List[str], baseline_template: str) -> str:
    """
    Insert an 'Instruction (Rubric):' block right before the {{question}} placeholder.
    Keeps the baseline template's 'return only (A–E)' instruction.
    """
    rubric = ""
    if instruction_lines:
        rubric = "Instruction (Rubric):\n" + "\n".join(f"- {normalize_spaces(r)}" for r in instruction_lines) + "\n\n"
    return baseline_template.replace("{{question}}", rubric + question)

# ---------------------------
# Meta-prompt construction (optimizer LLM)
# ---------------------------
def format_examples_for_calibration(items: List[Dict[str, Any]], max_chars_per_ex: int = 420) -> str:
    # Show minimal yet informative snippets + gold letter
    lines = []
    for ex in items:
        q = ex["prompt"].strip()
        q = q if len(q) <= max_chars_per_ex else (q[:max_chars_per_ex] + " …")
        lines.append(q + f"\nGold: ({ex['gold']})\n")
    return "\n".join(lines)

def format_prior_block_from_history(best_history: List[Tuple[str, float]], cap: int = 20) -> str:
    if not best_history:
        return "(none yet)\n"
    # Keep top-20 by accuracy, then by shorter length, then lexicographic
    best_history = sorted(best_history, key=lambda t: (-t[1], len(t[0]), t[0]))[:cap]
    return "\n".join([f"- {normalize_spaces(r)}  [acc={acc:.3f}]" for r, acc in best_history]) + "\n"

def render_meta_prompt(template_path: str,
                       calib_block: str,
                       prior_block: str,
                       K: int) -> str:
    t = Path(template_path).read_text(encoding="utf-8")
    return (t.replace("{{calibration_block}}", calib_block)
             .replace("{{prior_block}}", prior_block)
             .replace("{{K}}", str(K)))

# ---------------------------
# Optimizer LLM call (OpenAI-compatible)
# ---------------------------
def _detect_provider():
    ds_key = os.getenv("DASHSCOPE_API_KEY", "")
    ms_key = os.getenv("MOONSHOT_API_KEY", "")
    oa_key = os.getenv("OPENAI_API_KEY", "")
    if ds_key:
        return {
            "name": "DashScope/Qwen",
            "base": os.getenv("DASHSCOPE_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            "key": ds_key,
            "model": os.getenv("DASHSCOPE_MODEL", "qwen-flash")
        }
    if ms_key:
        return {
            "name": "Moonshot/Kimi",
            "base": os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.ai/v1"),
            "key": ms_key,
            "model": os.getenv("MOONSHOT_MODEL", "kimi-k2-0905-preview")
        }
    if oa_key:
        return {
            "name": "OpenAI-compatible",
            "base": os.getenv("OPENAI_BASE", "https://api.openai.com/v1"),
            "key": oa_key,
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        }
    raise RuntimeError("No API key configured for optimizer. Set DASHSCOPE_API_KEY (recommended), or MOONSHOT_API_KEY, or OPENAI_API_KEY.")

def post_chat_for_generation(prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
    """
    Optimizer LLM call (paper-aligned): temperature defaults to 1.0 for diversity.
    """
    import requests
    cfg = _detect_provider()
    url = f"{cfg['base'].rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json"}
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You are a helpful model that follows instructions exactly."},
            {"role": "user", "content": prompt}
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    print(f"[optimizer] Provider={cfg['name']} Model={cfg['model']} Temp={temperature} MaxTokens={max_tokens}")
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=float(os.getenv("TIMEOUT", "60")))
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

# ---------------------------
# Candidate parsing & de-dup
# ---------------------------
def parse_candidates(raw: str, K: int) -> List[str]:
    cands = []
    for line in raw.splitlines():
        s = normalize_spaces(line)
        if not s: continue
        s = s.lstrip("-•0123456789. ").strip()
        if not s: continue
        cands.append(s)
    return cands[:K]

def dedup_candidates(cands: List[str], jacc_thresh: float = 0.85) -> List[str]:
    kept: List[str] = []
    kept_bi: List[set] = []
    seen: set = set()
    for c in cands:
        canon = instruction_canonical(c)
        if canon in seen:  # exact canonical match
            continue
        bi = instruction_bigrams(c)
        if any(jaccard(bi, kb) >= jacc_thresh for kb in kept_bi):
            continue
        kept.append(c)
        kept_bi.append(bi)
        seen.add(canon)
    return kept

# ---------------------------
# Evaluation on a slice (solver, temp=0.0) with progress
# ---------------------------
def eval_instruction_on_items(instruction_line: str,
                              items: List[Dict[str, Any]],
                              baseline_template_text: str,
                              show_progress: bool = True,
                              progress_label: str = "") -> Tuple[float, float, int, List[Dict[str, Any]]]:
    """
    Returns (accuracy, compliance_rate, n, preds_list).
    If show_progress, renders a tqdm bar (if available) or prints heartbeats every 25 items.
    """
    ok = 0; compliant = 0; preds = []
    rules = [r.strip() for r in instruction_line.split(";") if r.strip()]

    iterator = items
    if show_progress and _HAS_TQDM:
        iterator = tqdm(items, desc=progress_label or "Scoring", leave=False)

    start = time.time()
    for idx, ex in enumerate(iterator):
        prompt = build_inference_prompt(ex["prompt"], rules, baseline_template_text)
        resp = call_model(prompt)
        pred = extract_letter(resp)
        compliant += int(bool(pred))
        ok += int(pred == ex["gold"])
        preds.append({"pred": pred, "gold": ex["gold"], "ok": int(pred == ex["gold"]), "raw": resp})

        if show_progress and not _HAS_TQDM and (idx + 1) % 25 == 0:
            elapsed = time.time() - start
            print(f"  [{progress_label}] {idx+1}/{len(items)} items done in {elapsed:.1f}s")

    n = len(items)
    acc = ok / n if n else 0.0
    comp = compliant / n if n else 0.0
    return acc, comp, n, preds

def rank_scored_instructions(scored: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    # Sort by: accuracy desc, compliance desc, length asc, lexicographic asc
    def key(t):
        instr, acc, comp = t
        return (-acc, -comp, len(instr), instr)
    return sorted(scored, key=key)

# ---------------------------
# Splits & exemplar selection
# ---------------------------
def random_split(items: List[Dict[str, Any]], val_size: int, test_size: int, seed: int = 42):
    rng = random.Random(seed)
    idx = list(range(len(items))); rng.shuffle(idx)
    val_idx = idx[:val_size]
    test_idx = idx[val_size:val_size + test_size]
    val = [items[i] for i in val_idx]
    test = [items[i] for i in test_idx]
    return val, test, {"val_ids": val_idx, "test_ids": test_idx, "seed": seed}

def pick_exemplars(val_items: List[Dict[str, Any]], calib_size: int = 3, seed: int = 42):
    # Paper: 3 exemplars per step
    rng = random.Random(seed + 1)
    idx = list(range(len(val_items))); rng.shuffle(idx)
    idx = idx[:calib_size]
    return [val_items[i] for i in idx], idx
