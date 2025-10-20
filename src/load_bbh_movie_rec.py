# src/load_bbh_movie_rec.py
import json, re, string
from pathlib import Path
from typing import List, Dict, Any, Optional

LETTER_RE = re.compile(r"\(([A-E])\)")
OPTION_BLOCK_RE = re.compile(r"\(([A-E])\)\s*(.*?)(?=\([A-E]\)\s*|$)", re.DOTALL)
CANDIDATE_LIST_KEYS = ["examples", "instances", "data", "items", "test", "validation", "train"]

def _find_instances(obj: Any):
    if isinstance(obj, list): return obj
    if isinstance(obj, dict):
        for k in CANDIDATE_LIST_KEYS:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    return None

def _extract_options(prompt_text: str) -> Dict[str, str]:
    lower = prompt_text.lower()
    start = lower.find("options:")
    section = prompt_text[start + len("options:") :] if start != -1 else prompt_text
    pairs = OPTION_BLOCK_RE.findall(section)
    opts = {}
    for letter, text in pairs:
        cleaned = " ".join(text.strip().split()).rstrip(",")
        opts[letter] = cleaned
    return opts

def _normalize(s: str) -> str:
    s = " ".join(s.lower().split())
    tbl = str.maketrans("", "", string.punctuation)
    return s.translate(tbl)

def _gold_from_target(target: Any, options: Dict[str, str]) -> Optional[str]:
    # Case 1: explicit letter
    m = LETTER_RE.search(str(target))
    if m: return m.group(1)
    # Case 2: map title → letter
    if isinstance(target, str) and target.strip():
        norm_t = _normalize(target)
        # single-option match
        for L, txt in options.items():
            if _normalize(txt) == norm_t:
                return L
        # adjacent 2-option merge (handles "Monsters, Inc" split as A="Monsters", B="Inc")
        letters = sorted(options.keys())
        for i in range(len(letters)-1):
            L1, L2 = letters[i], letters[i+1]
            merged = f"{options[L1]}, {options[L2]}"
            if _normalize(merged) == norm_t:
                return L1
    return None

def load_bbh_movie_recommendation(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text)
    instances = _find_instances(raw)
    if instances is None:
        keys = list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(f"Could not find a list of instances in {path}. Top-level keys/type: {keys}")

    if limit is not None:
        instances = instances[:limit]  # <-- slice early

    out = []
    for i, ex in enumerate(instances):
        if not isinstance(ex, dict) or "input" not in ex:
            raise ValueError(f"Item {i} missing 'input' field: {ex!r}")
        inp = ex["input"]
        options = _extract_options(inp)
        if not options:
            raise ValueError(f"Item {i} has no parsed options. Prompt snippet: {inp[:120]!r}")

        tgt = ex.get("target") or ex.get("output")
        if tgt is None and isinstance(ex.get("targets"), list) and ex["targets"]:
            tgt = ex["targets"][0]
        if tgt is None:
            raise ValueError(f"Item {i} missing target/targets/output: {ex!r}")

        gold = _gold_from_target(tgt, options)
        if gold not in {"A","B","C","D","E"}:
            raise ValueError(f"Item {i}: could not resolve gold to a letter A–E. "
                             f"target={tgt!r}, options={options}")
        out.append({"prompt": inp, "gold": gold})
    return out