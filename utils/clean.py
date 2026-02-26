## 
# This script reads a JSONL file, ensures that certain fields are present and are strings (coercing them if necessary), and writes out a cleaned JSONL file. It also counts and reports any lines that were skipped due to bad JSON formatting.
import json
import math
from typing import Any, Dict, Iterable, List, Optional

TEXT_FIELDS = [
    "instruction",
    "orig_instruction",
    "orig_response_A",
    "orig_response_B",
    "output",
    "orig_score",
]

def to_str(x: Any) -> str:
    # Normalize weird values into strings
    if x is None:
        return ""
    # Catch float NaNs (only if they exist in Python objects)
    if isinstance(x, float) and math.isnan(x):
        return ""
    # If it's already a string, keep it
    if isinstance(x, str):
        return x
    # Otherwise coerce (numbers, lists, dicts, bools, etc.)
    return str(x)

def clean_record(rec: Dict[str, Any], text_fields: Iterable[str]) -> Dict[str, Any]:
    for f in text_fields:
        if f in rec:
            rec[f] = to_str(rec[f])
        else:
            # Optional: enforce presence as empty string
            rec[f] = ""
    return rec

def clean_jsonl(
    in_path: str,
    out_path: str,
    text_fields: List[str] = TEXT_FIELDS,
    drop_bad_json: bool = True,
) -> None:
    bad = 0
    total = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError("Line is not a JSON object")
                rec = clean_record(rec, text_fields)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                bad += 1
                if not drop_bad_json:
                    raise

    print(f"Done. Wrote cleaned file to: {out_path}")
    print(f"Total lines read: {total}, bad lines skipped: {bad}")

if __name__ == "__main__":
    clean_jsonl("dataset_ref_based_pref_train_new_v2_with_explanations.jsonl", "dataset_ref_based_pref_train_new_v2_with_explanations_clean.jsonl")

