import re
from typing import Any, Dict, Optional, Tuple


def _extract_choice(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    patterns = [
        r'"Answer"\s*:\s*"([A-B])"',     # JSON style
        r"\bAnswer\b\W*([A-B])\b",       # Answer: A
        r"\b([A-B])\b"                   # fallback
    ]
    flags = re.IGNORECASE | re.DOTALL

    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1).upper()
    return None


def parse_claude_pref(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Claude pref outputs typically in record["response"] in your script
    text = record.get("response", "")
    llm_result = _extract_choice(text)

    # response_A/B not present in this file usually -> return None; main will fill from ref_jsonl
    return llm_result

