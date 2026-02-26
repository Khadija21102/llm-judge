import re
import json
from typing import Any, Dict, Optional, Tuple

def _extract_json_text(s: str) -> Optional[str]:
    m = re.search(r"```(?:[\w-]+)?\s*(.*?)```", s, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1].strip()
    return None

def parse_gpt(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[Any]]:
    raw = None
    try:
        raw = record["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        pass

    if not isinstance(raw, str) or not raw.strip():
        for k in ("response", "generated_output", "raw_generation"):
            v = record.get(k)
            if isinstance(v, str) and v.strip():
                raw = v
                break

    if not isinstance(raw, str) or not raw.strip():
        return None

    json_text = _extract_json_text(raw)
    if not json_text:
        return None

    try:
        parsed = json.loads(json_text)
    except Exception:
        return None

    def get_score(key):
        v = parsed.get(key)
        if isinstance(v, dict) and "score" in v:
            return v["score"]
        return None

    a = get_score("Alignment with guidelines")
    r = get_score("Relevance and completeness")
    h = get_score("Harmlessness")

    if None in (a, r, h):
        return None

    return (float(a) + float(r) + float(h)) / 3.0


    json_text = _extract_json_text(raw)
    if not json_text:
        return None

    try:
        parsed = json.loads(json_text)
    except Exception:
        return None

    def get_score(key):
        v = parsed.get(key)
        if isinstance(v, dict) and "score" in v:
            return v["score"]
        return None

    a = get_score("Alignment with guidelines")
    r = get_score("Relevance and completeness")
    h = get_score("Harmlessness")

    if None in (a, r, h):
        return None

    return (float(a) + float(r) + float(h)) / 3.0

