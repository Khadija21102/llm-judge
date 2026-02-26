import json
import re
from typing import Any, Dict, Optional, Tuple


def _extract_json_text(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None

    m = re.search(r"```(?:[\w-]+)?\s*(.*?)```", s, flags=re.S | re.I)
    if m:
        return m.group(1).strip()

    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1].strip()

    return None


def parse_gpt_pref(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    raw = None
    try:
        raw = record["response"]["body"]["choices"][0]["message"]["content"]
    except Exception:
        pass

    if not isinstance(raw, str) or not raw.strip():
        raw = record.get("response", "")

    json_text = _extract_json_text(raw)
    if not json_text:
        return None

    try:
        parsed = json.loads(json_text)
    except Exception:
        return None

    ans = parsed.get("Answer")
    if isinstance(ans, str):
        ans = ans.strip().upper()
        if ans in {"A", "B"}:
            return ans

    return None

