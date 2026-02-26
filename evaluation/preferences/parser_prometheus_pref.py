import re
from typing import Any, Dict, Optional, Tuple


def _get_result(payload: Dict[str, Any]) -> Optional[str]:
    text = str(payload.get("generated_output", ""))

    p1 = re.compile(r"\[RESULT\]\s*([AB])\b", re.IGNORECASE)
    matches = p1.findall(text)
    if matches:
        return matches[-1].upper()

    p2 = re.compile(r"\bRESULT\W*([AB])\b", re.IGNORECASE)
    matches = p2.findall(text)
    if matches:
        return matches[-1].upper()

    p3 = re.compile(
        r"^(?:\s*(?:Winner|Better\s*response|Choice)\s*[:\-]\s*([AB])\s*)$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = p3.search(text)
    if m:
        return m.group(1).upper()

    return None


def parse_prometheus_pref(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    response_A = record.get("response_A")
    response_B = record.get("response_B")
    if isinstance(response_A, str):
        response_A = response_A.strip()
    else:
        response_A = None
    if isinstance(response_B, str):
        response_B = response_B.strip()
    else:
        response_B = None

    llm_result = _get_result(record)
    return response_A, response_B, llm_result

