import re
from typing import Any, Dict, Optional, Tuple


def _extract_choice(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None

    patterns = [
        r'"Answer"\s*:\s*"([A-B])"',          # JSON style
        r"\*\*\s*\"?([A-B])\"?\s*\*\*",       # Markdown bold
        r"Response\s+([A-B])",                # "Response B"
        r"\b([A-B])\b"                        # fallback single-letter
    ]
    flags = re.IGNORECASE | re.DOTALL

    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1).upper()
    return None


def parse_meditron_pref(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (response_A, response_B, llm_result)
    """
    raw_output = record.get("response", "")
    llm_result = _extract_choice(raw_output)

    response_A = None
    response_B = None

    try:
        request_content = record["request"]["body"]["messages"][0]["content"]
        m1 = re.search(r"<\s*response_A\s*>(.*?)<\s*/\s*response_A\s*>", request_content,
                       re.DOTALL | re.IGNORECASE)
        m2 = re.search(r"<\s*response_B\s*>(.*?)<\s*/\s*response_B\s*>", request_content,
                       re.DOTALL | re.IGNORECASE)
        response_A = m1.group(1).strip() if m1 else None
        response_B = m2.group(1).strip() if m2 else None
    except Exception:
        pass

    return response_A, response_B, llm_result

