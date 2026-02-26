import re
from typing import Optional, Tuple, Dict, Any


def _extract_meditron_axis_scores(response_text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract numeric scores for:
      - Alignment with guidelines
      - Relevance and Completeness
      - Harmlessness

    Returns (alignment, relevance, harmlessness) as ints in [0-9] (as parsed), or None if not found.
    This mirrors your regex approach in meditron-clinicians-llm.py. :contentReference[oaicite:1]{index=1}
    """
    patterns = {
        "alignment": r"Alignment with guidelines[\s\S]*?(?:[Ss]core[^\d]*)?(\d)",
        "relevance": r"Relevance\s*(?:and|&)\s*Completeness[\s\S]*?(?:[Ss]core[^\d]*)?(\d)",
        "harmlessness": r"Harmlessness[\s\S]*?(?:[Ss]core[^\d]*)?(\d)",
    }
    flags = re.IGNORECASE | re.DOTALL

    def find_score(p: str) -> Optional[int]:
        m = re.search(p, response_text, flags)
        return int(m.group(1)) if m else None

    return (
        find_score(patterns["alignment"]),
        find_score(patterns["relevance"]),
        find_score(patterns["harmlessness"]),
    )


def _extract_response_tag(text: str) -> Optional[str]:
    """
    Extracts content inside <response>...</response> (case-insensitive, whitespace tolerant).
    Returns stripped string or None if not found.
    """
    if not isinstance(text, str):
        return None
    m = re.search(r"<\s*response\s*>(.*?)<\s*/\s*response\s*>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def parse_meditron(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """
    Adapter for Meditron CHUV JSONL outputs.

    Expected fields (as in your meditron script): :contentReference[oaicite:2]{index=2}
      - record["response"] : model verbose output containing the 3 rubric sections + scores
      - record["request"]["body"]["messages"][0]["content"] : contains <response>ANSWER</response>

    Returns:
      (answer_text, llm_mean_score)

    Notes:
      - If any axis score is missing, llm_mean_score = None (same behavior as your script).
      - If <response> tag missing, answer_text = None.
    """
    # 1) Answer extraction
    answer_text = None
    try:
        request_content = record["request"]["body"]["messages"][0]["content"]
        answer_text = _extract_response_tag(request_content)
    except Exception:
        answer_text = None

    # 2) Score extraction
    llm_mean = None
    raw_output = record.get("response", "")
    if isinstance(raw_output, str) and raw_output.strip():
        try:
            a, r, h = _extract_meditron_axis_scores(raw_output)
            if None not in (a, r, h):
                llm_mean = (a + r + h) / 3.0
            else:
                llm_mean = None
        except Exception:
            llm_mean = None

    return answer_text, llm_mean

