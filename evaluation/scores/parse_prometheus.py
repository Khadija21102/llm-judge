import json
from typing import Any, Dict, Optional, Tuple


def extract_scores_from_mixed_text(s: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Scan `s` for balanced JSON objects, parse each safely,
    and return the scores from the LAST object that has numeric scores.

    This mirrors the logic in prometheus-clinicians-llm.py. :contentReference[oaicite:1]{index=1}
    """
    if not isinstance(s, str) or not s:
        return None, None, None

    # Track brace positions while respecting strings/escapes
    starts = []
    in_str = False
    escape = False
    candidates = []

    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                starts.append(i)
            elif ch == "}" and starts:
                start = starts.pop()
                obj = s[start : i + 1]
                # quick pre-filter so we don't try to parse everything
                if '"scores"' in obj and '"feedback"' in obj:
                    candidates.append(obj)

    # Walk candidates from LAST to FIRST, pick first that parses and has numeric scores
    keys = ["Alignment_with_Guidelines", "Relevance_and_completeness", "Harmlessness"]
    for obj in reversed(candidates):
        try:
            parsed = json.loads(obj)
        except json.JSONDecodeError:
            continue

        # If the model sometimes returns a list, unwrap it
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]

        if not isinstance(parsed, dict):
            continue

        scores = parsed.get("scores")
        if not isinstance(scores, dict):
            continue

        # Require integers (filters out schema placeholders)
        if all(k in scores and isinstance(scores[k], int) for k in keys):
            return scores[keys[0]], scores[keys[1]], scores[keys[2]]

    return None, None, None


def parse_prometheus(record: Dict[str, Any], ref_row: Any = None) -> Tuple[Optional[str], Optional[float]]:
    """
    Prometheus parser.

    Inputs (per your script): :contentReference[oaicite:2]{index=2}
      - record["generated_output"]: long text containing JSON fragments with {"feedback":..., "scores": {...}}
      - record.get("response"): the answer text to merge on ("Answer" column)

    Returns:
      (Answer, llm_mean)
    """
    answer = record.get("response")
    if isinstance(answer, str):
        answer = answer.strip()
    else:
        answer = None

    raw_output = record.get("generated_output", "")
    a, r, h = extract_scores_from_mixed_text(raw_output)

    if None not in (a, r, h):
        mean_score = (a + r + h) / 3.0
    else:
        mean_score = None

    return answer, mean_score

