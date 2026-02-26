import json
import re
import json
import re

def _extract_balanced_object(text: str, start_idx: int):
    """
    Returns a balanced JSON object substring starting at start_idx (which must point to '{').
    Handles braces inside quoted strings and escaped quotes.
    """
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start_idx, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx:i+1]

    return None


def extract_scores(raw_generation: str):
    """
    Returns (s1, s2, s3) or None.
    Supports:
      - "scores": {"Score1":[4,3,3]}
      - "scores": {"Score1":"4","Score2":"3","Score3":"3"}
    Also survives extra trailing garbage after the first JSON.
    """

    if not raw_generation:
        return None

    # 1) Try to locate and parse the "scores" object directly (most robust)
    scores_key = re.search(r'"scores"\s*:\s*\{', raw_generation)
    if scores_key:
        obj_start = scores_key.end() - 1  # points to '{'
        scores_obj_str = _extract_balanced_object(raw_generation, obj_start)
        if scores_obj_str:
            try:
                scores_obj = json.loads(scores_obj_str)
                # Case A: Score1 list
                if isinstance(scores_obj.get("Score1"), list) and len(scores_obj["Score1"]) == 3:
                    return tuple(int(x) for x in scores_obj["Score1"])
                # Case B: Score1/2/3
                if all(k in scores_obj for k in ("Score1", "Score2", "Score3")):
                    return (int(scores_obj["Score1"]), int(scores_obj["Score2"]), int(scores_obj["Score3"]))
            except Exception:
                pass

    # 2) Fallback: just grab the first [a,b,c] triplet after Score/Score1 (covers your ```json {"Score":[...]} ``` too)
    m = re.search(r'(?:Score1|Score)\s*"?\s*[:=]\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', raw_generation)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # 3) Fallback: separate Score1/Score2/Score3 anywhere
    m1 = re.search(r'Score1"?\s*[:=]\s*"?(\d+)"?', raw_generation)
    m2 = re.search(r'Score2"?\s*[:=]\s*"?(\d+)"?', raw_generation)
    m3 = re.search(r'Score3"?\s*[:=]\s*"?(\d+)"?', raw_generation)
    if m1 and m2 and m3:
        return (int(m1.group(1)), int(m2.group(1)), int(m3.group(1)))

    return None

def process_jsonl(input_path, output_path):
    import json
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        print(infile)
        for line in infile:
            print(line)
            try:
                row = json.loads(line)
                raw_generation = row.get("raw_generation", "")
                print(raw_generation)
                scores = extract_scores(raw_generation)

                row["extracted_scores"] = scores
                outfile.write(json.dumps(row) + "\n")

            except Exception as e:
                print(e)

                continue

process_jsonl("/work/PRTNR/CHUV/DIR/jraisaro/llm4chuv/LLM_Judge/output_with_generated_3d.jsonl", "/work/PRTNR/CHUV/DIR/jraisaro/llm4chuv/LLM_Judge/output_with_generated_3d_new.jsonl")

