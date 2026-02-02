import ast
import re
from typing import Any, Dict, List, Optional, Tuple


YES_SET = {"yes", "y", "yeah", "yep", "true", "1"}
NO_SET = {"no", "n", "nope", "false", "0"}


def robospatial_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    # Use the RGB image; depth/mask are unused for text scoring here.
    img = doc["img"]
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    return [img]


def robospatial_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    return doc["question"].strip()


def robospatial_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc["answer"])


def _normalize_yes_no(text: str) -> str:
    text = _strip_think_prefix(text)
    cleaned = text.strip().lower()
    if not cleaned:
        return ""
    match = re.findall(r"\b(yes|no)\b", cleaned)
    if match:
        return match[-1]
    token = re.sub(r"[^a-z0-9]+", "", cleaned)
    if token in YES_SET:
        return "yes"
    if token in NO_SET:
        return "no"
    parts = cleaned.split()
    return parts[0] if parts else ""


def _extract_last_answer_tag(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    open_matches = list(re.finditer(r"(?is)<answer>", text))
    if not open_matches:
        return None
    match = open_matches[-1]
    content_start = match.end()
    close_match = re.search(r"(?is)</answer>", text[content_start:])
    if close_match:
        content_end = content_start + close_match.start()
        return text[content_start:content_end].strip()
    next_tag = re.search(r"(?is)<(reason|depth|loc|verifier|answer)>", text[content_start:])
    if next_tag:
        content_end = content_start + next_tag.start()
        return text[content_start:content_end].strip()
    return text[content_start:].strip()
    return None


def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    # Ray-casting algorithm
    num = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, num + 1):
        p2x, p2y = poly[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _extract_point(text: str) -> Optional[Tuple[float, float]]:
    # Try tuple format (x, y)
    tuple_match = re.search(r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", text)
    if tuple_match:
        try:
            return float(tuple_match.group(1)), float(tuple_match.group(2))
        except ValueError:
            pass

    # Try list format [x, y]
    list_match = re.search(r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", text)
    if list_match:
        try:
            return float(list_match.group(1)), float(list_match.group(2))
        except ValueError:
            pass

    # Fall back to first bracketed content
    match = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if match:
        list_content = match.group(1)
        list_content = re.sub(r",(\S)", r", \1", list_content).strip()
        if list_content.endswith(","):
            list_content = list_content[:-1]
        list_str = "[" + list_content + "]"
        try:
            gen_val = ast.literal_eval(list_str)
        except (SyntaxError, ValueError):
            tuple_match = re.search(r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", list_content)
            if tuple_match:
                try:
                    return float(tuple_match.group(1)), float(tuple_match.group(2))
                except ValueError:
                    return None
            return None

        if isinstance(gen_val, list):
            if len(gen_val) == 2 and all(isinstance(v, (int, float)) for v in gen_val):
                gen_point = tuple(gen_val)
            elif gen_val and isinstance(gen_val[0], tuple):
                gen_point = gen_val[0]
            elif gen_val and isinstance(gen_val[0], list) and len(gen_val[0]) == 2:
                gen_point = tuple(gen_val[0])
            else:
                return None
        elif isinstance(gen_val, tuple):
            gen_point = gen_val
        else:
            return None

        try:
            return float(gen_point[0]), float(gen_point[1])
        except (ValueError, TypeError, IndexError):
            return None
    return None


def _normalize_point_to_unit(point: Tuple[float, float], img: Any) -> Tuple[float, float]:
    """
    Normalize pixel coordinates to [0,1] using image width/height when the values
    fall outside the unit range. Leaves already-normalized coordinates untouched.
    """
    x, y = point
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return point

    if img is None or not hasattr(img, "size"):
        return point
    try:
        width, height = img.size
        if width and height:
            return (x / float(width), y / float(height))
    except Exception:
        pass
    return point


def robospatial_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    prediction = _strip_think_prefix(results[0] if results else "")
    tagged_answer = _extract_last_answer_tag(prediction)
    if tagged_answer is not None:
        prediction = tagged_answer

    gt = doc["answer"]

    if doc.get("category", "").lower() in {"compatibility", "configuration"} or gt.strip().lower() in {"yes", "no"}:
        normalized_pred = _normalize_yes_no(prediction)
        score = 1.0 if normalized_pred == gt.strip().lower() else 0.0
        return {"exact_match": score}

    # Context: polygon-based scoring
    try:
        gt_polygon = ast.literal_eval(gt)
    except Exception:
        return {"exact_match": 0.0}

    if not isinstance(gt_polygon, list) or len(gt_polygon) < 3:
        return {"exact_match": 0.0}

    point = _extract_point(prediction.lower())
    if point is not None:
        point = _normalize_point_to_unit(point, doc.get("img"))
    if point is None:
        return {"exact_match": 0.0}

    inside = _point_in_polygon(point[0], point[1], gt_polygon)
    return {"exact_match": 1.0 if inside else 0.0}


def _strip_think_prefix(text: str) -> str:
    """
    Drop a leading <plan>...</plan> or <think>...</think> block if present and return the remainder.
    If only a closing tag is present, take text after it.
    """
    if not isinstance(text, str):
        return text
    if re.search(r"</(?:plan|think)>", text, flags=re.IGNORECASE):
        return re.split(r"</(?:plan|think)>", text, flags=re.IGNORECASE)[-1].strip()
    return re.sub(r"(?is)^\s*<(?:plan|think)>.*?</(?:plan|think)>\s*", "", text, count=1)
