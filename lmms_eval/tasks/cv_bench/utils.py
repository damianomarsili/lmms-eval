import re
from typing import Dict, Iterable, List, Optional

from loguru import logger as eval_logger
from PIL import Image

CHOICE_LABELS: List[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Prefer explicit labels anywhere (e.g., "(C)" or "[B]"), then "Answer: C", etc.
_LABEL_PATTERNS = [
    r"[\(\[]\s*([A-Za-z])\s*[\)\]]",  # (C) or [C]
    r"(?i)\b(?:answer|ans|prediction|pred|output)\s*[:\-]?\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?",  # Answer: C / Ans (B)
    r"^\s*([A-Za-z])\s*[\)\].:]",  # C)  C.  C:  C]
    r"^\s*([A-Za-z])\s*$",  # just a single letter like "C"
]

def doc_to_visual(doc):
    image = doc["image"]
    if isinstance(image, Image.Image):
        return [image.convert("RGB")]
    return [Image.open(image).convert("RGB")]

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc.get("prompt")
    return prompt.strip()

def _choice_map(choices: Iterable[str]) -> Dict[str, str]:
    mapping = {}
    for idx, choice in enumerate(choices):
        if idx >= len(CHOICE_LABELS):
            break
        label = CHOICE_LABELS[idx]
        mapping[label] = _normalize_text(choice)
    return mapping

def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()

def _extract_label(text: Optional[str], valid_labels: Iterable[str]) -> Optional[str]:
    """
    Robustly extract a multiple-choice label from free-form model output.

    Priority:
      1) Parenthesized/bracketed label anywhere: "(C)" / "[B]"
      2) Prefixes like "Answer: C", "Ans (D)", "Prediction - A"
      3) Leading formats like "C)", "C.", "C:"
      4) A single-letter line: "D"
      5) Token-level isolated single letter elsewhere: ... C ...
      6) Fallback: if the entire cleaned output equals a label
    """
    if not isinstance(text, str):
        return None
    valid_labels = set(valid_labels)

    # 1–4: targeted patterns
    for pat in _LABEL_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).upper()
            if cand in valid_labels:
                return cand

    # 5) token-level isolated single letter (avoid grabbing 'A' in 'Answer')
    # Use word boundaries but ensure next/prev chars aren't letters.
    m = re.search(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", text)
    if m:
        cand = m.group(1).upper()
        if cand in valid_labels:
            return cand

    # 6) fallback: whole string equals a label
    text_clean = _normalize_text(text)
    upper_clean = text_clean.upper()
    if upper_clean in valid_labels:
        return upper_clean

    return None

def _match_choice_by_text(text: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text_clean = _normalize_text(text)
    for label, choice_text in mapping.items():
        if text_clean == choice_text:
            return label
        if choice_text and choice_text in text_clean:
            return label
    return None

def _resolve_label(raw_value: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    valid_labels = mapping.keys()
    label = _extract_label(raw_value, valid_labels)
    if label is not None:
        return label
    return _match_choice_by_text(raw_value, mapping)

def interleave_process_results(doc, results):
    choices = list(doc["choices"])
    choice_mapping = _choice_map(choices)
    gold_label = _resolve_label(doc["answer"], choice_mapping)
    pred_raw = results[0] if results else ""
    pred_label = _resolve_label(pred_raw, choice_mapping)
    correct = int(gold_label is not None and pred_label == gold_label)
    return {
        "overall_score": {
            "correct": correct,
            "type": str(doc["type"]).strip(),
            "source": str(doc["source"]).strip(),
            "gold": gold_label,
            "pred": pred_label,
        }
    }

def overall_score(results):
    buckets = {
        "2d_ade": {"correct": 0, "total": 0},
        "2d_coco": {"correct": 0, "total": 0},
        "3d_omni": {"correct": 0, "total": 0},
    }

    for result in results:
        dtype = result.get("type", "").strip().upper()
        source = result.get("source", "").strip().lower()
        correct = int(result.get("correct", 0))

        key = None
        if dtype == "2D":
            if "ade" in source:
                key = "2d_ade"
            elif "coco" in source:
                key = "2d_coco"
        elif dtype == "3D" and "omni" in source:
            key = "3d_omni"

        if key:
            buckets[key]["total"] += 1
            buckets[key]["correct"] += correct

    acc_2d_ade = buckets["2d_ade"]["correct"] / buckets["2d_ade"]["total"] if buckets["2d_ade"]["total"] else 0.0
    acc_2d_coco = buckets["2d_coco"]["correct"] / buckets["2d_coco"]["total"] if buckets["2d_coco"]["total"] else 0.0
    acc_3d_omni = buckets["3d_omni"]["correct"] / buckets["3d_omni"]["total"] if buckets["3d_omni"]["total"] else 0.0

    acc_2d_components = []
    if buckets["2d_ade"]["total"]:
        acc_2d_components.append(acc_2d_ade)
    if buckets["2d_coco"]["total"]:
        acc_2d_components.append(acc_2d_coco)
    acc_2d = sum(acc_2d_components) / len(acc_2d_components) if acc_2d_components else None

    scores_to_average = []
    if acc_2d is not None:
        scores_to_average.append(acc_2d)
    if buckets["3d_omni"]["total"]:
        scores_to_average.append(acc_3d_omni)

    overall = sum(scores_to_average) / len(scores_to_average) if scores_to_average else 0.0

    eval_logger.info(
        "CV-Bench accuracy | 2D ADE: {:.3f} ({}/{}) | 2D COCO: {:.3f} ({}/{}) | 3D Omni: {:.3f} ({}/{}) | Overall: {:.3f}".format(
            acc_2d_ade,
            buckets["2d_ade"]["correct"],
            buckets["2d_ade"]["total"],
            acc_2d_coco,
            buckets["2d_coco"]["correct"],
            buckets["2d_coco"]["total"],
            acc_3d_omni,
            buckets["3d_omni"]["correct"],
            buckets["3d_omni"]["total"],
            overall,
        )
    )

    return overall
