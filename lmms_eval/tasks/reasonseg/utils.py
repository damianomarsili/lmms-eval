import io
import os
import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


BBOX_2D_LINE_PATTERN = re.compile(
    r'^\s*label\s*=\s*"(?P<label>[^"\n]+?)"\s*,\s*'
    r"\[\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<y2>-?\d+(?:\.\d+)?)\s*\]\s*$",
    re.IGNORECASE,
)
FOUR_NUMBER_LIST_PATTERN = re.compile(
    r"(?<!\d)"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)"
    r"(?!\d)",
    re.IGNORECASE,
)


def _to_pil_image(obj: Any, force_rgb: bool = False) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB") if force_rgb else obj
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            img = Image.open(io.BytesIO(obj["bytes"]))
            return img.convert("RGB") if force_rgb else img
        for key in ("path", "file_name", "filename", "file_path", "filepath"):
            path = obj.get(key)
            if path and os.path.isfile(path):
                img = Image.open(path)
                return img.convert("RGB") if force_rgb else img
    if isinstance(obj, (bytes, bytearray)):
        img = Image.open(io.BytesIO(obj))
        return img.convert("RGB") if force_rgb else img
    if isinstance(obj, str) and os.path.isfile(obj):
        img = Image.open(obj)
        return img.convert("RGB") if force_rgb else img
    raise TypeError(f"Unsupported image type: {type(obj)}")


def _mask_to_bool_array(mask_obj: Any) -> np.ndarray:
    if isinstance(mask_obj, np.ndarray):
        arr = mask_obj
    elif isinstance(mask_obj, list):
        arr = np.asarray(mask_obj)
    elif isinstance(mask_obj, dict) and mask_obj.get("mask") is not None:
        return _mask_to_bool_array(mask_obj["mask"])
    else:
        arr = np.array(_to_pil_image(mask_obj, force_rgb=False))

    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask array, got shape={arr.shape}")
    return arr.astype(bool)


def _strip_think_prefix(text: str) -> str:
    if not isinstance(text, str):
        return text
    if re.search(r"</(?:plan|think)>", text, flags=re.IGNORECASE):
        return re.split(r"</(?:plan|think)>", text, flags=re.IGNORECASE)[-1].strip()
    return re.sub(r"(?is)^\s*<(?:plan|think)>.*?</(?:plan|think)>\s*", "", text, count=1)


def _extract_bbox_2d_payload(text: str) -> Optional[str]:
    matches = re.findall(r"(?is)<bbox_2d>(.*?)</bbox_2d>", text)
    if not matches:
        return None
    return matches[-1].strip()


def _parse_bbox_2d_boxes(payload: str) -> List[Tuple[float, float, float, float]]:
    strict_boxes: List[Tuple[float, float, float, float]] = []
    nonempty_line_count = 0
    strict_valid = True
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        nonempty_line_count += 1
        match = BBOX_2D_LINE_PATTERN.fullmatch(line)
        if match is None:
            strict_valid = False
            break
        strict_boxes.append(
            (
                float(match.group("x1")),
                float(match.group("y1")),
                float(match.group("x2")),
                float(match.group("y2")),
            )
        )
    if strict_valid and nonempty_line_count > 0:
        return strict_boxes

    boxes: List[Tuple[float, float, float, float]] = []
    for match in FOUR_NUMBER_LIST_PATTERN.finditer(payload):
        boxes.append(
            (
                float(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
                float(match.group(4)),
            )
        )
    return boxes


def _normalize_box_to_pixels(
    box: Tuple[float, float, float, float], width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = box
    x1 = (x1 / 1000.0) * width
    x2 = (x2 / 1000.0) * width
    y1 = (y1 / 1000.0) * height
    y2 = (y2 / 1000.0) * height
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _merge_contained_boxes(
    boxes: List[Tuple[float, float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    if len(boxes) <= 1:
        return boxes
    boxes_sorted = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: List[Tuple[float, float, float, float]] = []
    for box in boxes_sorted:
        contained = False
        for big in kept:
            if box[0] >= big[0] and box[1] >= big[1] and box[2] <= big[2] and box[3] <= big[3]:
                contained = True
                break
        if not contained:
            kept.append(box)
    return kept


def _mask_to_bboxes(mask: np.ndarray) -> List[Tuple[float, float, float, float]]:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask.shape}")
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    bboxes: List[Tuple[float, float, float, float]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            minx = maxx = x
            miny = maxy = y
            q = deque([(y, x)])
            visited[y, x] = True
            while q:
                cy, cx = q.popleft()
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            bboxes.append((float(minx), float(miny), float(maxx + 1), float(maxy + 1)))
    return _merge_contained_boxes(bboxes)


def _compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _match_boxes(
    gt_boxes: List[Tuple[float, float, float, float]],
    pred_boxes: List[Tuple[float, float, float, float]],
) -> List[float]:
    if not gt_boxes:
        return []
    if not pred_boxes:
        return [0.0 for _ in gt_boxes]

    gt_centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in gt_boxes]
    pred_centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in pred_boxes]

    pairs: List[Tuple[float, int, int]] = []
    for gi, (gx, gy) in enumerate(gt_centers):
        for pi, (px, py) in enumerate(pred_centers):
            dist = (gx - px) ** 2 + (gy - py) ** 2
            pairs.append((dist, gi, pi))
    pairs.sort(key=lambda x: x[0])

    assigned_gt = set()
    assigned_pred = set()
    ious = [0.0 for _ in gt_boxes]
    for _, gi, pi in pairs:
        if gi in assigned_gt or pi in assigned_pred:
            continue
        assigned_gt.add(gi)
        assigned_pred.add(pi)
        ious[gi] = _compute_iou(gt_boxes[gi], pred_boxes[pi])
        if len(assigned_gt) == len(gt_boxes) or len(assigned_pred) == len(pred_boxes):
            break
    return ious


def reasonseg_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> List[Any]:
    del lmms_eval_specific_kwargs
    image = _to_pil_image(doc["image"], force_rgb=True)
    return [image]


def reasonseg_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    post_prompt_box = lmms_eval_specific_kwargs.get("post_prompt_box", "")
    mode = os.getenv("REASONSEG_MODE", "").strip().lower()
    if not mode:
        mode = os.getenv("MODE", "").strip().lower()
    if not mode:
        mode = lmms_eval_specific_kwargs.get("mode", "box").lower().strip()
    if mode == "box" and post_prompt_box:
        post_prompt = post_prompt_box

    query = str(doc.get("text", "")).strip()
    query = re.sub(r"^(\s*)segment\b", r"\1Locate", query, flags=re.IGNORECASE)
    doc["_reasonseg_mode"] = mode

    parts = [pre_prompt, query, post_prompt]
    return "\n".join([part for part in parts if part])


def reasonseg_doc_to_target(doc: Dict[str, Any]) -> str:
    return ""


def reasonseg_box_miou(results: List[Any]) -> float:
    all_ious: List[float] = []
    for result in results:
        if isinstance(result, dict):
            ious = result.get("ious")
            if isinstance(ious, list):
                all_ious.extend([float(v) for v in ious])
        elif isinstance(result, (int, float)):
            all_ious.append(float(result))
    if not all_ious:
        return 0.0
    return sum(all_ious) / len(all_ious)


def reasonseg_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    prediction = _strip_think_prefix(results[0] if results else "")
    payload = _extract_bbox_2d_payload(prediction)
    text_for_parsing = payload if payload else prediction

    pred_boxes_raw = _parse_bbox_2d_boxes(text_for_parsing)
    if not pred_boxes_raw:
        return {"reasonseg_box_miou": {"ious": []}}

    mask = _mask_to_bool_array(doc["mask"])
    height, width = mask.shape
    gt_boxes = _mask_to_bboxes(mask)

    pred_boxes: List[Tuple[float, float, float, float]] = []
    for box in pred_boxes_raw:
        norm_box = _normalize_box_to_pixels(box, width, height)
        if norm_box is not None:
            pred_boxes.append(norm_box)

    ious = _match_boxes(gt_boxes, pred_boxes)
    return {"reasonseg_box_miou": {"ious": ious}}
