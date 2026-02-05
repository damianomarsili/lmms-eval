import io
import os
import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image


DEFAULT_REPO_ID = "aadarsh99/ConvSeg"
_CACHE_DIR: Optional[str] = None
_CACHE_REPO: Optional[str] = None


def _get_repo_id(lmms_eval_specific_kwargs: Optional[dict[str, Any]]) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    return lmms_eval_specific_kwargs.get("repo_id") or os.getenv("CONVSEG_REPO") or DEFAULT_REPO_ID


def _ensure_cache_dir(repo_id: str) -> str:
    global _CACHE_DIR, _CACHE_REPO
    if _CACHE_DIR is not None and _CACHE_REPO == repo_id:
        return _CACHE_DIR

    cache_override = os.getenv("CONVSEG_CACHE_DIR", "").strip()
    if cache_override:
        if not os.path.isdir(cache_override):
            raise FileNotFoundError(f"CONVSEG_CACHE_DIR does not exist: {cache_override}")
        _CACHE_DIR = cache_override
        _CACHE_REPO = repo_id
        return _CACHE_DIR

    _CACHE_DIR = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir_use_symlinks=False)
    _CACHE_REPO = repo_id
    return _CACHE_DIR


def _open_from_hf_path(rel_path: str, repo_id: str) -> Image.Image:
    cache_dir = _ensure_cache_dir(repo_id)
    candidates = [rel_path]
    if not rel_path.startswith("data/"):
        candidates.append(os.path.join("data", rel_path))

    for cand in candidates:
        local_path = os.path.join(cache_dir, cand)
        if os.path.isfile(local_path):
            return Image.open(local_path)

    msg = f"Could not resolve image path '{rel_path}' from {repo_id}"
    raise FileNotFoundError(msg)


def _to_pil_image(obj: Any, force_rgb: bool = False, repo_id: Optional[str] = None) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB") if force_rgb else obj
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            img = Image.open(io.BytesIO(obj["bytes"]))
            return img.convert("RGB") if force_rgb else img
        if obj.get("path"):
            path = obj["path"]
            if os.path.isfile(path):
                img = Image.open(path)
                return img.convert("RGB") if force_rgb else img
            if repo_id:
                img = _open_from_hf_path(path, repo_id)
                return img.convert("RGB") if force_rgb else img
            raise FileNotFoundError(f"Image path does not exist: {path}")
    if isinstance(obj, (bytes, bytearray)):
        img = Image.open(io.BytesIO(obj))
        return img.convert("RGB") if force_rgb else img
    if isinstance(obj, str):
        if os.path.isfile(obj):
            img = Image.open(obj)
            return img.convert("RGB") if force_rgb else img
        if repo_id:
            img = _open_from_hf_path(obj, repo_id)
            return img.convert("RGB") if force_rgb else img
    raise TypeError(f"Unsupported image type: {type(obj)}")


def convseg_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> List[Any]:
    repo_id = _get_repo_id(lmms_eval_specific_kwargs)
    _ensure_cache_dir(repo_id)
    doc["_convseg_repo"] = repo_id
    image = _to_pil_image(doc["image"], force_rgb=True, repo_id=repo_id)
    return [image]


def convseg_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    repo_id = _get_repo_id(lmms_eval_specific_kwargs)
    doc["_convseg_repo"] = repo_id
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    post_prompt_box = lmms_eval_specific_kwargs.get("post_prompt_box", "")
    mode = os.getenv("CONVSEG_MODE", "").strip().lower()
    if not mode:
        mode = os.getenv("MODE", "").strip().lower()
    if not mode:
        mode = lmms_eval_specific_kwargs.get("mode", "point").lower().strip()
    if mode == "box" and post_prompt_box:
        post_prompt = post_prompt_box

    prompt = str(doc.get("prompt", "")).strip()
    prompt = re.sub(r"^(\s*)segment\b", r"\1Locate", prompt, flags=re.IGNORECASE)

    doc["_convseg_mode"] = mode

    parts = [pre_prompt, prompt, post_prompt]
    return "\n".join([part for part in parts if part])


def convseg_doc_to_target(doc: Dict[str, Any]) -> str:
    return ""


def _strip_think_prefix(text: str) -> str:
    if not isinstance(text, str):
        return text
    if re.search(r"</(?:plan|think)>", text, flags=re.IGNORECASE):
        return re.split(r"</(?:plan|think)>", text, flags=re.IGNORECASE)[-1].strip()
    return re.sub(r"(?is)^\s*<(?:plan|think)>.*?</(?:plan|think)>\s*", "", text, count=1)


def _extract_loc_payload(text: str) -> Optional[str]:
    matches = re.findall(r"(?is)<loc>(.*?)</loc>", text)
    if not matches:
        return None
    return matches[-1].strip()


def _parse_loc_points(payload: str, mode: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for entry in payload.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            _, entry = entry.split(":", 1)
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", entry)]
        if mode == "point":
            if len(nums) >= 2:
                points.append((nums[0], nums[1]))
        elif mode == "box":
            if len(nums) >= 4:
                x1, y1, x2, y2 = nums[:4]
                points.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        else:
            raise ValueError(f"mode must be 'point' or 'box', got {mode}")
    return points


def _parse_loc_boxes(payload: str) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    entries = [e.strip() for e in payload.split(";") if e.strip()]
    if not entries:
        entries = [payload.strip()]
    for entry in entries:
        if ":" in entry:
            _, entry = entry.split(":", 1)
        nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", entry)]
        for i in range(0, len(nums) - 3, 4):
            boxes.append((nums[i], nums[i + 1], nums[i + 2], nums[i + 3]))
    return boxes


def _infer_mode_from_payload(payload: str) -> str:
    for entry in payload.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            _, entry = entry.split(":", 1)
        nums = re.findall(r"-?\d+(?:\.\d+)?", entry)
        if len(nums) >= 4:
            return "box"
    return "point"


def _normalize_to_pixel(point: Tuple[float, float], width: int, height: int) -> Optional[Tuple[int, int]]:
    x, y = point
    if width <= 0 or height <= 0:
        return None
    if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        return None

    x_px = int(round((x / 1000.0) * width))
    y_px = int(round((y / 1000.0) * height))
    x_px = max(0, min(width - 1, x_px))
    y_px = max(0, min(height - 1, y_px))
    return x_px, y_px


def _normalize_box_to_pixels(
    box: Tuple[float, float, float, float], width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = box
    # normalize from [0,1000] into pixel coords
    x1 = (x1 / 1000.0) * width
    x2 = (x2 / 1000.0) * width
    y1 = (y1 / 1000.0) * height
    y2 = (y2 / 1000.0) * height
    # ensure order
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # clip to image bounds
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _mask_contains_point(mask_img: Image.Image, point: Tuple[float, float]) -> bool:
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")
    width, height = mask_img.size
    pixel = _normalize_to_pixel(point, width, height)
    if pixel is None:
        return False
    return mask_img.getpixel(pixel) > 0


def _mask_to_bboxes(mask_img: Image.Image) -> List[Tuple[float, float, float, float]]:
    mask = np.array(mask_img.convert("L")) > 0
    if mask.ndim != 2:
        mask = mask[:, :, 0]
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
            # Use exclusive max coords for area computation
            bboxes.append((float(minx), float(miny), float(maxx + 1), float(maxy + 1)))

    # Filter tiny components (noise)
    min_area_px = float(os.getenv("CONVSEG_MIN_GT_BOX_AREA", "64"))
    min_side_px = float(os.getenv("CONVSEG_MIN_GT_BOX_SIDE", "4"))
    filtered: List[Tuple[float, float, float, float]] = []
    for box in bboxes:
        w = max(0.0, box[2] - box[0])
        h = max(0.0, box[3] - box[1])
        area = w * h
        if area < min_area_px or min(w, h) < min_side_px:
            continue
        filtered.append(box)

    if filtered:
        filtered = _merge_contained_boxes(filtered)
        return filtered

    # If everything got filtered, fall back to largest box
    if bboxes:
        bboxes = _merge_contained_boxes(bboxes)
        bboxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return [bboxes[0]]
    return []


def _merge_contained_boxes(
    boxes: List[Tuple[float, float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    if len(boxes) <= 1:
        return boxes
    # Sort by area descending so larger boxes come first
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


def convseg_box_miou(results: List[Any]) -> float:
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


def convseg_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    prediction = _strip_think_prefix(results[0] if results else "")
    payload = _extract_loc_payload(prediction)
    if not payload:
        return {"convseg_point_acc": 0.0, "convseg_box_miou": {"ious": []}}

    mode = os.getenv("CONVSEG_MODE", "").strip().lower()
    if not mode:
        mode = os.getenv("MODE", "").strip().lower()
    if not mode:
        mode = _infer_mode_from_payload(payload)
    if mode == "box":
        repo_id = str(doc.get("_convseg_repo") or DEFAULT_REPO_ID)
        mask = _to_pil_image(doc["mask"], force_rgb=False, repo_id=repo_id)
        gt_boxes = _mask_to_bboxes(mask)
        pred_boxes_raw = _parse_loc_boxes(payload)
        pred_boxes: List[Tuple[float, float, float, float]] = []
        width, height = mask.size
        for box in pred_boxes_raw:
            norm_box = _normalize_box_to_pixels(box, width, height)
            if norm_box is not None:
                pred_boxes.append(norm_box)
        ious = _match_boxes(gt_boxes, pred_boxes)
        return {"convseg_point_acc": 0.0, "convseg_box_miou": {"ious": ious}}

    points = _parse_loc_points(payload, mode)
    if not points:
        return {"convseg_point_acc": 0.0, "convseg_box_miou": {"ious": []}}

    repo_id = str(doc.get("_convseg_repo") or DEFAULT_REPO_ID)
    mask = _to_pil_image(doc["mask"], force_rgb=False, repo_id=repo_id)
    point = points[0]
    return {
        "convseg_point_acc": 1.0 if _mask_contains_point(mask, point) else 0.0,
        "convseg_box_miou": {"ious": []},
    }
