import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import snapshot_download
from PIL import Image


DEFAULT_REPO_ID = "moondream/refcoco-m"
_CACHE_DIR: Optional[str] = None
_CACHE_REPO: Optional[str] = None


def _get_repo_id(lmms_eval_specific_kwargs: Optional[dict[str, Any]]) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    return lmms_eval_specific_kwargs.get("repo_id") or os.getenv("REFCOCO_M_REPO") or DEFAULT_REPO_ID


def _ensure_cache_dir(repo_id: str) -> str:
    global _CACHE_DIR, _CACHE_REPO
    if _CACHE_DIR is not None and _CACHE_REPO == repo_id:
        return _CACHE_DIR

    cache_override = os.getenv("REFCOCO_M_CACHE_DIR", "").strip()
    if cache_override:
        if not os.path.isdir(cache_override):
            raise FileNotFoundError(f"REFCOCO_M_CACHE_DIR does not exist: {cache_override}")
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


def _get_first_sample(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    samples = doc.get("samples")
    if isinstance(samples, list) and samples:
        if isinstance(samples[0], dict):
            return samples[0]
    return None


def refcoco_m_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> List[Any]:
    repo_id = _get_repo_id(lmms_eval_specific_kwargs)
    _ensure_cache_dir(repo_id)
    doc["_refcoco_m_repo"] = repo_id
    image = _to_pil_image(doc["image"], force_rgb=True, repo_id=repo_id)
    return [image]


def refcoco_m_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    repo_id = _get_repo_id(lmms_eval_specific_kwargs)
    doc["_refcoco_m_repo"] = repo_id
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    post_prompt_box = lmms_eval_specific_kwargs.get("post_prompt_box", "")

    mode = os.getenv("REFCOCO_M_MODE", "").strip().lower()
    if not mode:
        mode = os.getenv("MODE", "").strip().lower()
    if not mode:
        mode = lmms_eval_specific_kwargs.get("mode", "box").lower().strip()
    if mode == "box" and post_prompt_box:
        post_prompt = post_prompt_box

    sample = _get_first_sample(doc)
    sentence = ""
    if sample:
        sentences = sample.get("sentences")
        if isinstance(sentences, list) and sentences:
            sentence = str(sentences[-1]).strip()
        elif isinstance(sentences, str):
            sentence = sentences.strip()

    query = f"Locate the {sentence}".strip()
    parts = [pre_prompt, query, post_prompt]
    return "\n".join([part for part in parts if part])


def refcoco_m_doc_to_target(doc: Dict[str, Any]) -> str:
    return ""


def refcoco_m_box_miou(results: List[Any]) -> float:
    vals: List[float] = []
    for result in results:
        if isinstance(result, (int, float)):
            vals.append(float(result))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def refcoco_m_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    prediction = _strip_think_prefix(results[0] if results else "")
    payload = _extract_loc_payload(prediction)
    if not payload:
        return {"refcoco_m_box_miou": 0.0}

    sample = _get_first_sample(doc)
    if not sample:
        return {"refcoco_m_box_miou": 0.0}

    bbox = sample.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return {"refcoco_m_box_miou": 0.0}

    repo_id = str(doc.get("_refcoco_m_repo") or DEFAULT_REPO_ID)
    image = _to_pil_image(doc["image"], force_rgb=False, repo_id=repo_id)
    width, height = image.size

    x, y, w, h = [float(v) for v in bbox[:4]]
    max_val = max(x, y, w, h)
    if max_val <= 1.5:
        x *= width
        w *= width
        y *= height
        h *= height
    gt_box = (x, y, x + w, y + h)

    pred_boxes_raw = _parse_loc_boxes(payload)
    pred_boxes: List[Tuple[float, float, float, float]] = []
    for box in pred_boxes_raw:
        norm_box = _normalize_box_to_pixels(box, width, height)
        if norm_box is not None:
            pred_boxes.append(norm_box)

    if not pred_boxes:
        return {"refcoco_m_box_miou": 0.0}

    best_iou = max(_compute_iou(gt_box, pred_box) for pred_box in pred_boxes)
    return {"refcoco_m_box_miou": float(best_iou)}
