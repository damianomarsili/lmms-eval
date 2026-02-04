import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

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
    prompt = prompt.replace("Segment the", "Locate the").replace("segment the", "Locate the")

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


def _mask_contains_point(mask_img: Image.Image, point: Tuple[float, float]) -> bool:
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")
    width, height = mask_img.size
    pixel = _normalize_to_pixel(point, width, height)
    if pixel is None:
        return False
    return mask_img.getpixel(pixel) > 0


def convseg_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    prediction = _strip_think_prefix(results[0] if results else "")
    payload = _extract_loc_payload(prediction)
    if not payload:
        return {"convseg_point_acc": 0.0}

    mode = str(doc.get("_convseg_mode", "point")).lower().strip()
    points = _parse_loc_points(payload, mode)
    if not points:
        return {"convseg_point_acc": 0.0}

    repo_id = str(doc.get("_convseg_repo") or DEFAULT_REPO_ID)
    mask = _to_pil_image(doc["mask"], force_rgb=False, repo_id=repo_id)
    point = points[0]
    return {"convseg_point_acc": 1.0 if _mask_contains_point(mask, point) else 0.0}
