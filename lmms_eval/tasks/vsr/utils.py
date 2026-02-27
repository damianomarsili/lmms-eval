import hashlib
import io
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from PIL import Image


def _get_cache_dir() -> str:
    cache_dir = os.getenv("VSR_IMAGE_CACHE_DIR", "").strip()
    if not cache_dir:
        cache_dir = os.path.join(tempfile.gettempdir(), "lmms_eval_vsr_images")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _to_pil_image(obj: Any, force_rgb: bool = False) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB") if force_rgb else obj
    if isinstance(obj, dict):
        if obj.get("image") is not None:
            return _to_pil_image(obj["image"], force_rgb=force_rgb)
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


def _resolve_image_url(doc: Dict[str, Any]) -> str:
    for key in ("image_link", "image_url", "url"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    image_val = doc.get("image")
    if isinstance(image_val, str) and image_val.startswith(("http://", "https://")):
        return image_val

    image_name = image_val if isinstance(image_val, str) else ""
    image_name = image_name.strip()
    if not image_name:
        raise KeyError("VSR sample is missing both image URL and image filename.")

    coco_split = os.getenv("VSR_COCO_SPLIT", "train2017").strip() or "train2017"
    return f"http://images.cocodataset.org/{coco_split}/{image_name}"


def _cache_file_path(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    _, ext = os.path.splitext(parsed.path)
    ext = ext if ext else ".jpg"
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return os.path.join(_get_cache_dir(), f"{digest}{ext}")


def _download_image(url: str) -> Image.Image:
    candidate_urls = [url]
    if url.startswith("http://"):
        candidate_urls.append("https://" + url[len("http://") :])

    errors: List[str] = []
    for candidate in candidate_urls:
        cache_path = _cache_file_path(candidate)
        if os.path.isfile(cache_path):
            return Image.open(cache_path).convert("RGB")

        request = urllib.request.Request(candidate, headers={"User-Agent": "lmms-eval-vsr"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                image_bytes = response.read()
            with open(cache_path, "wb") as f:
                f.write(image_bytes)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")

    raise FileNotFoundError(f"Failed to fetch VSR image. Attempts: {' | '.join(errors)}")


def _strip_think_prefix(text: str) -> str:
    if not isinstance(text, str):
        return text
    if re.search(r"</(?:plan|think)>", text, flags=re.IGNORECASE):
        return re.split(r"</(?:plan|think)>", text, flags=re.IGNORECASE)[-1].strip()
    return re.sub(r"(?is)^\s*<(?:plan|think)>.*?</(?:plan|think)>\s*", "", text, count=1)


def _parse_binary_prediction(prediction: str) -> Optional[int]:
    text = _strip_think_prefix(prediction).strip()
    if not text:
        return None

    tagged = re.findall(r"(?is)<answer>(.*?)</answer>", text)
    if tagged:
        text = tagged[-1].strip()

    bits = re.findall(r"(?<!\d)([01])(?!\d)", text)
    if bits:
        return int(bits[-1])

    lower = text.lower()
    token_hits: List[tuple[int, int]] = []
    for m in re.finditer(r"\btrue\b", lower):
        token_hits.append((m.start(), 1))
    for m in re.finditer(r"\b(?:false|talse)\b", lower):
        token_hits.append((m.start(), 0))
    for m in re.finditer(r"\byes\b", lower):
        token_hits.append((m.start(), 1))
    for m in re.finditer(r"\bno\b", lower):
        token_hits.append((m.start(), 0))

    if token_hits:
        token_hits.sort(key=lambda x: x[0])
        return token_hits[-1][1]

    return None


def vsr_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> List[Any]:
    del lmms_eval_specific_kwargs

    image_val = doc.get("image")
    if image_val is not None:
        try:
            return [_to_pil_image(image_val, force_rgb=True)]
        except Exception:
            pass

    images_dir = os.getenv("VSR_IMAGES_DIR", "").strip()
    if images_dir and isinstance(image_val, str):
        local_path = os.path.join(images_dir, image_val)
        if os.path.isfile(local_path):
            return [Image.open(local_path).convert("RGB")]

    image_url = _resolve_image_url(doc)
    return [_download_image(image_url)]


def vsr_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    query = str(doc.get("caption", "")).strip()

    parts = [pre_prompt, query, post_prompt]
    return "\n".join([part for part in parts if part])


def vsr_doc_to_target(doc: Dict[str, Any]) -> str:
    label = doc.get("label", 0)
    try:
        label_int = int(label)
    except Exception:
        label_int = 1 if str(label).strip().lower() in {"1", "true", "yes"} else 0
    return "1" if label_int == 1 else "0"


def vsr_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    prediction = results[0] if results else ""
    pred_label = _parse_binary_prediction(prediction)
    gt_label = int(vsr_doc_to_target(doc))
    score = 1.0 if pred_label is not None and pred_label == gt_label else 0.0
    return {"vsr_accuracy": score}
