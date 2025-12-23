import json
from io import BytesIO
from typing import Any, Dict, List

from PIL import Image


def subset_dataset(dataset, dataset_source: str):
    """Filter a HF dataset down to a single dataset_source."""

    if "dataset_source" not in dataset.column_names:
        raise ValueError("FGVQA dataset must include a 'dataset_source' column.")

    return dataset.filter(lambda example, ds=dataset_source: example.get("dataset_source") == ds)


def parse_extra(doc: Dict) -> Dict[str, Any]:
    extra = doc.get("extra")
    if isinstance(extra, dict):
        return extra
    if isinstance(extra, str) and extra:
        try:
            return json.loads(extra)
        except json.JSONDecodeError:
            return {}
    return {}


def load_images_from_doc(doc: Dict) -> List[Image.Image]:
    visuals: List[Image.Image] = []
    for entry in doc.get("images", []):
        visuals.append(_entry_to_image(entry))
    return visuals


def _entry_to_image(entry: Any) -> Image.Image:
    if isinstance(entry, Image.Image):
        image = entry
    elif isinstance(entry, str):
        image = Image.open(entry)
    elif isinstance(entry, dict):
        data = entry.get("bytes")
        path = entry.get("path")
        if data is not None:
            image = Image.open(BytesIO(data))
        elif path:
            image = Image.open(path)
        else:
            raise ValueError(f"Invalid image entry: {entry}")
    else:
        raise ValueError(f"Unsupported image entry type: {type(entry)}")
    return image.convert("RGB")
