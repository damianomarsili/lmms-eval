import json
from io import BytesIO
from typing import Any, Dict, List

from PIL import Image


def subset_dataset(dataset, dataset_source: str):
    """Filter a HF dataset down to a single dataset_source and expand extra metadata."""

    if "dataset_source" not in dataset.column_names:
        raise ValueError("FGVQA dataset must include a 'dataset_source' column.")

    dataset = dataset.filter(lambda example, ds=dataset_source: example.get("dataset_source") == ds)
    dataset = dataset.map(_merge_extra_fields)
    if "extra" in dataset.column_names:
        dataset = dataset.remove_columns("extra")
    return dataset


def _merge_extra_fields(example: Dict[str, Any]) -> Dict[str, Any]:
    extra = example.get("extra")
    if not extra:
        return example
    if isinstance(extra, dict):
        extra_data = extra
    else:
        try:
            extra_data = json.loads(extra)
        except json.JSONDecodeError:
            return example
    for key, value in extra_data.items():
        if key not in example or example[key] in (None, "", []):
            example[key] = value
    return example


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
