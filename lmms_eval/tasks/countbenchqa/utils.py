from typing import Any, Dict, Optional


def countbenchqa_doc_to_visual(doc: Dict[str, Any]) -> list:
    image = doc["image"]
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    return [image]


def countbenchqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"].strip()

    parts = [pre_prompt, question, post_prompt]
    return "\n".join([p for p in parts if p])


def countbenchqa_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc["number"])
