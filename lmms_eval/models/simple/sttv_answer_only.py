from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.sttv_no_verifier import STTVNoVerifier


@register_model("sttv_answer_only")
class STTVAnswerOnly(STTVNoVerifier):
    """
    STTV answer-only HF model:
    - one pass
    - no grounding / no verifier
    - prompt asks directly for <reason> then <answer>
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        depth: Union[bool, str, int, float] = False,
        instruction_mode: str = "box",
        max_image_side: int = 768,
        no_verifier_max_new_tokens: int = 768,
        torch_dtype: Optional[str] = "auto",
        trust_remote_code: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device_map=device_map,
            batch_size=batch_size,
            depth=depth,
            prompt_path=None,
            instruction_mode=instruction_mode,
            max_image_side=max_image_side,
            no_verifier_max_new_tokens=no_verifier_max_new_tokens,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def _build_prompted_context(self, query: str) -> str:
        query_text = query.strip()
        return (
            f"{query_text}\n\n"
            "Please answer the query by first reasoning inside <reason> tags and then putting ONLY your final answer "
            "inside <answer>. Ensure that the answer is either yes/no, one word, or one number. "
            "Do not round answers, express all ratios as unrounded decimals. "
            "Nothing else."
        )
