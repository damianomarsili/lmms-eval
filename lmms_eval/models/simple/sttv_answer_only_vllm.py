from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.sttv_no_verifier_vllm import STTVNoVerifierVLLM


@register_model("sttv_answer_only_vllm")
class STTVAnswerOnlyVLLM(STTVNoVerifierVLLM):
    """
    STTV answer-only vLLM model:
    - one pass
    - no grounding / no verifier
    - prompt asks directly for <reason> then <answer>
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: Optional[Union[int, str]] = 1,
        depth: Union[bool, str, int, float] = False,
        instruction_mode: str = "box",
        max_image_side: int = 768,
        no_verifier_max_new_tokens: int = 768,
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        min_image_pixels: int = 28,
        seed: int = 1,
        disable_log_stats: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            batch_size=batch_size,
            depth=depth,
            prompt_path=None,
            instruction_mode=instruction_mode,
            max_image_side=max_image_side,
            no_verifier_max_new_tokens=no_verifier_max_new_tokens,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            min_image_pixels=min_image_pixels,
            seed=seed,
            disable_log_stats=disable_log_stats,
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
