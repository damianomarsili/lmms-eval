import re
from typing import Optional, Union

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL


GRIT_PROMPT_SUFFIX = (
    " First, think between <think> and </think> while output necessary coordinates "
    "needed to answer the question in JSON with key 'bbox_2d'. Then, based on the "
    "thinking contents and coordinates, rethink between <rethink> </rethink> and "
    "then answer the question after <answer>."
)


@register_model("grit")
class GRIT(Qwen2_5_VL):
    """GRIT baseline model wrapper.

    Inference reference:
    https://github.com/eric-ai-lab/GRIT?tab=readme-ov-file#inference
    HF checkpoint:
    https://huggingface.co/yfan1997/GRIT-20-Qwen2.5-VL-3B
    """

    def __init__(
        self,
        pretrained: str = "yfan1997/GRIT-20-Qwen2.5-VL-3B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        dtype: str = "bfloat16",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = GRIT_PROMPT_SUFFIX,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            dtype=dtype,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            use_custom_video_loader=use_custom_video_loader,
            fps=fps,
            max_image_size=max_image_size,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
            **kwargs,
        )

    @staticmethod
    def _ensure_answer_tag_closed(text: str) -> str:
        if not isinstance(text, str):
            return text

        open_count = len(re.findall(r"(?is)<answer>", text))
        close_count = len(re.findall(r"(?is)</answer>", text))
        if open_count > close_count:
            return f"{text.rstrip()} </answer>"
        return text

    def generate_until(self, requests: list[Instance]) -> list[str]:
        original_arguments = [req.arguments for req in requests]
        try:
            for req in requests:
                args = list(req.arguments if isinstance(req.arguments, tuple) else (req.arguments,))
                if not args:
                    continue

                context = args[0]
                if isinstance(context, str):
                    stripped = context.strip()
                    if not stripped.lower().startswith("question:"):
                        args[0] = f"Question: {stripped}"
                    else:
                        args[0] = stripped

                if len(args) > 1 and isinstance(args[1], dict):
                    gen_kwargs = dict(args[1])
                    gen_kwargs.setdefault("temperature", 0.001)
                    gen_kwargs.setdefault("max_new_tokens", 512)
                    gen_kwargs.setdefault("num_beams", 1)
                    args[1] = gen_kwargs

                req.arguments = tuple(args)

            outputs = super().generate_until(requests)
            return [self._ensure_answer_tag_closed(output) for output in outputs]
        finally:
            for req, original in zip(requests, original_arguments):
                req.arguments = original
