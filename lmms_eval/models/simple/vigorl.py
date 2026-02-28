from typing import Optional, Union

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL


VIGORL_SYSTEM_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant systematically reasons through the problem step by step by checking and verifying possible solutions and image regions, while grounding reasoning steps to specific objects and their relationships in the image using (x,y) coordinates. There may be one image or two images concatenated together, in which case the Assistant must compare the spatial relationships between the two images.

All reasoning processes must be enclosed within a single set of '<think>' tags, and reasoning steps must include specific reference coordinates:

For example, <think>
{Reasoning text}. {Further reasoning text} {more reasoning}
</think>

The final answer should be enclosed in '<answer>' tags in the format:
<answer> {text of selected answer choice} </answer>

The Assistant must help the user identify the correct answer choice from the options provided.
-If the correct answer is unclear, select the most relevant option based on the spatial relationships and dynamics within the image.
- The Assistant should verify each step and check multiple possible solutions before selecting the final answer."""


@register_model("vigorl")
class ViGoRL(Qwen2_5_VL):
    """ViGoRL baseline model wrapper.

    HF: https://huggingface.co/gsarch/ViGoRL-3b-Spatial
    """

    def __init__(
        self,
        pretrained: str = "gsarch/ViGoRL-3b-Spatial",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        dtype: str = "bfloat16",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 3136,
        max_pixels: int = 12845056,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = VIGORL_SYSTEM_PROMPT,
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
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
