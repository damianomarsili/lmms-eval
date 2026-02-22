import re
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.sttv_vllm import STTVVLLM


@register_model("sttv_no_verifier_vllm")
class STTVNoVerifierVLLM(STTVVLLM):
    """
    STTV no-verifier vLLM model:
    - one pass
    - no verifier calls
    - prompt enforces exactly one <bbox_2d>, one <reason>, one <answer>
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: Optional[Union[int, str]] = 1,
        depth: Union[bool, str, int, float] = False,
        prompt_path: Optional[str] = None,
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
        if self._coerce_bool(depth):
            raise ValueError("sttv_no_verifier_vllm does not support depth prompts.")
        if instruction_mode.lower() != "box":
            raise ValueError(f"Only box mode is supported; got instruction_mode={instruction_mode}")

        if prompt_path is None:
            prompt_path = str(Path(__file__).resolve().parents[3] / "prompts" / "sttv_no_verifier_single_turn.txt")

        super().__init__(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            batch_size=batch_size,
            depth=False,
            prompt_path=prompt_path,
            instruction_mode=instruction_mode,
            max_image_side=max_image_side,
            verifier_max_attempts=1,
            verifier_max_new_tokens=32,
            verifier_image_side=max_image_side,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            min_image_pixels=min_image_pixels,
            seed=seed,
            disable_log_stats=disable_log_stats,
            **kwargs,
        )
        self.no_verifier_max_new_tokens = int(no_verifier_max_new_tokens)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(x[0]), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        self.last_generation_metadata = None
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i, context in enumerate(contexts):
                if "<image" in context:
                    context = re.sub(r"<image\s*\d+>", "", context)
                    context = context.replace("<image>", "")
                prompted_context = self._build_prompted_context(context)
                messages = self._build_messages(prompted_context, visual_list[i])

                answer, _ = self._generate_once(
                    messages,
                    dict(gen_kwargs),
                    stop_sequences=["</answer>"],
                    max_new_tokens=self.no_verifier_max_new_tokens,
                )
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)
                self._cleanup_after_sample()

        res = re_ords.get_original(res)
        pbar.close()
        return res
