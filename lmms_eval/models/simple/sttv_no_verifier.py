import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("sttv_no_verifier")
class STTVNoVerifier(lmms):
    """
    STTV No Verifier Model
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        depth: Union[bool, str, int, float] = False,
        prompt_path: Optional[str] = None,
        instruction_mode: str = "point",
        max_image_side: int = 768,
        trust_remote_code: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self.device_map = device_map

        self._model = AutoModelForVision2Seq.from_pretrained(
            pretrained,
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._tokenizer = self.processor.tokenizer
        if self._tokenizer is None:
            raise ValueError("AutoProcessor did not provide a tokenizer.")

        self.pretrained = pretrained
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.max_image_side = max_image_side
        self.depth_enabled = self._coerce_bool(depth)

        self.prompt_template = self._load_prompt_template(prompt_path, self.depth_enabled)
        self.instruction_text = self._load_instruction_text(instruction_mode)
        self.depth_instruction_text = self._load_depth_instruction_text(instruction_mode) if self.depth_enabled else None

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        self._device = self.model.device

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for STTVNoVerifier")

    def _coerce_bool(self, value: Union[bool, str, int, float]) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        normalized = value.strip().lower()
        return normalized not in {"0", "false", "no", "off", ""}

    def _load_prompt_template(self, prompt_path: Optional[str], depth_enabled: bool) -> str:
        if prompt_path is None:
            filename = "sttv_depth.txt" if depth_enabled else "sttv_no_verifier.txt"
            prompt_file = Path(__file__).resolve().parents[3] / "prompts" / filename
        else:
            prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        prompt_text = prompt_file.read_text(encoding="utf-8")
        prompt_text = prompt_text.replace("<plan>", "<reason>").replace("</plan>", "</reason>")

        prompt_template = prompt_text.strip()
        if not prompt_template:
            raise ValueError(f"Prompt file is empty: {prompt_file}")
        return prompt_template

    def _load_instruction_text(self, instruction_mode: str) -> str:
        mode = instruction_mode.lower()
        if mode not in {"point", "box"}:
            raise ValueError(f"instruction_mode must be 'point' or 'box', got {instruction_mode}")
        filename = "instructions_pt.txt" if mode == "point" else "instructions_box.txt"
        instruction_file = Path(__file__).resolve().parents[3] / "prompts" / filename
        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        instruction_text = instruction_file.read_text(encoding="utf-8").strip()
        if not instruction_text:
            raise ValueError(f"Instruction file is empty: {instruction_file}")
        return instruction_text

    def _load_depth_instruction_text(self, instruction_mode: str) -> str:
        mode = instruction_mode.lower()
        if mode not in {"point", "box"}:
            raise ValueError(f"instruction_mode must be 'point' or 'box', got {instruction_mode}")
        filename = "instructions_depth_pt.txt" if mode == "point" else "instructions_depth_box.txt"
        instruction_file = Path(__file__).resolve().parents[3] / "prompts" / filename
        if not instruction_file.exists():
            raise FileNotFoundError(f"Depth instruction file not found: {instruction_file}")
        instruction_text = instruction_file.read_text(encoding="utf-8").strip()
        if not instruction_text:
            raise ValueError(f"Depth instruction file is empty: {instruction_file}")
        return instruction_text

    def _build_prompted_context(self, context: str) -> str:
        if self.depth_enabled:
            if self.depth_instruction_text is None:
                raise ValueError("Depth mode enabled but depth instructions are missing.")
            return self.prompt_template.format(self.instruction_text, self.depth_instruction_text, context.strip())
        return self.prompt_template.format(self.instruction_text, context.strip())

    def _resize_longest_side(self, image: Image.Image, longest_side: int) -> Image.Image:
        width, height = image.size
        if max(width, height) <= longest_side:
            return image
        if width >= height:
            new_width = longest_side
            new_height = int(round(height * (longest_side / width)))
        else:
            new_height = longest_side
            new_width = int(round(width * (longest_side / height)))
        return image.resize((new_width, new_height), Image.BICUBIC)

    def _build_messages(self, context: str, visual: Optional[Union[List[object], object]]) -> List[Dict[str, object]]:
        content_parts: List[Dict[str, object]] = []
        if visual is not None:
            visual_items = visual if isinstance(visual, (list, tuple)) else [visual]
            for item in visual_items:
                if isinstance(item, Image.Image):
                    resized = self._resize_longest_side(item.convert("RGB"), self.max_image_side)
                    content_parts.append({"type": "image", "image": resized})
                else:
                    eval_logger.warning(f"Unsupported visual type: {type(item)}")
        content_parts.append({"type": "text", "text": context})
        return [{"role": "user", "content": content_parts}]

    def _generate_once(
        self,
        messages: List[Dict[str, object]],
        gen_kwargs: Dict[str, object],
    ) -> str:
        del gen_kwargs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        cont = self.model.generate(**inputs, max_new_tokens=512)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return answers[0]

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

                answer = self._generate_once(messages, dict(gen_kwargs))
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
