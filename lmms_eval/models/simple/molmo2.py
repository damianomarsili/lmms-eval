from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("molmo2")
class Molmo2(lmms):
    def __init__(
        self,
        pretrained: str = "allenai/Molmo2-8B",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_num_frames: int = 32,
        max_image_side: Optional[int] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self.device_map = device_map

        self._model = AutoModelForImageTextToText.from_pretrained(
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
        self.max_num_frames = max_num_frames
        self.max_image_side = max_image_side

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
        raise NotImplementedError("Loglikelihood is not implemented for Molmo2")

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

    def _prepare_image(self, item: object) -> Optional[Image.Image]:
        if isinstance(item, Image.Image):
            image = item.convert("RGB")
        elif isinstance(item, str):
            image = Image.open(item).convert("RGB")
        else:
            return None

        if self.max_image_side:
            image = self._resize_longest_side(image, self.max_image_side)
        return image

    def _load_video(self, video_path: str) -> Optional[Dict[str, object]]:
        try:
            reader = VideoReader(video_path, ctx=cpu(0))
        except Exception as exc:
            eval_logger.warning(f"Failed to load video {video_path}: {exc}")
            return None

        total_frames = len(reader)
        if total_frames == 0:
            eval_logger.warning(f"Video has no frames: {video_path}")
            return None

        max_frames = min(self.max_num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        if total_frames - 1 not in indices:
            indices[-1] = total_frames - 1

        frames = reader.get_batch(indices).asnumpy()
        fps = float(reader.get_avg_fps() or 0.0)
        if fps > 0:
            timestamps = indices / fps
        else:
            timestamps = np.arange(len(indices), dtype=np.float32)

        return {
            "frames": frames,
            "timestamps": timestamps,
            "sampled_fps": fps,
        }

    def _build_messages(self, context: str, visual: Optional[Union[List[object], object]]) -> List[Dict[str, object]]:
        content_parts: List[Dict[str, object]] = []
        visuals: List[object] = []
        if visual is not None:
            visuals = visual if isinstance(visual, (list, tuple)) else [visual]

        for item in visuals:
            if isinstance(item, str) and item.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                video_payload = self._load_video(item)
                if video_payload is not None:
                    content_parts.append({"type": "video", "video": video_payload})
                continue

            image = self._prepare_image(item)
            if image is not None:
                content_parts.append({"type": "image", "image": image})
            else:
                eval_logger.warning(f"Unsupported visual type: {type(item)}")

        content_parts.append({"type": "text", "text": context})
        return [{"role": "user", "content": content_parts}]

    def _generate_once(self, messages: List[Dict[str, object]], gen_kwargs: Dict[str, object]) -> str:
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 512))
        temperature = float(gen_kwargs.get("temperature", 0))
        top_p = float(gen_kwargs.get("top_p", 1.0))
        do_sample = temperature > 0

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.post_process_image_text_to_text(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

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
                if "<image>" in context:
                    context = context.replace("<image>", "")
                messages = self._build_messages(context, visual_list[i])
                answer = self._generate_once(messages, dict(gen_kwargs))
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
