import logging
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=1024,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    eos_token_id=151645,
    pad_token_id=151645
)


def build_transform(input_size: int):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12, use_thumbnail=True):
    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        pil_img = Image.open(image).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(pil_img, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


@register_model("internvl3_5")
class InternVL3_5(lmms):
    """
    InternVL3.5 models released by OpenGVLab.
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-1B",
        modality: str = "image",
        device: Union[str, torch.device] = "cuda",
        device_map: Optional[Union[str, dict]] = None,
        batch_size: Union[int, str] = 1,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        use_flash_attn: bool = True,
        load_in_8bit: bool = False,
        image_size: int = 448,
        max_tiles: int = 12,
        use_thumbnail: bool = True,
        **kwargs,
    ):
        super().__init__()

        assert modality in ["image"], f"Unsupported modality {modality}."
        self.modality = modality
        self.image_size = image_size
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail
        self.batch_size_per_gpu = int(batch_size)
        if dtype == "float32":
            self._dtype = torch.float32
        else:
            self._dtype = torch.bfloat16

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = torch.device(device) if isinstance(device, str) else device

        model_kwargs = dict(
            trust_remote_code=trust_remote_code,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
        )
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if use_flash_attn is not None:
            model_kwargs["use_flash_attn"] = use_flash_attn
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self._model = AutoModel.from_pretrained(pretrained, **model_kwargs).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code, use_fast=False)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._config = self._model.config
        self.model.tie_weights()
        self.path = pretrained

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected DistributedType.DEEPSPEED. Please make sure zero-stage is set to 0.")
            if accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            if device_map is None:
                self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

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
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def flatten(self, input):
        new_list = []
        for item in input:
            if isinstance(item, (list, tuple)):
                new_list.extend(self.flatten(item))
            elif item is not None:
                new_list.append(item)
        return new_list

    def _prepare_pixel_values(self, visuals: List) -> Tuple[Optional[torch.Tensor], Optional[List[int]], str]:
        if not visuals:
            return None, None, ""
        processed = []
        num_patches = []
        for visual in visuals:
            pixels = load_image(visual, input_size=self.image_size, max_num=self.max_tiles, use_thumbnail=self.use_thumbnail)
            processed.append(pixels)
            num_patches.append(pixels.size(0))
        pixel_values = torch.cat(processed, dim=0).to(device=self.device, dtype=self._dtype)
        image_tokens = " ".join(["<image>"] * len(processed))
        return pixel_values, num_patches, image_tokens

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []
        re_ords = utils.Collator([reg.args for reg in requests], lambda x: (-len(self.tok_encode(x[0])), x[0]), grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            gen_kwargs = dict(all_gen_kwargs[0])
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                gen_kwargs.setdefault(k, v)
            gen_kwargs.pop("image_aspect_ratio", None)

            assert self.batch_size_per_gpu == 1, "InternVL3.5 interface currently supports batch_size_per_gpu=1."
            context = contexts[0]
            pixel_values, num_patches_list, image_tokens = self._prepare_pixel_values(visuals)
            if image_tokens:
                context = image_tokens + "\n" + context
            response, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                context,
                gen_kwargs,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
            responses.append(response)
            pbar.update(1)
        pbar.close()
        responses = re_ords.get_original(responses)
        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for InternVL3.5 models.")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for InternVL3.5")
