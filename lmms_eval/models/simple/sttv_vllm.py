import base64
import gc
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

BOX_COLOR = (0, 0, 255)
FONT_SCALE = 0.022
BOX_OUTLINE_SCALE = 0.005
LABEL_PADDING = 2
BBOX_2D_ENTRY_PATTERN = re.compile(
    r'^\s*(?P<idx>\d+)\s*:\s*label\s*=\s*"(?P<label>[^"\n]+?)"\s*,\s*\[(?P<coords>[^\]]+)\]\s*$',
    flags=re.IGNORECASE,
)
VERIFIER_EDIT_PATTERN = re.compile(r"(?i)^EDIT\s+(?P<idx>\d+)\s*:\s*(?P<body>.+)$")
VERIFIER_DISALLOWED_REMOVE_PATTERN = re.compile(r"(?i)^REMOVE\s+\d+\s*$")
VERIFIER_ADD_PATTERN = re.compile(
    r'(?i)^ADD\s+label\s*=\s*"(?P<label>[^"\n]+?)"\s*,\s*\['
    r"\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)\s*\]\s*$"
)
VERIFIER_COORD_KEYWORD_PATTERN = re.compile(r"(?i)\b(x_min|x_max|y_min|y_max|left|right|top|bottom|width|height)\b")
VERIFIER_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class LocEntry:
    image_index: int
    label: str
    coords: Tuple[float, ...]


@register_model("sttv_vllm")
class STTVVLLM(lmms):
    """
    STTV Model with verifier.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: Optional[Union[int, str]] = 1,
        depth: Union[bool, str, int, float] = False,
        prompt_path: Optional[str] = None,
        verifier_prompt_path: Optional[str] = None,
        instruction_mode: str = "box",
        max_image_side: int = 768,
        loc_verifier_rounds: int = 3,
        verifier_max_new_tokens: int = 96,
        verifier_image_side: int = 1024,
        generation_max_new_tokens: int = 2048,
        generation_chunk_max_new_tokens: int = 256,
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        min_image_pixels: int = 28,
        seed: int = 1,
        disable_log_stats: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if LLM is None or SamplingParams is None:
            raise ImportError("vllm is not installed. Please install vllm to use sttv_vllm.")

        # Convert JSON-like string kwargs into dicts (vllm compatibility).
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like string for argument '{key}': {value}")

        self.model_name = model
        self.pretrained = model
        self.chat_template = self._load_chat_template(chat_template)
        self.min_image_pixels = int(min_image_pixels)
        self._enforce_image_resize = self._is_qwen_vl_model(model)

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        if data_parallel_size > 1:
            assert tensor_parallel_size == 1, "Data parallelism is not supported with tensor parallelism for vllm."
        if accelerator.num_processes > 1:
            kwargs["distributed_executor_backend"] = "external_launcher"

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.client = LLM(
            model=self.model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            disable_log_stats=disable_log_stats,
            seed=seed,
            **kwargs,
        )
        self.disable_log_stats = disable_log_stats
        self.seed = seed

        self._config = None
        self._tokenizer = None
        self._device = self.accelerator.device
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.max_image_side = max_image_side
        self.depth_enabled = self._coerce_bool(depth)
        self.instruction_mode = instruction_mode.lower()
        if self.instruction_mode != "box":
            raise ValueError(f"Only box mode is supported; got instruction_mode={instruction_mode}")

        self.loc_verifier_rounds = max(0, int(loc_verifier_rounds))
        self.verifier_max_new_tokens = int(verifier_max_new_tokens)
        self.verifier_image_side = int(verifier_image_side)
        self.generation_max_new_tokens = max(64, int(generation_max_new_tokens))
        self.generation_chunk_max_new_tokens = max(1, int(generation_chunk_max_new_tokens))

        self.prompt_template = self._load_prompt_template(prompt_path, self.depth_enabled)
        self.instruction_text = self._load_instruction_text(self.instruction_mode)
        self.depth_instruction_text = self._load_depth_instruction_text(self.instruction_mode) if self.depth_enabled else None
        self.verifier_template = self._load_verifier_template(verifier_prompt_path)

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.client

    @property
    def eot_token_id(self):
        return None

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
        raise NotImplementedError("Loglikelihood is not implemented for STTV")

    def _coerce_bool(self, value: Union[bool, str, int, float]) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        normalized = value.strip().lower()
        return normalized not in {"0", "false", "no", "off", ""}

    def _load_prompt_template(self, prompt_path: Optional[str], depth_enabled: bool) -> str:
        if prompt_path is None:
            if depth_enabled:
                prompt_file = Path(__file__).resolve().parents[3] / "prompts" / "sttv_verifier_depth.txt"
            else:
                prompt_file = Path(__file__).resolve().parents[3] / "prompts" / "sttv_verifier_single_turn.txt"
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
        if mode != "box":
            raise ValueError(f"Only box mode is supported; got instruction_mode={instruction_mode}")
        filename = "instructions_box.txt"
        instruction_file = Path(__file__).resolve().parents[3] / "prompts" / filename
        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        instruction_text = instruction_file.read_text(encoding="utf-8").strip()
        if not instruction_text:
            raise ValueError(f"Instruction file is empty: {instruction_file}")
        return instruction_text

    def _load_depth_instruction_text(self, instruction_mode: str) -> str:
        mode = instruction_mode.lower()
        if mode != "box":
            raise ValueError(f"Only box mode is supported; got instruction_mode={instruction_mode}")
        filename = "instructions_depth_box.txt"
        instruction_file = Path(__file__).resolve().parents[3] / "prompts" / filename
        if not instruction_file.exists():
            raise FileNotFoundError(f"Depth instruction file not found: {instruction_file}")
        instruction_text = instruction_file.read_text(encoding="utf-8").strip()
        if not instruction_text:
            raise ValueError(f"Depth instruction file is empty: {instruction_file}")
        return instruction_text

    def _load_verifier_template(self, prompt_path: Optional[str]) -> str:
        if prompt_path is None:
            prompt_file = Path(__file__).resolve().parents[3] / "prompts" / "verifier_instructions.txt"
        else:
            prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Verifier prompt file not found: {prompt_file}")
        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Verifier prompt file is empty: {prompt_file}")
        return prompt_text

    def _build_prompted_context(self, context: str) -> str:
        if self.depth_enabled:
            if self.depth_instruction_text is None:
                raise ValueError("Depth mode enabled but depth instructions are missing.")
            return self.prompt_template.format(self.instruction_text, self.depth_instruction_text, context.strip())
        return self.prompt_template.format(self.instruction_text, context.strip())

    def _cleanup_after_sample(self) -> None:
        gc.collect()

    def _load_chat_template(self, chat_template: Optional[str]) -> Optional[str]:
        if chat_template is None:
            return None
        if os.path.sep in chat_template or chat_template.endswith((".jinja", ".jinja2", ".j2")):
            if not os.path.isfile(chat_template):
                raise FileNotFoundError(f"Chat template file not found: {chat_template}")
            with open(chat_template, "r", encoding="utf-8") as handle:
                return handle.read()
        return chat_template

    def _is_qwen_vl_model(self, model: str) -> bool:
        qwen_vl_patterns = ["qwen2-vl", "qwen2.5-vl", "qwen3-vl"]
        return any(pattern in model.lower() for pattern in qwen_vl_patterns)

    def _maybe_resize_image(self, img: Image.Image) -> Image.Image:
        if self.min_image_pixels <= 0:
            return img
        if min(img.size) <= 0:
            raise ValueError(f"Invalid image dimensions: {img.size}")
        if not self._enforce_image_resize or min(img.size) >= self.min_image_pixels:
            return img
        scale = self.min_image_pixels / min(img.size)
        new_size = tuple(int(dim * scale) for dim in img.size)
        return img.resize(new_size, Image.BICUBIC)

    def _encode_image(self, image: Image.Image) -> str:
        img = self._maybe_resize_image(image)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

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
                    encoded = self._encode_image(resized)
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
                else:
                    eval_logger.warning(f"Unsupported visual type: {type(item)}")
        content_parts.append({"type": "text", "text": context})
        return [{"role": "user", "content": content_parts}]

    def _generate_once(
        self,
        messages: List[Dict[str, object]],
        gen_kwargs: Dict[str, object],
        stop_sequences: Optional[List[str]] = None,
        max_new_tokens: int = 256,
    ) -> Tuple[str, int]:
        del gen_kwargs
        params: Dict[str, object] = {"max_tokens": max_new_tokens, "temperature": 0, "top_p": 1.0, "seed": self.seed}
        if stop_sequences:
            params["stop"] = stop_sequences
        sampling_params = SamplingParams(**params)

        if self.chat_template is not None:
            response = self.client.chat(sampling_params=sampling_params, messages=messages, chat_template=self.chat_template)
        else:
            response = self.client.chat(sampling_params=sampling_params, messages=messages)

        output = response[0].outputs[0]
        decoded = output.text
        token_count = len(getattr(output, "token_ids", []) or [])
        if stop_sequences:
            stop_index = None
            stop_token = None
            for sequence in stop_sequences:
                idx = decoded.find(sequence)
                if idx != -1 and (stop_index is None or idx < stop_index):
                    stop_index = idx
                    stop_token = sequence
            if stop_index is not None and stop_token is not None:
                decoded = decoded[: stop_index + len(stop_token)]
            else:
                stop_reason = getattr(output, "stop_reason", None)
                if isinstance(stop_reason, str) and stop_reason in stop_sequences:
                    decoded = f"{decoded}{stop_reason}"
        return decoded, token_count

    def _extract_bbox_2d_payloads(self, text: str) -> List[str]:
        if not text:
            return []
        payloads: List[str] = []
        for raw_payload in re.findall(r"(?is)<bbox_2d>(.*?)</bbox_2d>", text):
            payloads.append(str(raw_payload).strip())
        return payloads

    def _parse_bbox_2d_entries(self, payload: str) -> List[LocEntry]:
        entries: List[LocEntry] = []
        nonempty_line_count = 0
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            nonempty_line_count += 1
            match = BBOX_2D_ENTRY_PATTERN.fullmatch(line)
            if match is None:
                return []
            try:
                idx = int(match.group("idx"))
            except (TypeError, ValueError):
                return []
            if idx != len(entries) + 1:
                return []

            label = match.group("label").strip()
            if not label:
                return []

            numbers = re.findall(r"-?\d+(?:\.\d+)?", match.group("coords"))
            if len(numbers) != 4:
                return []
            coords_tuple = tuple(float(n) for n in numbers[:4])
            x1, y1, x2, y2 = coords_tuple
            if not (0.0 <= x1 <= 1000.0 and 0.0 <= y1 <= 1000.0 and 0.0 <= x2 <= 1000.0 and 0.0 <= y2 <= 1000.0):
                return []
            if x2 <= x1 or y2 <= y1:
                return []
            entries.append(LocEntry(image_index=1, label=label, coords=coords_tuple))
        if nonempty_line_count == 0:
            return []
        return entries

    def _has_missing_label(self, payload: str) -> bool:
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            has_coords = bool(re.search(r"\[[^\]]*\]", line))
            has_label = bool(re.search(r'(?i)\blabel\s*=\s*"', line))
            if has_coords and not has_label:
                return True
        return False

    def _has_invalid_box(self, entries: List[LocEntry]) -> bool:
        for entry in entries:
            if len(entry.coords) < 4:
                continue
            x1, y1, x2, y2 = entry.coords[:4]
            if x2 <= x1 or y2 <= y1:
                return True
            if not (0.0 <= x1 <= 1000.0 and 0.0 <= y1 <= 1000.0 and 0.0 <= x2 <= 1000.0 and 0.0 <= y2 <= 1000.0):
                return True
        return False

    def _load_font(self, image: Image.Image, scale: float = FONT_SCALE) -> ImageFont.ImageFont:
        width, height = image.size
        size = max(12, int(min(width, height) * scale))
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def _scale_point(self, x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        x_px = int(round(x / 1000.0 * width))
        y_px = int(round(y / 1000.0 * height))
        return max(0, min(width - 1, x_px)), max(0, min(height - 1, y_px))

    def _measure_text(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        if hasattr(font, "getbbox"):
            left, top, right, bottom = font.getbbox(text)
            return right - left, bottom - top
        return font.getsize(text)

    def _overlay_boxes(self, image: Image.Image, entries: List[LocEntry]) -> Image.Image:
        base = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = self._load_font(base)

        width, height = base.size
        outline_width = max(2, int(min(width, height) * BOX_OUTLINE_SCALE))
        indexed_entries: List[Tuple[int, LocEntry, float]] = []
        for idx, entry in enumerate(entries, start=1):
            x1, y1, x2, y2 = entry.coords[:4]
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            indexed_entries.append((idx, entry, area))
        # Draw larger boxes first so smaller nested boxes remain visible on top.
        indexed_entries.sort(key=lambda item: item[2], reverse=True)

        for idx, entry, _ in indexed_entries:
            x1, y1, x2, y2 = entry.coords[:4]
            left, top = self._scale_point(x1, y1, width, height)
            right, bottom = self._scale_point(x2, y2, width, height)
            if right < left:
                left, right = right, left
            if bottom < top:
                top, bottom = bottom, top
            draw.rectangle(
                (left, top, right, bottom),
                outline=BOX_COLOR,
                width=outline_width,
            )
            label = f"{idx}:{entry.label}" if entry.label else str(idx)
            text_w, text_h = self._measure_text(draw, label, font)
            text_x = min(width - text_w - LABEL_PADDING, max(0, left + LABEL_PADDING))
            text_y = min(height - text_h - LABEL_PADDING, max(0, top + LABEL_PADDING))
            draw.rectangle(
                (text_x - LABEL_PADDING, text_y - LABEL_PADDING, text_x + text_w + LABEL_PADDING, text_y + text_h),
                fill=(255, 255, 255, 200),
            )
            draw.text((text_x, text_y), label, fill=BOX_COLOR, font=font)
        return Image.alpha_composite(base, overlay).convert("RGB")

    def _build_verifier_prompt(self, entries: List[LocEntry], image_count: int) -> str:
        targets = sorted({entry.label for entry in entries})
        targets_str = ", ".join(targets) if targets else "(none)"

        lines: List[str] = []
        for i, entry in enumerate(entries, 1):
            prefix = f"image_{entry.image_index}, " if image_count > 1 else ""
            x1, y1, x2, y2 = entry.coords[:4]
            lines.append(f'{i}) {prefix}label="{entry.label}", [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]')

        preds_str = "\n".join(lines) if lines else "(none)"
        return self.verifier_template.format(targets=targets_str, preds=preds_str)

    def _run_verifier(self, originals: List[Image.Image], overlays: List[Image.Image], prompt: str) -> str:
        resized_originals = [self._resize_longest_side(image, self.verifier_image_side) for image in originals]
        resized_overlays = [self._resize_longest_side(image, self.verifier_image_side) for image in overlays]

        content: List[Dict[str, object]] = []
        for original, overlay in zip(resized_originals, resized_overlays):
            encoded_original = self._encode_image(original)
            encoded_overlay = self._encode_image(overlay)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_original}"}})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_overlay}"}})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        sampling_params = SamplingParams(
            max_tokens=self.verifier_max_new_tokens,
            temperature=0,
            top_p=1.0,
            seed=self.seed,
        )
        if self.chat_template is not None:
            response = self.client.chat(sampling_params=sampling_params, messages=messages, chat_template=self.chat_template)
        else:
            response = self.client.chat(sampling_params=sampling_params, messages=messages)
        return response[0].outputs[0].text

    def _parse_verifier_feedback(self, text: str, entries: List[LocEntry]) -> Tuple[str, str, Dict[str, object]]:
        cleaned = text.replace("<|im_end|>", "").strip()
        valid_indices = set(range(1, len(entries) + 1))
        index_actions: Dict[int, Tuple[str, int]] = {}
        raw_add_actions: List[Tuple[str, int, Tuple[str, float, float, float, float]]] = []
        line_order = 0

        def _format_coord(value: float) -> str:
            if float(value).is_integer():
                return str(int(value))
            return f"{float(value):.3f}".rstrip("0").rstrip(".")

        def _normalize_label(label: str) -> str:
            return " ".join(str(label).strip().lower().split())

        def _canonical_signature(
            *,
            label: str,
            x1: float,
            y1: float,
            x2: float,
            y2: float,
        ) -> Tuple[str, float, float, float, float]:
            return (
                _normalize_label(label),
                round(float(x1), 3),
                round(float(y1), 3),
                round(float(x2), 3),
                round(float(y2), 3),
            )

        existing_signatures: set[Tuple[str, float, float, float, float]] = set()
        for entry in entries:
            x1, y1, x2, y2 = entry.coords[:4]
            signature = _canonical_signature(label=entry.label, x1=x1, y1=y1, x2=x2, y2=y2)
            existing_signatures.add(signature)

        duplicate_add_existing_count = 0
        disallowed_remove_count = 0

        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if not line:
                continue
            if line.lower().startswith("corrections:"):
                line = line.split(":", 1)[1].strip()
                if not line:
                    continue

            edit_match = VERIFIER_EDIT_PATTERN.match(line)
            if edit_match is not None:
                idx = int(edit_match.group("idx"))
                body = edit_match.group("body").strip()
                has_coord_keyword = VERIFIER_COORD_KEYWORD_PATTERN.search(body) is not None
                has_numeric_value = VERIFIER_NUMBER_PATTERN.search(body) is not None
                if idx in valid_indices and body and has_coord_keyword and has_numeric_value:
                    index_actions[idx] = (f"EDIT {idx}: {body}", line_order)
                    line_order += 1
                continue

            if VERIFIER_DISALLOWED_REMOVE_PATTERN.match(line) is not None:
                disallowed_remove_count += 1
                continue

            add_match = VERIFIER_ADD_PATTERN.match(line)
            if add_match is not None:
                label = add_match.group("label").strip()
                if not label:
                    continue
                x1 = float(add_match.group("x1"))
                y1 = float(add_match.group("y1"))
                x2 = float(add_match.group("x2"))
                y2 = float(add_match.group("y2"))
                if not (
                    0.0 <= x1 <= 1000.0
                    and 0.0 <= y1 <= 1000.0
                    and 0.0 <= x2 <= 1000.0
                    and 0.0 <= y2 <= 1000.0
                    and x2 > x1
                    and y2 > y1
                ):
                    continue
                normalized = (
                    f'ADD label="{label}", '
                    f"[{_format_coord(x1)}, {_format_coord(y1)}, {_format_coord(x2)}, {_format_coord(y2)}]"
                )
                signature = _canonical_signature(label=label, x1=x1, y1=y1, x2=x2, y2=y2)
                raw_add_actions.append((normalized, line_order, signature))
                line_order += 1

        add_actions: List[Tuple[str, int]] = []
        add_seen: set[Tuple[str, float, float, float, float]] = set()
        for line, order, signature in raw_add_actions:
            if signature in existing_signatures:
                duplicate_add_existing_count += 1
                continue
            if signature in add_seen:
                continue
            add_seen.add(signature)
            add_actions.append((line, order))

        normalized_lines: List[Tuple[str, int]] = list(index_actions.values()) + add_actions
        normalized_lines.sort(key=lambda item: item[1])
        has_effect = len(normalized_lines) > 0
        feedback_valid_for_reward = has_effect and duplicate_add_existing_count == 0 and disallowed_remove_count == 0
        feedback_info: Dict[str, object] = {
            "feedback_has_effect": bool(has_effect),
            "feedback_valid_for_reward": bool(feedback_valid_for_reward),
            "feedback_has_duplicate_add_existing": bool(duplicate_add_existing_count > 0),
            "feedback_has_disallowed_remove": bool(disallowed_remove_count > 0),
            "feedback_duplicate_add_existing_count": int(duplicate_add_existing_count),
            "feedback_disallowed_remove_count": int(disallowed_remove_count),
        }
        if len(normalized_lines) == 0:
            no_op = "NO_VALID_CORRECTIONS. Re-emit all boxes unchanged in one <bbox_2d> block."
            return no_op, cleaned, feedback_info
        corrections = "\n".join(line for line, _ in normalized_lines)
        return corrections, cleaned, feedback_info

    def _format_bbox_block(self, entries: List[LocEntry]) -> str:
        def _format_coord(value: float) -> str:
            if float(value).is_integer():
                return str(int(value))
            return f"{float(value):.3f}".rstrip("0").rstrip(".")

        lines: List[str] = []
        for idx, entry in enumerate(entries, start=1):
            x1, y1, x2, y2 = entry.coords[:4]
            lines.append(
                f'{idx}: label="{entry.label}", '
                f'[{_format_coord(x1)}, {_format_coord(y1)}, {_format_coord(x2)}, {_format_coord(y2)}]'
            )
        return "<bbox_2d>\n" + "\n".join(lines) + "\n</bbox_2d>"

    def _serialize_loc_entries(self, entries: List[LocEntry]) -> List[Dict[str, object]]:
        serialized: List[Dict[str, object]] = []
        for entry in entries:
            serialized.append(
                {
                    "image_index": int(entry.image_index),
                    "label": str(entry.label),
                    "coords": [float(value) for value in entry.coords],
                }
            )
        return serialized

    def _build_grounding_round_record(
        self,
        *,
        stage: str,
        round_index: int,
        raw_output: str,
        candidate_entries: List[LocEntry],
        accepted_entries: Optional[List[LocEntry]] = None,
        accepted: Optional[bool] = None,
        verifier_prompt: Optional[str] = None,
        verifier_output: Optional[str] = None,
        parsed_corrections: Optional[str] = None,
        raw_verifier_feedback: Optional[str] = None,
        feedback_info: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if accepted_entries is None:
            accepted_entries = candidate_entries

        record: Dict[str, object] = {
            "stage": str(stage),
            "round_index": int(round_index),
            "raw_output": str(raw_output or ""),
            "bbox_payloads": self._extract_bbox_2d_payloads(str(raw_output or "")),
            "candidate_entries": self._serialize_loc_entries(candidate_entries),
            "accepted_entries": self._serialize_loc_entries(accepted_entries),
            "candidate_bbox_block": self._format_bbox_block(candidate_entries) if candidate_entries else None,
            "accepted_bbox_block": self._format_bbox_block(accepted_entries) if accepted_entries else None,
        }
        if accepted is not None:
            record["accepted"] = bool(accepted)
        if verifier_prompt is not None:
            record["verifier_prompt"] = str(verifier_prompt)
        if verifier_output is not None:
            record["verifier_output"] = str(verifier_output)
        if parsed_corrections is not None:
            record["parsed_corrections"] = str(parsed_corrections)
        if raw_verifier_feedback is not None:
            record["raw_verifier_feedback"] = str(raw_verifier_feedback)
        if feedback_info is not None:
            record["feedback_info"] = dict(feedback_info)
        return record

    def _build_clean_answer_prompt(self, query: str, latest_bbox_block: str) -> str:
        query_text = query.strip()
        return (
            f"{self.instruction_text}\n\n"
            f"Original query:\n{query_text}\n\n"
            f"Detected objects:\n{latest_bbox_block}\n\n"
            f"Here is the query again:\n{query_text}\n\n"
            "Please now answer the query by first reasoning inside <reason> tags and then putting ONLY your final "
            "answer inside <answer>. Do not round answers, express all ratios as unrounded decimals. "
            "Do not output another <bbox_2d>."
        )

    def _is_grounding_only_task(self, task_name: str) -> bool:
        normalized = str(task_name or "").strip().lower()
        return normalized.startswith("convseg") or normalized.startswith("refcoco_m") or (
            normalized.startswith("refcoco") and "_bbox" in normalized
        )

    def _generate_with_verifier(
        self,
        messages: List[Dict[str, object]],
        gen_kwargs: Dict[str, object],
        visuals: List[Image.Image],
        query: str,
        grounding_only: bool = False,
    ) -> Tuple[str, Dict[str, object]]:
        output_chunks: List[str] = []
        generation_metadata: Dict[str, object] = {
            "grounding_only": bool(grounding_only),
            "requested_grounding_verifier_rounds": int(self.loc_verifier_rounds),
            "performed_grounding_verifier_rounds": 0,
            "grounding_rounds": [],
        }
        max_new_tokens_per_chunk = self.generation_chunk_max_new_tokens
        bbox_line_format = '1: label="object_name", [x_min, y_min, x_max, y_max]'
        total_generated_tokens = 0
        max_total_tokens = self.generation_max_new_tokens
        if max_total_tokens <= 0:
            generation_metadata["grounding_stop_reason"] = "generation_budget_exhausted_before_start"
            return "", generation_metadata

        remaining = max_total_tokens - total_generated_tokens
        chunk, new_tokens = self._generate_once(
            messages,
            gen_kwargs,
            stop_sequences=["</bbox_2d>"],
            max_new_tokens=max(1, min(max_new_tokens_per_chunk, remaining)),
        )
        total_generated_tokens += new_tokens
        if not chunk.strip():
            generation_metadata["grounding_stop_reason"] = "empty_initial_grounding_output"
            return "", generation_metadata
        output_chunks.append(chunk)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})

        loc_payloads = self._extract_bbox_2d_payloads(chunk)
        if len(loc_payloads) != 1:
            generation_metadata["grounding_stop_reason"] = "initial_grounding_output_missing_single_bbox_block"
            return "".join(output_chunks), generation_metadata
        loc_payload = loc_payloads[0]
        if self._has_missing_label(loc_payload):
            generation_metadata["grounding_stop_reason"] = "initial_grounding_output_missing_label"
            return "".join(output_chunks), generation_metadata
        entries = self._parse_bbox_2d_entries(loc_payload)
        if not entries or self._has_invalid_box(entries):
            generation_metadata["grounding_stop_reason"] = "initial_grounding_output_invalid_boxes"
            return "".join(output_chunks), generation_metadata

        current_entries = entries
        grounding_rounds = generation_metadata["grounding_rounds"]
        assert isinstance(grounding_rounds, list)
        grounding_rounds.append(
            self._build_grounding_round_record(
                stage="initial_prediction",
                round_index=0,
                raw_output=chunk,
                candidate_entries=current_entries,
                accepted_entries=current_entries,
                accepted=True,
            )
        )

        for round_index in range(1, self.loc_verifier_rounds + 1):
            entries_by_image: Dict[int, List[LocEntry]] = {}
            for entry in current_entries:
                entries_by_image.setdefault(entry.image_index, []).append(entry)

            original_images: List[Image.Image] = []
            overlay_images: List[Image.Image] = []
            for image_idx in range(1, len(visuals) + 1):
                original = visuals[image_idx - 1].convert("RGB")
                image_entries = entries_by_image.get(image_idx, [])
                overlay = self._overlay_boxes(original, image_entries)
                original_images.append(original)
                overlay_images.append(overlay)

            verifier_prompt = self._build_verifier_prompt(current_entries, len(visuals))
            verifier_output = self._run_verifier(original_images, overlay_images, verifier_prompt)
            corrections, raw_verifier_feedback, feedback_info = self._parse_verifier_feedback(verifier_output, current_entries)

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I have some feedback for you to incorporate. "
                                f"Please output ONLY one <bbox_2d> block using lines formatted as {bbox_line_format} "
                                "that incorporates the feedback.\n"
                                f"Feedback: {corrections}\n"
                                "You MUST re-predict ALL boxes, including unchanged ones, and keep indices sequential "
                                "starting at 1. You MUST incorporate the feedback and MUST NOT make unrelated changes."
                            ),
                        }
                    ],
                }
            )
            remaining = max_total_tokens - total_generated_tokens
            if remaining <= 0:
                generation_metadata["grounding_stop_reason"] = "generation_budget_exhausted_during_grounding_verification"
                return "".join(output_chunks), generation_metadata
            correction_chunk, correction_tokens = self._generate_once(
                messages,
                gen_kwargs,
                stop_sequences=["</bbox_2d>"],
                max_new_tokens=max(1, min(max_new_tokens_per_chunk, remaining)),
            )
            total_generated_tokens += correction_tokens
            output_chunks.append(correction_chunk)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": correction_chunk}]})

            corrected_entries: List[LocEntry] = []
            accepted = False
            corrected_payloads = self._extract_bbox_2d_payloads(correction_chunk)
            if len(corrected_payloads) != 1:
                accepted_entries = current_entries
            else:
                corrected_payload = corrected_payloads[0]
                if not self._has_missing_label(corrected_payload):
                    corrected_entries = self._parse_bbox_2d_entries(corrected_payload)
                    if corrected_entries and not self._has_invalid_box(corrected_entries):
                        accepted = True
                        current_entries = corrected_entries
                accepted_entries = current_entries

            grounding_rounds.append(
                self._build_grounding_round_record(
                    stage="loc_verifier_round",
                    round_index=round_index,
                    raw_output=correction_chunk,
                    candidate_entries=corrected_entries,
                    accepted_entries=accepted_entries,
                    accepted=accepted,
                    verifier_prompt=verifier_prompt,
                    verifier_output=verifier_output,
                    parsed_corrections=corrections,
                    raw_verifier_feedback=raw_verifier_feedback,
                    feedback_info=feedback_info,
                )
            )
            generation_metadata["performed_grounding_verifier_rounds"] = round_index

        latest_bbox_block = self._format_bbox_block(current_entries)
        generation_metadata["final_bbox_block"] = latest_bbox_block
        generation_metadata["final_bbox_entries"] = self._serialize_loc_entries(current_entries)
        if grounding_only:
            generation_metadata["grounding_stop_reason"] = "grounding_only_task"
            return "".join(output_chunks), generation_metadata
        answer_prompt = self._build_clean_answer_prompt(query, latest_bbox_block)
        answer_messages = self._build_messages(answer_prompt, visuals)
        answer_chunk, _ = self._generate_once(
            answer_messages,
            gen_kwargs,
            stop_sequences=["</answer>"],
            max_new_tokens=max_new_tokens_per_chunk,
        )
        output_chunks.append(answer_chunk)
        generation_metadata["grounding_stop_reason"] = "continued_to_answer_generation"
        return "".join(output_chunks), generation_metadata

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(x[0]), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        self.last_generation_metadata = None
        all_generation_metadata: List[Dict[str, object]] = []
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            grounding_only = self._is_grounding_only_task(task)
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i, context in enumerate(contexts):
                if "<image" in context:
                    context = re.sub(r"<image\s*\d+>", "", context)
                    context = context.replace("<image>", "")
                query_text = context
                prompted_context = self._build_prompted_context(context)
                messages = self._build_messages(prompted_context, visual_list[i])

                visuals = visual_list[i] if isinstance(visual_list[i], list) else [visual_list[i]]
                visuals = [item for item in visuals if isinstance(item, Image.Image)]
                answer, metadata = self._generate_with_verifier(
                    messages,
                    dict(gen_kwargs),
                    visuals,
                    query_text,
                    grounding_only=grounding_only,
                )
                res.append(answer)
                all_generation_metadata.append(metadata)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)
                self._cleanup_after_sample()

        res = re_ords.get_original(res)
        self.last_generation_metadata = re_ords.get_original(all_generation_metadata)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
