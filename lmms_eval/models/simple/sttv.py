import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

BOX_COLOR = (0, 0, 255)
BOX_FILL_RGBA = (0, 0, 255, 25)
FONT_SCALE = 0.022
BOX_OUTLINE_SCALE = 0.005
LABEL_PADDING = 2
BBOX_2D_ENTRY_PATTERN = re.compile(
    r'^\s*label\s*=\s*"(?P<label>[^"\n]+?)"\s*,\s*\[(?P<coords>[^\]]+)\]\s*$',
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class LocEntry:
    image_index: int
    label: str
    coords: Tuple[float, ...]


@register_model("sttv")
class STTV(lmms):
    """
    STTV Model with verifier.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        depth: Union[bool, str, int, float] = False,
        prompt_path: Optional[str] = None,
        instruction_mode: str = "box",
        max_image_side: int = 768,
        verifier_max_attempts: int = 3,
        verifier_max_new_tokens: int = 96,
        verifier_image_side: int = 1024,
        trust_remote_code: Optional[bool] = True,
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
        self.instruction_mode = instruction_mode.lower()
        if self.instruction_mode != "box":
            raise ValueError(f"Only box mode is supported; got instruction_mode={instruction_mode}")

        self.verifier_max_attempts = int(verifier_max_attempts)
        self.verifier_max_new_tokens = int(verifier_max_new_tokens)
        self.verifier_image_side = int(verifier_image_side)

        self.prompt_template = self._load_prompt_template(prompt_path, self.depth_enabled)
        self.instruction_text = self._load_instruction_text(self.instruction_mode)
        self.depth_instruction_text = self._load_depth_instruction_text(self.instruction_mode) if self.depth_enabled else None
        self.verifier_template = self._load_verifier_template()
        self.self_verifier_template = self._load_self_verifier_template()
        self.self_verifier_max_new_tokens = int(self.verifier_max_new_tokens)

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
                prompt_file = Path(__file__).resolve().parents[3] / "prompts" / "sttv_verifier.txt"
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

    def _load_verifier_template(self) -> str:
        prompt_file = Path(__file__).resolve().parents[3] / "prompts" / "verifier_instructions.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Verifier prompt file not found: {prompt_file}")
        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Verifier prompt file is empty: {prompt_file}")
        return prompt_text

    def _load_self_verifier_template(self) -> str:
        prompt_file = Path(__file__).resolve().parents[3] / "prompts" / "self_verifier_instructions.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Self-verifier prompt file not found: {prompt_file}")
        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Self-verifier prompt file is empty: {prompt_file}")
        return prompt_text

    def _build_prompted_context(self, context: str) -> str:
        if self.depth_enabled:
            if self.depth_instruction_text is None:
                raise ValueError("Depth mode enabled but depth instructions are missing.")
            return self.prompt_template.format(self.instruction_text, self.depth_instruction_text, context.strip())
        return self.prompt_template.format(self.instruction_text, context.strip())

    def _cleanup_after_sample(self) -> None:
        if not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()

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
        stop_sequences: Optional[List[str]] = None,
        max_new_tokens: int = 256,
    ) -> Tuple[str, int]:
        del gen_kwargs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generate_args: Dict[str, object] = {"max_new_tokens": max_new_tokens}
        if stop_sequences:
            stopping_criteria = stop_sequences_criteria(
                self.tokenizer,
                stop_sequences,
                inputs.input_ids.shape[-1],
                inputs.input_ids.shape[0],
            )
            generate_args["stopping_criteria"] = stopping_criteria

        cont = self.model.generate(**inputs, **generate_args)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        token_count = sum(len(ids) for ids in generated_ids_trimmed)
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded = answers[0]
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
        return decoded, token_count

    def _extract_last_bbox_2d_payload(self, text: str) -> str:
        matches = re.findall(r"(?is)<bbox_2d>(.*?)</bbox_2d>", text)
        if not matches:
            return ""
        return matches[-1].strip()

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

            label = match.group("label").strip()
            if not label:
                return []

            numbers = re.findall(r"-?\d+(?:\.\d+)?", match.group("coords"))
            if len(numbers) != 4:
                return []
            coords_tuple = tuple(float(n) for n in numbers[:4])
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
            if x2 < x1 or y2 < y1:
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
        outline_w = max(4, int(min(width, height) * BOX_OUTLINE_SCALE))
        idx = 1
        for entry in entries:
            x1, y1 = self._scale_point(entry.coords[0], entry.coords[1], width, height)
            x2, y2 = self._scale_point(entry.coords[2], entry.coords[3], width, height)
            draw.rectangle((x1, y1, x2, y2), fill=BOX_FILL_RGBA)
            draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOR + (255,), width=outline_w)
            tag = f"{entry.label}#{idx}"
            tag_y = max(0, y1 - getattr(font, "size", 12) - 3)
            draw.text((x1 + 2, tag_y), tag, fill=(255, 255, 255, 255), font=font)
            idx += 1
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
            content.append({"type": "image", "image": original})
            content.append({"type": "image", "image": overlay})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=self.verifier_max_new_tokens)
        trimmed = outputs[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _run_self_verifier(self, visuals: List[Image.Image], prompt: str) -> str:
        messages = self._build_messages(prompt, visuals)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=self.self_verifier_max_new_tokens)
        trimmed = outputs[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _parse_verifier_answer(self, text: str) -> Tuple[str, str, str]:
        text = text.replace("<|im_end|>", "").strip()
        match = re.search(r"(?is)<answer>(.*?)</answer>", text)
        cleaned = text.strip()
        if not match:
            return "incorrect", cleaned, cleaned

        answer_text = match.group(1).strip()
        tail = text[match.end() :].strip()
        verdict_lower = answer_text.lower()

        if verdict_lower.startswith("correct"):
            return "correct", cleaned, ""

        verdict = "incorrect"
        if verdict_lower.startswith("incorrect"):
            feedback = re.sub(r"(?is)^incorrect[:\\s-]*", "", answer_text).strip()
        else:
            feedback = answer_text.strip()

        if tail:
            feedback = f"{feedback} {tail}".strip() if feedback else tail
        return verdict, cleaned, feedback

    def _parse_self_verifier_answer(self, text: str) -> Tuple[str, str, str]:
        text = text.replace("<|im_end|>", "").strip()
        match = re.search(r"(?is)<answer>(.*?)</answer>", text)
        cleaned = text.strip()
        if not match:
            return "incorrect", cleaned, cleaned

        answer_text = match.group(1).strip()
        tail = text[match.end() :].strip()
        verdict_lower = answer_text.lower()

        if verdict_lower.startswith(("valid", "correct")):
            return "correct", cleaned, ""

        verdict = "incorrect"
        if verdict_lower.startswith("invalid"):
            feedback = re.sub(r"(?is)^invalid[:\\s-]*", "", answer_text).strip()
        elif verdict_lower.startswith("incorrect"):
            feedback = re.sub(r"(?is)^incorrect[:\\s-]*", "", answer_text).strip()
        else:
            feedback = answer_text.strip()

        if tail:
            feedback = f"{feedback} {tail}".strip() if feedback else tail
        return verdict, cleaned, feedback

    def _format_self_verifier_feedback(self, verdict: str, feedback: str) -> str:
        if verdict != "incorrect":
            return ""
        cleaned = feedback.strip()
        if cleaned:
            cleaned = cleaned.rstrip(".!?")
        if not cleaned:
            cleaned = "please review your reasoning and update the final answer"
        return f"I'm not happy with your solution. Here is some feedback: {cleaned}. Please retry with this feedback."

    def _iter_tag_payloads(self, text: str) -> List[Tuple[str, str]]:
        entries: List[Tuple[str, str]] = []
        tag_open = re.compile(r"(?is)<(reason|depth|bbox_2d|answer)>")
        i = 0
        while True:
            open_match = tag_open.search(text, i)
            if not open_match:
                break
            tag = open_match.group(1).lower()
            content_start = open_match.end()
            close_match = re.search(rf"(?is)</{tag}>", text[content_start:])
            if close_match:
                content_end = content_start + close_match.start()
                end_index = content_start + close_match.end()
            else:
                next_tag = re.search(r"(?is)<(reason|depth|bbox_2d|answer|verifier)>", text[content_start:])
                if next_tag:
                    content_end = content_start + next_tag.start()
                    end_index = content_start + next_tag.start()
                else:
                    content_end = len(text)
                    end_index = len(text)
            payload = text[content_start:content_end].strip()
            entries.append((tag, payload))
            i = end_index
        return entries

    def _collapse_for_self_verifier(self, text: str) -> str:
        cleaned_text = re.sub(r"(?is)<verifier>.*?</verifier>", "", text).strip()
        entries = self._iter_tag_payloads(cleaned_text)
        if not entries:
            return cleaned_text

        parts: List[str] = []
        for tag, payload in entries:
            if tag == "reason":
                parts.append(f"<reason>{payload}</reason>")
            elif tag == "answer":
                parts.append(f"<answer>{payload}</answer>")

        return "\n".join(parts) if parts else ""

    def _build_self_verifier_prompt(self, query: str, collapsed_output: str) -> str:
        return self.self_verifier_template.format(query=query, steps=collapsed_output).strip()

    def _generate_with_verifier(
        self,
        messages: List[Dict[str, object]],
        gen_kwargs: Dict[str, object],
        visuals: List[Image.Image],
        query: str,
    ) -> str:
        output_chunks: List[str] = []
        failures = 0
        retry_expected = False
        retry_payload = ""
        last_verifier_feedback = ""
        step_count = 0
        max_failed_loc_rounds = 3
        max_steps = 8
        max_new_tokens_per_chunk = 256
        max_final_answer_tokens = 64
        final_answer_prompt = "I can now predict the final answer which is: "
        final_fail_prompt = "I am unable to locate the objects correctly. Provide a final <reason> step and " "then the final <answer>."
        bbox_line_format = (
            'label="object_name", [x_min, y_min, x_max, y_max]'
            if self.instruction_mode == "box"
            else 'label="object_name", [x, y]'
        )
        empty_loc_prompt = "You did not place an object in your <bbox_2d> tags, please review the format and try again."
        missing_label_prompt = (
            f"Missing required label syntax. Use one line per object as: {bbox_line_format}."
        )
        generic_format_prompt = (
            f"Formatting error in <bbox_2d>. Use ONLY lines in this exact format: {bbox_line_format}."
        )
        total_generated_tokens = 0
        max_total_tokens = max_steps * max_new_tokens_per_chunk
        max_self_verify_attempts = self.verifier_max_attempts
        max_empty_generation_attempts = 3
        self_verify_failures = 0
        missing_loc_failures = 0
        empty_generation_failures = 0

        def _format_incorrect_message(feedback: str) -> str:
            cleaned = feedback.strip()
            if cleaned:
                cleaned = cleaned.rstrip(".!?")
            if not cleaned:
                cleaned = "Please adjust the coordinates to match the correct object"
            return f"That looks incorrect. {cleaned}. Please try again and update your prediction."

        def _inject_final_answer(prompt_prefix: str, max_new_tokens: int = max_new_tokens_per_chunk) -> str:
            nonlocal total_generated_tokens
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_prefix}],
                }
            )
            final_chunk, new_tokens = self._generate_once(messages, gen_kwargs, stop_sequences=["</answer>"], max_new_tokens=max_new_tokens)
            total_generated_tokens += new_tokens
            output_chunks.append(final_chunk)
            return "".join(output_chunks)

        accepted_loc_payloads: List[str] = []
        solution_start_index = 0

        def _run_self_verifier_once() -> Tuple[str, str, str]:
            current_output = "".join(output_chunks[solution_start_index:])
            collapsed = self._collapse_for_self_verifier(current_output)
            if not collapsed.strip():
                collapsed = re.sub(r"(?is)<verifier>.*?</verifier>", "", current_output)
                collapsed = re.sub(r"(?is)<(bbox_2d|depth)>.*?</\1>", "", collapsed)
                collapsed = collapsed.strip()
            prompt = self._build_self_verifier_prompt(query, collapsed)
            self_verifier_output = self._run_self_verifier(visuals, prompt)
            verdict, _, verifier_feedback = self._parse_self_verifier_answer(self_verifier_output)
            return verdict, verifier_feedback, prompt

        def _handle_self_verifier() -> bool:
            nonlocal solution_start_index
            nonlocal self_verify_failures
            verdict, verifier_feedback, _ = _run_self_verifier_once()
            if verdict == "correct":
                self_verify_failures = 0
                output_chunks.append("<verifier>self-verifier: correct</verifier>")
                return True
            self_verify_failures += 1
            if self_verify_failures >= max_self_verify_attempts:
                output_chunks.append("<verifier>self-verifier: max attempts reached</verifier>")
                return True
            feedback_text = self._format_self_verifier_feedback(verdict, verifier_feedback)
            if feedback_text:
                output_chunks.append(f"<verifier>{feedback_text}</verifier>")
                messages.append({"role": "user", "content": [{"type": "text", "text": feedback_text}]})
                solution_start_index = len(output_chunks)
                return False
            return True

        def _inject_and_verify(
            prefix: str,
            max_new_tokens: int = max_new_tokens_per_chunk,
            count_step: bool = False,
        ) -> Optional[str]:
            nonlocal step_count
            if count_step:
                step_count += 1
            _inject_final_answer(prefix, max_new_tokens=max_new_tokens)
            if _handle_self_verifier():
                return "".join(output_chunks)
            return None

        while True:
            chunk, new_tokens = self._generate_once(
                messages,
                gen_kwargs,
                stop_sequences=["</bbox_2d>", "</answer>"],
                max_new_tokens=max_new_tokens_per_chunk,
            )
            if new_tokens == 0 and not chunk.strip():
                empty_generation_failures += 1
                if empty_generation_failures >= max_empty_generation_attempts:
                    return _inject_final_answer(final_fail_prompt, max_new_tokens=max_final_answer_tokens)
            else:
                empty_generation_failures = 0
            total_generated_tokens += new_tokens
            output_chunks.append(chunk)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})

            if total_generated_tokens >= max_total_tokens:
                return _inject_final_answer(final_answer_prompt, max_new_tokens=max_final_answer_tokens)
            if "</answer>" in chunk:
                if _handle_self_verifier():
                    return "".join(output_chunks)
                continue

            reason_steps = len(re.findall(r"(?is)<reason>.*?</reason>", chunk))
            if reason_steps == 0 and "<reason>" in chunk:
                reason_steps = 1
            depth_steps = 0
            if self.depth_enabled:
                depth_steps = len(re.findall(r"(?is)<depth>.*?</depth>", chunk))
                if depth_steps == 0 and "<depth>" in chunk:
                    depth_steps = 1
            step_count += reason_steps + depth_steps
            if step_count >= max_steps:
                return _inject_final_answer(final_answer_prompt)

            loc_payload = self._extract_last_bbox_2d_payload(chunk)
            if not loc_payload:
                if "<bbox_2d>" in chunk:
                    failures += 1
                    output_chunks.append(f"<verifier>{empty_loc_prompt}</verifier>")
                    if failures >= max_failed_loc_rounds:
                        injected = _inject_and_verify(final_fail_prompt, count_step=True)
                        if injected is not None:
                            return injected
                        continue
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": empty_loc_prompt}],
                        }
                    )
                    continue
                if self.depth_enabled and ("<depth>" in chunk or "<reason>" in chunk):
                    continue
                missing_loc_failures += 1
                if missing_loc_failures >= max_failed_loc_rounds:
                    return _inject_final_answer(final_fail_prompt, max_new_tokens=max_final_answer_tokens)
                injected = _inject_and_verify(final_answer_prompt)
                if injected is not None:
                    return injected
                continue
            missing_loc_failures = 0

            if self._has_missing_label(loc_payload):
                failures += 1
                output_chunks.append(f"<verifier>{missing_label_prompt}</verifier>")
                if failures >= max_failed_loc_rounds:
                    injected = _inject_and_verify(final_fail_prompt, count_step=True)
                    if injected is not None:
                        return injected
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": missing_label_prompt}],
                    }
                )
                continue

            normalized_payload = re.sub(r"\s+", "", loc_payload)
            if retry_expected and normalized_payload == re.sub(r"\s+", "", retry_payload):
                failures += 1
                feedback_text = last_verifier_feedback or "adjust the coordinates to match the correct object."
                verifier_message = _format_incorrect_message(feedback_text)
                if failures >= max_failed_loc_rounds:
                    injected = _inject_and_verify(final_fail_prompt, count_step=True)
                    if injected is not None:
                        return injected
                    continue

                output_chunks.append(f"<verifier>{verifier_message}</verifier>")
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (f"{verifier_message}\nYou repeated the same <bbox_2d> coordinates. " "You must change them. Output only a corrected <bbox_2d> step."),
                            }
                        ],
                    }
                )
                continue

            entries = self._parse_bbox_2d_entries(loc_payload)
            if not entries:
                failures += 1
                output_chunks.append(f"<verifier>{generic_format_prompt}</verifier>")
                if failures >= max_failed_loc_rounds:
                    injected = _inject_and_verify(final_fail_prompt, count_step=True)
                    if injected is not None:
                        return injected
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": generic_format_prompt}],
                    }
                )
                continue
            if self._has_invalid_box(entries):
                failures += 1
                output_chunks.append(f"<verifier>{generic_format_prompt}</verifier>")
                if failures >= max_failed_loc_rounds:
                    injected = _inject_and_verify(final_fail_prompt, count_step=True)
                    if injected is not None:
                        return injected
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": generic_format_prompt}],
                    }
                )
                continue

            entries_by_image: Dict[int, List[LocEntry]] = {}
            for entry in entries:
                entries_by_image.setdefault(entry.image_index, []).append(entry)

            original_images: List[Image.Image] = []
            overlay_images: List[Image.Image] = []
            for image_idx in range(1, len(visuals) + 1):
                original = visuals[image_idx - 1].convert("RGB")
                image_entries = entries_by_image.get(image_idx, [])
                overlay = self._overlay_boxes(original, image_entries)
                original_images.append(original)
                overlay_images.append(overlay)

            verifier_prompt = self._build_verifier_prompt(entries, len(visuals))
            verifier_output = self._run_verifier(original_images, overlay_images, verifier_prompt)
            verdict, _, verifier_feedback = self._parse_verifier_answer(verifier_output)

            if verdict == "correct":
                verifier_message = "That looks correct, you can proceed."
            else:
                feedback_text = verifier_feedback or "adjust the coordinates to match the correct object."
                verifier_message = _format_incorrect_message(feedback_text)

            output_chunks.append(f"<verifier>{verifier_message}</verifier>")
            if verdict == "correct":
                accepted_loc_payloads.append(loc_payload)
                step_count += 1
                if step_count >= max_steps:
                    return _inject_final_answer(f"{verifier_message}\n{final_answer_prompt}")
                messages.append({"role": "user", "content": [{"type": "text", "text": verifier_message}]})
                retry_expected = False
                retry_payload = ""
                last_verifier_feedback = ""
                continue

            failures += 1
            retry_expected = True
            retry_payload = loc_payload
            last_verifier_feedback = verifier_feedback
            if failures >= max_failed_loc_rounds:
                injected = _inject_and_verify(final_fail_prompt, count_step=True)
                if injected is not None:
                    return injected
                continue

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{verifier_message}\nUse the verifier feedback above to correct your " "last <bbox_2d> step. You must change at least one coordinate and should not repeat " "the same <bbox_2d>. Output only a corrected <bbox_2d> step."
                            ),
                        }
                    ],
                }
            )

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
                query_text = context
                prompted_context = self._build_prompted_context(context)
                messages = self._build_messages(prompted_context, visual_list[i])

                visuals = visual_list[i] if isinstance(visual_list[i], list) else [visual_list[i]]
                visuals = [item for item in visuals if isinstance(item, Image.Image)]
                answer = self._generate_with_verifier(messages, dict(gen_kwargs), visuals, query_text)
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)
                self._cleanup_after_sample()

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
