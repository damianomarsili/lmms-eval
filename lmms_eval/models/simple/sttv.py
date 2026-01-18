import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

VERIFIER_BOX_RULES = """- The target object is the main content of the box (>=70% of box area)
- The box covers most of the object (>=80% of object)
- The box is not wildly oversized"""

VERIFIER_POINT_RULES = """- The point lands on the target object (>=90% on-object)
- The point is not on the correct object."""

POINT_COLOR = (0, 0, 255)
BOX_COLOR = (0, 0, 255)
BOX_FILL_RGBA = (0, 0, 255, 60)
POINT_ALPHA = 220
FONT_SCALE = 0.022
POINT_RADIUS_SCALE = 0.012
BOX_OUTLINE_SCALE = 0.005
LABEL_PADDING = 2


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
        instruction_mode: str = "point",
        max_image_side: int = 768,
        verifier_max_attempts: int = 3,
        verifier_max_new_tokens: int = 256,
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

        self.verifier_max_attempts = int(verifier_max_attempts)
        self.verifier_max_new_tokens = int(verifier_max_new_tokens)
        self.verifier_image_side = int(verifier_image_side)

        self.prompt_template = self._load_prompt_template(prompt_path, self.depth_enabled)
        self.instruction_text = self._load_instruction_text(self.instruction_mode)
        self.depth_instruction_text = self._load_depth_instruction_text(self.instruction_mode) if self.depth_enabled else None
        self.verifier_template = self._load_verifier_template()

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

    def _load_verifier_template(self) -> str:
        prompt_file = Path(__file__).resolve().parents[3] / "prompts" / "verifier_instructions.txt"
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
        max_new_tokens: int = 512,
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
        answers = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return answers[0]

    def _extract_last_loc_payload(self, text: str) -> str:
        matches = re.findall(r"(?is)<loc>(.*?)</loc>", text)
        if not matches:
            return ""
        return matches[-1].strip()

    def _split_image_prefix(self, chunk: str) -> Tuple[int, str]:
        match = re.match(r"\s*image_(\d+)\s*,\s*(.*)", chunk, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), match.group(2).strip()
        return 1, chunk.strip()

    def _parse_loc_entries(self, payload: str, mode: str) -> List[LocEntry]:
        entries: List[LocEntry] = []
        chunks = [chunk.strip() for chunk in re.split(r"\s*;\s*", payload) if chunk.strip()]
        last_label = ""
        last_label_by_image: Dict[int, str] = {}
        for chunk in chunks:
            image_idx, remainder = self._split_image_prefix(chunk)
            if not remainder:
                continue
            if ":" in remainder:
                label, coords = remainder.split(":", 1)
                label = label.strip()
                last_label = label
                last_label_by_image[image_idx] = label
            else:
                label = last_label_by_image.get(image_idx, last_label)
                coords = remainder
                if not label:
                    continue
            numbers = re.findall(r"-?\d+(?:\.\d+)?", coords)
            if mode == "point" and len(numbers) >= 2:
                coords_tuple = (float(numbers[0]), float(numbers[1]))
            elif mode == "box" and len(numbers) >= 4:
                coords_tuple = tuple(float(n) for n in numbers[:4])
            else:
                continue
            entries.append(LocEntry(image_index=image_idx, label=label, coords=coords_tuple))
        return entries

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

    def _overlay_points(self, image: Image.Image, entries: List[LocEntry]) -> Image.Image:
        base = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = self._load_font(base)

        width, height = base.size
        radius = max(6, int(min(width, height) * POINT_RADIUS_SCALE))
        idx = 1
        for entry in entries:
            x, y = self._scale_point(entry.coords[0], entry.coords[1], width, height)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=POINT_COLOR + (POINT_ALPHA,),
                outline=(255, 255, 255, 255),
                width=2,
            )
            tag = f"{entry.label}#{idx}"
            text_w, text_h = self._measure_text(draw, tag, font)
            text_x = max(0, x - radius - LABEL_PADDING - text_w)
            text_y = max(0, y - radius - LABEL_PADDING - text_h)
            draw.text((text_x, text_y), tag, fill=(255, 255, 255, 255), font=font)
            idx += 1
        return Image.alpha_composite(base, overlay).convert("RGB")

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

    def _build_verifier_prompt(self, entries: List[LocEntry], image_count: int, mode: str) -> str:
        targets = sorted({entry.label for entry in entries})
        targets_str = ", ".join(targets) if targets else "(none)"

        lines: List[str] = []
        for i, entry in enumerate(entries, 1):
            prefix = f"image_{entry.image_index}," if image_count > 1 else ""
            if mode == "box":
                x1, y1, x2, y2 = entry.coords[:4]
                lines.append(f"{i}) {prefix}{entry.label}:{int(x1)},{int(y1)},{int(x2)},{int(y2)}")
            else:
                x, y = entry.coords[:2]
                lines.append(f"{i}) {prefix}{entry.label}:{int(x)},{int(y)}")

        preds_str = "\n".join(lines) if lines else "(none)"
        mode_word = "box" if mode == "box" else "point"
        rules = VERIFIER_BOX_RULES if mode == "box" else VERIFIER_POINT_RULES

        return self.verifier_template.format(
            mode_word=mode_word,
            targets=targets_str,
            preds=preds_str,
            rules=rules,
        )

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
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

    def _parse_verifier_answer(self, text: str) -> Tuple[str, str, str]:
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

    def _generate_with_verifier(
        self,
        messages: List[Dict[str, object]],
        gen_kwargs: Dict[str, object],
        visuals: List[Image.Image],
    ) -> str:
        output_chunks: List[str] = []
        failures = 0
        retry_expected = False
        retry_payload = ""
        last_verifier_text = ""
        last_verifier_feedback = ""

        while True:
            chunk = self._generate_once(messages, gen_kwargs, stop_sequences=["</loc>", "</answer>"])
            output_chunks.append(chunk)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})

            if "</answer>" in chunk:
                return "".join(output_chunks)

            loc_payload = self._extract_last_loc_payload(chunk)
            if not loc_payload:
                return "".join(output_chunks)

            normalized_payload = re.sub(r"\s+", "", loc_payload)
            if retry_expected and normalized_payload == re.sub(r"\s+", "", retry_payload):
                failures += 1
                verifier_tag = (
                    f"<verifier>{last_verifier_text}</verifier>"
                    if last_verifier_text
                    else "<verifier>incorrect</verifier>"
                )
                feedback_text = (
                    last_verifier_feedback or "adjust the coordinates to match the correct object."
                )
                feedback_line = f"\nThe correct detection should be: {feedback_text}"
                output_chunks.append(verifier_tag)
                if failures >= self.verifier_max_attempts:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{verifier_tag}\nverifier check failed too many times, move on with your prediction.",
                                }
                            ],
                        }
                    )
                    final_chunk = self._generate_once(messages, gen_kwargs, max_new_tokens=512)
                    output_chunks.append(final_chunk)
                    return "".join(output_chunks)

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"{verifier_tag}{feedback_line}\nYou repeated the same <loc> coordinates. "
                                    "You must change them. Output only a corrected <loc> step."
                                ),
                            }
                        ],
                    }
                )
                continue

            entries = self._parse_loc_entries(loc_payload, self.instruction_mode)
            if not entries:
                return "".join(output_chunks)

            entries_by_image: Dict[int, List[LocEntry]] = {}
            for entry in entries:
                entries_by_image.setdefault(entry.image_index, []).append(entry)

            original_images: List[Image.Image] = []
            overlay_images: List[Image.Image] = []
            for image_idx in range(1, len(visuals) + 1):
                original = visuals[image_idx - 1].convert("RGB")
                image_entries = entries_by_image.get(image_idx, [])
                if self.instruction_mode == "box":
                    overlay = self._overlay_boxes(original, image_entries)
                else:
                    overlay = self._overlay_points(original, image_entries)
                original_images.append(original)
                overlay_images.append(overlay)

            verifier_prompt = self._build_verifier_prompt(entries, len(visuals), self.instruction_mode)
            verifier_output = self._run_verifier(original_images, overlay_images, verifier_prompt)
            verdict, verifier_text, verifier_feedback = self._parse_verifier_answer(verifier_output)

            if verdict == "correct":
                verifier_tag = "<verifier>correct</verifier>"
            else:
                verifier_tag = f"<verifier>{verifier_text}</verifier>"
            output_chunks.append(verifier_tag)

            if verdict == "correct":
                messages.append({"role": "user", "content": [{"type": "text", "text": verifier_tag}]})
                retry_expected = False
                retry_payload = ""
                last_verifier_text = ""
                last_verifier_feedback = ""
                continue

            failures += 1
            retry_expected = True
            retry_payload = loc_payload
            last_verifier_text = verifier_text
            last_verifier_feedback = verifier_feedback
            if failures >= self.verifier_max_attempts:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{verifier_tag}\nverifier check failed too many times, move on with your prediction.",
                            }
                        ],
                    }
                )
                final_chunk = self._generate_once(messages, gen_kwargs, max_new_tokens=512)
                output_chunks.append(final_chunk)
                return "".join(output_chunks)

            feedback_text = (
                verifier_feedback or "adjust the coordinates to match the correct object."
            )
            feedback_line = f"\nThe correct detection should be: {feedback_text}"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{verifier_tag}{feedback_line}\nUse the verifier feedback above to correct your "
                                "last <loc> step. You must change at least one coordinate and should not repeat "
                                "the same <loc>. Output only a corrected <loc> step."
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
                prompted_context = self._build_prompted_context(context)
                messages = self._build_messages(prompted_context, visual_list[i])

                visuals = visual_list[i] if isinstance(visual_list[i], list) else [visual_list[i]]
                visuals = [item for item in visuals if isinstance(item, Image.Image)]
                answer = self._generate_with_verifier(messages, dict(gen_kwargs), visuals)
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
