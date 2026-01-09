import argparse
import itertools
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import datasets
from loguru import logger as eval_logger

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "CUTE"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "lmms_eval_cute"
DEFAULT_POSITIVE = 250
DEFAULT_NEGATIVE = 250
DEFAULT_SEED = 7


@dataclass(frozen=True)
class PairSpec:
    image_a: Path
    image_b: Path
    answer: str
    object_id: str
    instance_ids: Tuple[str, str]


def _list_object_dirs(data_root: Path) -> List[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"CUTE data root not found: {data_root}")
    object_dirs = []
    for entry in data_root.iterdir():
        if not entry.is_dir():
            continue
        instance_1 = entry / "instance_1"
        instance_2 = entry / "instance_2"
        if instance_1.is_dir() and instance_2.is_dir():
            object_dirs.append(entry)
    if not object_dirs:
        raise ValueError(f"No CUTE objects found under {data_root}")
    return sorted(object_dirs, key=lambda path: path.name)


def _list_images(instance_dir: Path) -> List[Path]:
    in_the_wild = instance_dir / "in_the_wild"
    if not in_the_wild.is_dir():
        raise FileNotFoundError(f"Missing in_the_wild directory: {in_the_wild}")
    images = [path for path in in_the_wild.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    images = sorted(images, key=lambda path: path.name)
    if len(images) < 2:
        raise ValueError(f"Not enough images under {in_the_wild} (found {len(images)})")
    return images


def _build_positive_pairs(object_id: str, instance_id: str, images: Sequence[Path]) -> List[PairSpec]:
    return [
        PairSpec(
            image_a=first,
            image_b=second,
            answer="yes",
            object_id=object_id,
            instance_ids=(instance_id, instance_id),
        )
        for first, second in itertools.combinations(images, 2)
    ]


def _build_negative_pairs(object_id: str, inst1_images: Sequence[Path], inst2_images: Sequence[Path]) -> List[PairSpec]:
    return [
        PairSpec(
            image_a=first,
            image_b=second,
            answer="no",
            object_id=object_id,
            instance_ids=("instance_1", "instance_2"),
        )
        for first, second in itertools.product(inst1_images, inst2_images)
    ]


def _collect_pairs(data_root: Path) -> Tuple[List[PairSpec], List[PairSpec]]:
    positive_pairs: List[PairSpec] = []
    negative_pairs: List[PairSpec] = []
    for object_dir in _list_object_dirs(data_root):
        object_id = object_dir.name
        inst1_images = _list_images(object_dir / "instance_1")
        inst2_images = _list_images(object_dir / "instance_2")
        positive_pairs.extend(_build_positive_pairs(object_id, "instance_1", inst1_images))
        positive_pairs.extend(_build_positive_pairs(object_id, "instance_2", inst2_images))
        negative_pairs.extend(_build_negative_pairs(object_id, inst1_images, inst2_images))
    return positive_pairs, negative_pairs


def _sample_pairs(pairs: Sequence[PairSpec], count: int, rng: random.Random) -> List[PairSpec]:
    if count > len(pairs):
        raise ValueError(f"Requested {count} pairs but only {len(pairs)} available")
    return rng.sample(list(pairs), count)


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _pair_to_doc(pair: PairSpec) -> dict:
    image_a = _relative_path(pair.image_a)
    image_b = _relative_path(pair.image_b)
    return {
        "images": [image_a, image_b],
        "answer": pair.answer,
        "question_type": "pairwise",
        "extra": {
            "product_ids": [pair.object_id, pair.object_id],
            "instance_ids": list(pair.instance_ids),
            "source_paths": [image_a, image_b],
        },
    }


def build_cute_docs(
    data_root: Path,
    positive_count: int = DEFAULT_POSITIVE,
    negative_count: int = DEFAULT_NEGATIVE,
    seed: int = DEFAULT_SEED,
) -> List[dict]:
    positive_pairs, negative_pairs = _collect_pairs(data_root)
    rng = random.Random(seed)
    selected_positive = _sample_pairs(positive_pairs, positive_count, rng)
    selected_negative = _sample_pairs(negative_pairs, negative_count, rng)
    docs = [_pair_to_doc(pair) for pair in selected_positive + selected_negative]
    rng.shuffle(docs)
    return docs


def save_dataset(docs: Sequence[dict], output_dir: Path, overwrite: bool = False) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = datasets.Dataset.from_list(list(docs))
    datasets.DatasetDict({"test": dataset}).save_to_disk(str(output_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CUTE pairwise dataset for lmms-eval.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Path to the CUTE dataset root.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for the dataset.")
    parser.add_argument("--positive", type=int, default=DEFAULT_POSITIVE, help="Number of positive pairs to sample.")
    parser.add_argument("--negative", type=int, default=DEFAULT_NEGATIVE, help="Number of negative pairs to sample.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling pairs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    docs = build_cute_docs(args.data_root, args.positive, args.negative, args.seed)
    eval_logger.info(
        "Prepared {} docs ({} positive, {} negative).",
        len(docs),
        args.positive,
        args.negative,
    )
    save_dataset(docs, args.output, overwrite=args.overwrite)
    eval_logger.info("Saved CUTE dataset to {}", args.output)


if __name__ == "__main__":
    main()
