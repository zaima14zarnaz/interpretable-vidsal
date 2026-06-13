"""Cluster saved top examples for one concept without saliency model or LLM."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from cluster_concepts import (
    cluster_examples_by_clip_features,
    cluster_examples_by_patch_difference,
    load_clip_model,
)


@dataclass
class SavedExample:
    concept_idx: int
    rank: int
    score: float
    video_name: str
    batch_index: int
    patch_index: int
    grid_hw: Optional[Tuple[int, int]]
    image_path: str
    frame_t_path: str
    frame_t1_path: str
    pair_path: str
    full_t_boxed_path: str = ""
    full_t1_boxed_path: str = ""
    context_panel_path: str = ""
    activation_score: float = 0.0
    patch_pred_saliency: float = 0.0
    retrieval_score: float = 0.0
    is_salient_region: bool = False


def load_examples_from_metadata(concept_dir: Path, metadata_path: Path) -> List[SavedExample]:
    """Load SavedExample entries from top_examples/examples_metadata.json."""
    entries = json.loads(metadata_path.read_text(encoding="utf-8"))
    examples: List[SavedExample] = []

    for entry in entries:
        pair_path = str(entry["pair_path"])
        activation_score = float(entry.get("activation_score", entry.get("score", 0.0)))
        patch_pred_saliency = float(
            entry.get(
                "patch_pred_saliency",
                entry.get("patch_gt_saliency", 0.0),
            )
        )
        retrieval_score = float(
            entry.get(
                "retrieval_score",
                entry.get("activation_score", entry.get("score", 0.0)),
            )
        )
        is_salient_region = bool(
            entry.get("is_salient_region", patch_pred_saliency > 0.0)
        )
        examples.append(
            SavedExample(
                concept_idx=int(entry.get("global_concept_index", 0)),
                rank=int(entry["rank"]),
                score=retrieval_score,
                video_name=str(entry.get("video_filename", "")),
                batch_index=int(entry.get("batch_index", -1)),
                patch_index=int(entry.get("sample_index", entry.get("patch_index", -1))),
                grid_hw=None,
                image_path=pair_path,
                frame_t_path=str(entry["frame_t_path"]),
                frame_t1_path=str(entry["frame_t1_path"]),
                pair_path=pair_path,
                full_t_boxed_path=str(
                    entry.get("full_t_boxed_path", entry.get("pair_path", ""))
                ),
                full_t1_boxed_path=str(
                    entry.get("full_t1_boxed_path", entry.get("pair_path", ""))
                ),
                context_panel_path=str(
                    entry.get("context_panel_path", entry.get("pair_path", ""))
                ),
                activation_score=activation_score,
                patch_pred_saliency=patch_pred_saliency,
                retrieval_score=retrieval_score,
                is_salient_region=is_salient_region,
            )
        )

    return sorted(examples, key=lambda item: item.rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster saved top examples for one concept directory using CLIP or "
            "patch-difference features. Does not load the saliency model or LLM."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Optional dataset path (accepted for CLI compatibility; unused in cluster-only mode).",
    )
    parser.add_argument(
        "--cluster-only-concept-dir",
        required=True,
        help="Concept directory containing top_examples/examples_metadata.json.",
    )
    parser.add_argument(
        "--cluster-feature",
        choices=["diff", "clip"],
        default="clip",
        help="Feature type used to cluster top examples.",
    )
    parser.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model used for visual feature clustering.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=16,
        help="Batch size for CLIP visual embedding extraction.",
    )
    parser.add_argument(
        "--clip-image-source",
        choices=["frame_t", "large_crop_t"],
        default="large_crop_t",
        help="Image source used only for CLIP feature extraction during clustering.",
    )
    parser.add_argument(
        "--clip-crop-scale",
        type=float,
        default=1.0,
        help="Scale factor for enlarged crop around the activated patch for CLIP clustering.",
    )
    parser.add_argument(
        "--max-diff-clusters",
        type=int,
        default=4,
        help="Maximum number of clusters per concept.",
    )
    parser.add_argument(
        "--diff-image-size",
        type=int,
        default=64,
        help=(
            "Resize frame_t/frame_t1 patches to this square size before "
            "computing difference features."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    concept_dir = Path(args.cluster_only_concept_dir)
    metadata_path = concept_dir / "top_examples" / "examples_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing examples metadata: {metadata_path}"
        )

    examples = load_examples_from_metadata(concept_dir, metadata_path)
    if not examples:
        print(f"No examples found in {metadata_path}")
        return

    print(f"Loaded {len(examples)} examples from {metadata_path}")
    print(f"Cluster feature: {args.cluster_feature}")
    if args.cluster_feature == "clip":
        print(f"CLIP image source: {args.clip_image_source}")
        if args.clip_image_source == "large_crop_t":
            print(f"CLIP crop scale: {args.clip_crop_scale}")

    if args.cluster_feature == "clip":
        clip_model, clip_processor, device = load_clip_model(
            model_name=args.clip_model_name,
        )
        clustered = cluster_examples_by_clip_features(
            concept_dir=concept_dir,
            examples=examples,
            clip_model=clip_model,
            clip_processor=clip_processor,
            device=device,
            max_clusters=args.max_diff_clusters,
            clip_batch_size=args.clip_batch_size,
            model_name=args.clip_model_name,
            clip_image_source=args.clip_image_source,
            clip_crop_scale=args.clip_crop_scale,
        )
    else:
        image_size = (args.diff_image_size, args.diff_image_size)
        clustered = cluster_examples_by_patch_difference(
            concept_dir=concept_dir,
            examples=examples,
            max_clusters=args.max_diff_clusters,
            image_size=image_size,
            save_diff_images=True,
        )

    print("Clusters:")
    for cluster_id in sorted(clustered):
        cluster_examples = clustered[cluster_id]
        ranks = [example.rank for example in cluster_examples]
        print(
            f"cluster {cluster_id}: {len(cluster_examples)} examples, ranks={ranks}"
        )


if __name__ == "__main__":
    main()
