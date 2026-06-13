"""Cluster saved top patch-pair examples by visual similarity or patch differences."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from transformers import CLIPModel, CLIPProcessor

if TYPE_CHECKING:
    from explanation_generation import SavedExample

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def _resolve_example_image_path(concept_dir: Path, rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return concept_dir / path


def _fallback_frame_paths(concept_dir: Path, example: "SavedExample") -> Tuple[Path, Path]:
    top_examples_dir = concept_dir / "top_examples"
    prefix = f"example_{example.rank:03d}"
    return (
        top_examples_dir / f"{prefix}_frame_t.png",
        top_examples_dir / f"{prefix}_frame_t1.png",
    )


def _relative_to_concept_dir(concept_dir: Path, path: Path) -> str:
    return str(path.relative_to(concept_dir))


def _example_source_path(
    concept_dir: Path,
    example: "SavedExample",
    path_attr: str,
    fallback_suffix: str,
) -> Optional[Path]:
    rel_path = getattr(example, path_attr, "") or ""
    if rel_path:
        candidate = _resolve_example_image_path(concept_dir, rel_path)
        if candidate.exists():
            return candidate

    fallback = concept_dir / "top_examples" / f"example_{example.rank:03d}_{fallback_suffix}.png"
    if fallback.exists():
        return fallback
    return None


def _copy_image_file(src_path: Path, dst_path: Path) -> bool:
    if not src_path.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return True


def _build_cluster_metadata_entry(
    example: "SavedExample",
    diff_image_path: str,
    clip_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "rank": int(example.rank),
        "original_frame_t_path": str(example.frame_t_path),
        "original_frame_t1_path": str(example.frame_t1_path),
        "original_pair_path": str(example.pair_path),
        "original_context_panel_path": str(getattr(example, "context_panel_path", "") or ""),
        "diff_image_path": diff_image_path,
        "activation_score": float(example.activation_score),
        "patch_pred_saliency": float(example.patch_pred_saliency),
        "retrieval_score": float(example.retrieval_score),
        "is_salient_region": bool(example.is_salient_region),
        "video_name": str(example.video_name),
        "patch_index": int(example.patch_index),
    }
    if clip_metadata:
        for key, value in clip_metadata.items():
            if value is not None:
                entry[key] = value
    return entry


def save_clustered_top_examples(
    concept_dir: Path,
    clustered: Dict[int, List["SavedExample"]],
    diff_by_rank: Dict[int, np.ndarray],
    save_diff_images: bool = True,
    cluster_metadata_extra: Optional[Dict[str, Any]] = None,
    clip_metadata_by_rank: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    """
    Copy clustered examples into ``concept_dir/clustered_top_examples/cluster_XX/``.

    This is an additional interpretability view; it does not modify ``top_examples/``.
    """
    clustered_root = concept_dir / "clustered_top_examples"

    for cluster_id, cluster_examples in sorted(clustered.items()):
        if not cluster_examples:
            continue

        cluster_dir = clustered_root / f"cluster_{int(cluster_id):02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        metadata_entries: List[Dict[str, Any]] = []

        for example in cluster_examples:
            rank = int(example.rank)
            prefix = f"example_{rank:03d}"
            asset_specs = (
                ("frame_t_path", "frame_t", f"{prefix}_frame_t.png"),
                ("frame_t1_path", "frame_t1", f"{prefix}_frame_t1.png"),
                ("sequence_path", "sequence", f"{prefix}_sequence.png"),
                ("pair_path", "sequence", f"{prefix}_sequence.png"),
                ("context_panel_path", "context_panel", f"{prefix}_context_panel.png"),
            )

            for path_attr, fallback_suffix, dst_name in asset_specs:
                src_path = _example_source_path(
                    concept_dir,
                    example,
                    path_attr,
                    fallback_suffix,
                )
                if src_path is None:
                    continue
                _copy_image_file(src_path, cluster_dir / dst_name)

            frame_paths = getattr(example, "frame_paths", None) or []
            for frame_idx, frame_rel in enumerate(frame_paths):
                frame_src = _resolve_example_image_path(concept_dir, str(frame_rel))
                if frame_src.exists():
                    _copy_image_file(
                        frame_src,
                        cluster_dir / f"{prefix}_frame_{frame_idx:02d}.png",
                    )

            diff_image_path = ""
            diff_dst = cluster_dir / f"{prefix}_diff.png"
            if rank in diff_by_rank:
                save_difference_image(diff_by_rank[rank], diff_dst)
                diff_image_path = _relative_to_concept_dir(concept_dir, diff_dst)
            elif save_diff_images:
                src_diff = concept_dir / "top_examples" / "diff_images" / f"{prefix}_diff.png"
                if _copy_image_file(src_diff, diff_dst):
                    diff_image_path = _relative_to_concept_dir(concept_dir, diff_dst)

            metadata_entries.append(
                _build_cluster_metadata_entry(
                    example,
                    diff_image_path,
                    clip_metadata=(clip_metadata_by_rank or {}).get(rank),
                )
            )

        metadata_payload: Dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "size": len(metadata_entries),
            "samples": metadata_entries,
        }
        if cluster_metadata_extra:
            metadata_payload.update(cluster_metadata_extra)
        metadata_path = cluster_dir / "cluster_metadata.json"
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")


def load_patch_pair_arrays(
    concept_dir: Path,
    example: "SavedExample",
    image_size: Tuple[int, int] = (64, 64),
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load the frame_t and frame_t1 patch PNGs for a SavedExample.

    Return two RGB float arrays in [0,1] with shape [H,W,3].
    Return None if either file is missing or unreadable.
    """
    frame_t_rel = getattr(example, "frame_t_path", "") or ""
    frame_t1_rel = getattr(example, "frame_t1_path", "") or ""

    if frame_t_rel and frame_t1_rel:
        frame_t_path = _resolve_example_image_path(concept_dir, frame_t_rel)
        frame_t1_path = _resolve_example_image_path(concept_dir, frame_t1_rel)
    else:
        frame_t_path, frame_t1_path = _fallback_frame_paths(concept_dir, example)

    if not frame_t_path.exists() or not frame_t1_path.exists():
        frame_t_path, frame_t1_path = _fallback_frame_paths(concept_dir, example)

    if not frame_t_path.exists() or not frame_t1_path.exists():
        print(
            f"WARNING: missing frame_t/frame_t1 images for rank {example.rank} "
            f"({example.video_name}): {frame_t_path}, {frame_t1_path}"
        )
        return None

    try:
        frame_t_img = Image.open(frame_t_path).convert("RGB")
        frame_t1_img = Image.open(frame_t1_path).convert("RGB")
        frame_t_img = frame_t_img.resize(image_size, Image.Resampling.BICUBIC)
        frame_t1_img = frame_t1_img.resize(image_size, Image.Resampling.BICUBIC)

        frame_t = np.asarray(frame_t_img, dtype=np.float32) / 255.0
        frame_t1 = np.asarray(frame_t1_img, dtype=np.float32) / 255.0
        return frame_t, frame_t1
    except (OSError, ValueError) as exc:
        print(
            f"WARNING: failed to load frame_t/frame_t1 for rank {example.rank} "
            f"({example.video_name}): {exc}"
        )
        return None


def compute_patch_difference_feature(
    frame_t: np.ndarray,
    frame_t1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a 2D difference image and flattened clustering feature.

    Difference image:
    - abs(frame_t1 - frame_t)
    - average over RGB channels to make a 2D array [H,W]
    - normalize to [0,1] if max > 0

    Feature:
    - flatten the 2D difference image into a 1D vector
    - L2 normalize to avoid magnitude-only clustering
    """
    diff = np.abs(frame_t1.astype(np.float32) - frame_t.astype(np.float32))
    diff_2d = diff.mean(axis=-1)

    max_val = float(diff_2d.max())
    if max_val > 0.0:
        diff_2d = diff_2d / max_val

    feature = diff_2d.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm > 0.0:
        feature = feature / norm

    return diff_2d, feature


def save_difference_image(diff_2d: np.ndarray, output_path: Path) -> None:
    """Save the normalized 2D difference array as a grayscale PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(diff_2d, 0.0, 1.0)
    gray = (clipped * 255.0).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(output_path)


def choose_num_clusters(features: np.ndarray, max_clusters: int = 4) -> int:
    """
    Choose a reasonable number of clusters based on the number of examples.
    """
    n_examples = int(features.shape[0])
    if n_examples < 4:
        return 1

    upper_k = min(max_clusters, n_examples - 1)
    if upper_k < 2:
        return 1

    best_k = min(2, n_examples)
    best_score = float("-inf")
    silhouette_failed = True

    for k in range(2, upper_k + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(features)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(features, labels))
            silhouette_failed = False
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    if silhouette_failed:
        return min(2, n_examples)
    return best_k


def cluster_examples_by_patch_difference(
    concept_dir: Path,
    examples: List["SavedExample"],
    max_clusters: int = 4,
    image_size: Tuple[int, int] = (64, 64),
    save_diff_images: bool = True,
) -> Dict[int, List["SavedExample"]]:
    """
    Cluster examples for one concept based on frame_t -> frame_t1 patch difference images.
    """
    diff_dir = concept_dir / "top_examples" / "diff_images"
    valid_examples: List["SavedExample"] = []
    valid_features: List[np.ndarray] = []
    failed_examples: List["SavedExample"] = []
    diff_by_rank: Dict[int, np.ndarray] = {}

    for example in sorted(examples, key=lambda item: item.rank):
        pair_arrays = load_patch_pair_arrays(concept_dir, example, image_size=image_size)
        if pair_arrays is None:
            failed_examples.append(example)
            continue

        frame_t, frame_t1 = pair_arrays
        diff_2d, feature = compute_patch_difference_feature(frame_t, frame_t1)
        diff_by_rank[int(example.rank)] = diff_2d

        if save_diff_images:
            diff_path = diff_dir / f"example_{example.rank:03d}_diff.png"
            save_difference_image(diff_2d, diff_path)

        valid_examples.append(example)
        valid_features.append(feature)

    if len(valid_features) < 2:
        clustered: Dict[int, List["SavedExample"]] = {
            0: sorted(examples, key=lambda item: item.rank)
        }
        save_clustered_top_examples(
            concept_dir,
            clustered,
            diff_by_rank=diff_by_rank,
            save_diff_images=save_diff_images,
        )
        return clustered

    feature_matrix = np.stack(valid_features, axis=0).astype(np.float32)
    num_clusters = choose_num_clusters(feature_matrix, max_clusters=max_clusters)
    labels = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init="auto",
    ).fit_predict(feature_matrix)

    clustered = {cluster_id: [] for cluster_id in range(num_clusters)}
    for example, label in zip(valid_examples, labels.tolist()):
        clustered[int(label)].append(example)

    if failed_examples:
        clustered.setdefault(0, [])
        clustered[0].extend(failed_examples)

    for cluster_id in clustered:
        clustered[cluster_id].sort(key=lambda item: item.rank)

    save_clustered_top_examples(
        concept_dir,
        clustered,
        diff_by_rank=diff_by_rank,
        save_diff_images=save_diff_images,
    )
    return clustered


def load_clip_model(
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[torch.device] = None,
) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """
    Load CLIP image encoder and processor.
    """
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()
    clip_model.to(device)
    return clip_model, clip_processor, device


def _extract_clip_image_embeddings(
    clip_model: CLIPModel,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """Return a [batch, dim] tensor from CLIP image features across transformers versions."""
    image_features = clip_model.get_image_features(pixel_values=pixel_values)
    if isinstance(image_features, torch.Tensor):
        return image_features
    if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
        return image_features.pooler_output
    raise TypeError(f"Unexpected CLIP image feature type: {type(image_features)!r}")


def _resolve_frame_t_patch_path(concept_dir: Path, example: "SavedExample") -> Optional[Path]:
    """Resolve the small cropped patch at frame t."""
    frame_t_rel = getattr(example, "frame_t_path", "") or ""
    if frame_t_rel:
        candidate = _resolve_example_image_path(concept_dir, frame_t_rel)
        if candidate.exists():
            return candidate

    fallback = concept_dir / "top_examples" / f"example_{example.rank:03d}_frame_t.png"
    if fallback.exists():
        return fallback
    return None


def _detect_red_box_bounds(arr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    mask = (red > 180) & (green < 80) & (blue < 80)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _expand_box_bounds(
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    img_w: int,
    img_h: int,
    crop_scale: float,
) -> Tuple[int, int, int, int]:
    box_w = max_x - min_x + 1
    box_h = max_y - min_y + 1
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    half_w = box_w * crop_scale / 2.0
    half_h = box_h * crop_scale / 2.0
    x0 = max(0, int(round(center_x - half_w)))
    y0 = max(0, int(round(center_y - half_h)))
    x1 = min(img_w - 1, int(round(center_x + half_w)))
    y1 = min(img_h - 1, int(round(center_y + half_h)))
    return x0, y0, x1, y1


def _save_resized_clip_crop(
    image: Image.Image,
    output_path: Path,
    output_size: Tuple[int, int],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized = image.resize(output_size, Image.Resampling.BICUBIC)
    resized.save(output_path)
    return output_path


def create_large_clip_crop_from_full_frame(
    concept_dir: Path,
    example: "SavedExample",
    crop_scale: float = 3.0,
    output_size: Tuple[int, int] = (224, 224),
) -> Optional[Path]:
    """
    Create an enlarged crop around the activated patch from full frame t.

    Save it under:
      concept_dir/top_examples/clip_crops/example_{rank:03d}_large_crop_t.png

    Return the saved crop path.
    """
    output_path = (
        concept_dir
        / "top_examples"
        / "clip_crops"
        / f"example_{example.rank:03d}_large_crop_t.png"
    )
    if output_path.exists():
        return output_path

    full_t_path = _example_source_path(
        concept_dir,
        example,
        "full_t_boxed_path",
        "full_t_boxed",
    )
    if full_t_path is not None:
        try:
            full_img = Image.open(full_t_path).convert("RGB")
            bounds = _detect_red_box_bounds(np.asarray(full_img))
            if bounds is not None:
                min_x, min_y, max_x, max_y = bounds
                img_w, img_h = full_img.size
                x0, y0, x1, y1 = _expand_box_bounds(
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    img_w,
                    img_h,
                    crop_scale,
                )
                crop = full_img.crop((x0, y0, x1 + 1, y1 + 1))
                return _save_resized_clip_crop(crop, output_path, output_size)
        except (OSError, ValueError) as exc:
            print(
                f"WARNING: failed to create large CLIP crop from full frame for "
                f"rank {example.rank}: {exc}"
            )

    frame_t_path = _resolve_frame_t_patch_path(concept_dir, example)
    if frame_t_path is None:
        print(
            f"WARNING: missing frame_t image for large CLIP crop fallback: "
            f"rank={example.rank}"
        )
        return None

    try:
        patch_img = Image.open(frame_t_path).convert("RGB")
        return _save_resized_clip_crop(patch_img, output_path, output_size)
    except (OSError, ValueError) as exc:
        print(
            f"WARNING: failed frame_t fallback for large CLIP crop rank "
            f"{example.rank}: {exc}"
        )
        return None


def resolve_clip_image_path(
    concept_dir: Path,
    example: "SavedExample",
    clip_image_source: str = "large_crop_t",
    clip_crop_scale: float = 3.0,
) -> Optional[Path]:
    """
    Choose the image used for CLIP visual embedding.
    """
    if clip_image_source == "large_crop_t":
        return create_large_clip_crop_from_full_frame(
            concept_dir,
            example,
            crop_scale=clip_crop_scale,
        )

    frame_t_path = _resolve_frame_t_patch_path(concept_dir, example)
    if frame_t_path is None:
        print(f"WARNING: missing frame_t image for CLIP clustering: rank={example.rank}")
    return frame_t_path


def compute_clip_features_for_examples(
    concept_dir: Path,
    examples: List["SavedExample"],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    batch_size: int = 16,
    clip_image_source: str = "large_crop_t",
    clip_crop_scale: float = 3.0,
) -> Tuple[List["SavedExample"], np.ndarray, List["SavedExample"], Dict[int, Dict[str, Any]]]:
    """
    Compute CLIP visual embeddings for examples.
    """
    valid_examples: List["SavedExample"] = []
    failed_examples: List["SavedExample"] = []
    pending_images: List[Image.Image] = []
    clip_metadata_by_rank: Dict[int, Dict[str, Any]] = {}

    for example in sorted(examples, key=lambda item: item.rank):
        image_path = resolve_clip_image_path(
            concept_dir,
            example,
            clip_image_source=clip_image_source,
            clip_crop_scale=clip_crop_scale,
        )
        if image_path is None:
            failed_examples.append(example)
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, ValueError) as exc:
            print(
                f"WARNING: failed to load CLIP image for rank {example.rank} "
                f"({example.video_name}): {exc}"
            )
            failed_examples.append(example)
            continue

        clip_metadata: Dict[str, Any] = {"clip_image_source": clip_image_source}
        if clip_image_source == "large_crop_t":
            clip_metadata["clip_crop_scale"] = clip_crop_scale
            clip_metadata["clip_crop_path"] = _relative_to_concept_dir(concept_dir, image_path)
        clip_metadata_by_rank[int(example.rank)] = clip_metadata

        valid_examples.append(example)
        pending_images.append(image)

    if not pending_images:
        return [], np.empty((0, 0), dtype=np.float32), failed_examples, clip_metadata_by_rank

    feature_batches: List[np.ndarray] = []
    for start in range(0, len(pending_images), batch_size):
        batch_images = pending_images[start : start + batch_size]
        inputs = clip_processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features = _extract_clip_image_embeddings(
                clip_model,
                inputs["pixel_values"],
            )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feature_batches.append(image_features.detach().cpu().numpy().astype(np.float32))

    feature_matrix = np.concatenate(feature_batches, axis=0)
    return valid_examples, feature_matrix, failed_examples, clip_metadata_by_rank


def choose_num_clusters_from_features(features: np.ndarray, max_clusters: int = 4) -> int:
    """
    Choose k automatically using silhouette score.
    """
    n_examples = int(features.shape[0])
    if n_examples < 4:
        return 1

    upper_k = min(max_clusters, n_examples - 1)
    if upper_k < 2:
        return 1

    best_k = 1
    best_score = float("-inf")
    silhouette_failed = True

    for k in range(2, upper_k + 1):
        try:
            labels = AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="average",
            ).fit_predict(features)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(features, labels, metric="cosine"))
            silhouette_failed = False
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    if silhouette_failed:
        return 1
    return best_k


def cluster_examples_by_clip_features(
    concept_dir: Path,
    examples: List["SavedExample"],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
    max_clusters: int = 4,
    clip_batch_size: int = 16,
    model_name: str = "openai/clip-vit-base-patch32",
    clip_image_source: str = "large_crop_t",
    clip_crop_scale: float = 3.0,
) -> Dict[int, List["SavedExample"]]:
    """
    Cluster examples based on CLIP visual embeddings.
    """
    (
        valid_examples,
        feature_matrix,
        failed_examples,
        clip_metadata_by_rank,
    ) = compute_clip_features_for_examples(
        concept_dir=concept_dir,
        examples=examples,
        clip_model=clip_model,
        clip_processor=clip_processor,
        device=device,
        batch_size=clip_batch_size,
        clip_image_source=clip_image_source,
        clip_crop_scale=clip_crop_scale,
    )

    cluster_metadata_extra = {
        "clustering_feature": "clip_visual_embedding",
        "clip_model": model_name,
        "clip_image_source": clip_image_source,
        "clip_crop_scale": clip_crop_scale,
        "cluster_algorithm": "AgglomerativeClustering cosine average linkage",
    }

    if len(valid_examples) < 2:
        clustered: Dict[int, List["SavedExample"]] = {
            0: sorted(examples, key=lambda item: item.rank)
        }
        save_clustered_top_examples(
            concept_dir,
            clustered,
            diff_by_rank={},
            save_diff_images=False,
            cluster_metadata_extra=cluster_metadata_extra,
            clip_metadata_by_rank=clip_metadata_by_rank,
        )
        return clustered

    num_clusters = choose_num_clusters_from_features(feature_matrix, max_clusters=max_clusters)
    if num_clusters <= 1:
        clustered = {0: sorted(examples, key=lambda item: item.rank)}
        save_clustered_top_examples(
            concept_dir,
            clustered,
            diff_by_rank={},
            save_diff_images=False,
            cluster_metadata_extra=cluster_metadata_extra,
            clip_metadata_by_rank=clip_metadata_by_rank,
        )
        return clustered

    labels = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric="cosine",
        linkage="average",
    ).fit_predict(feature_matrix)

    clustered = {cluster_id: [] for cluster_id in range(num_clusters)}
    for example, label in zip(valid_examples, labels.tolist()):
        clustered[int(label)].append(example)

    if failed_examples:
        clustered.setdefault(0, [])
        clustered[0].extend(failed_examples)

    for cluster_id in clustered:
        clustered[cluster_id].sort(key=lambda item: item.rank)

    save_clustered_top_examples(
        concept_dir,
        clustered,
        diff_by_rank={},
        save_diff_images=False,
        cluster_metadata_extra=cluster_metadata_extra,
        clip_metadata_by_rank=clip_metadata_by_rank,
    )
    return clustered
