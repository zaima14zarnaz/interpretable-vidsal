from pathlib import Path
import json
from explanation_generation import SavedExample
from cluster_concepts import cluster_examples_by_patch_difference

concept_dir = Path(
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/concepts/c_tr_000"
)

metadata_path = concept_dir / "top_examples" / "examples_metadata.json"

with metadata_path.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

examples = []

for item in metadata:
    examples.append(
        SavedExample(
            concept_idx=int(item.get("global_concept_index", 0)),
            rank=int(item["rank"]),
            score=float(item.get("retrieval_score", item.get("score", 0.0))),
            video_name=str(item.get("video_filename", "")),
            batch_index=int(item.get("batch_index", -1)),
            patch_index=int(item.get("sample_index", item.get("patch_index", -1))),
            grid_hw=None,
            image_path=str(item.get("pair_path", "")),
            frame_t_path=str(item.get("frame_t_path", "")),
            frame_t1_path=str(item.get("frame_t1_path", "")),
            pair_path=str(item.get("pair_path", "")),
            full_t_boxed_path=str(item.get("full_t_boxed_path", "")),
            full_t1_boxed_path=str(item.get("full_t1_boxed_path", "")),
            context_panel_path=str(item.get("context_panel_path", "")),
            activation_score=float(item.get("activation_score", 0.0)),
            patch_pred_saliency=float(item.get("patch_pred_saliency", 0.0)),
            retrieval_score=float(item.get("retrieval_score", item.get("score", 0.0))),
            is_salient_region=bool(item.get("is_salient_region", False)),
        )
    )

clusters = cluster_examples_by_patch_difference(
    concept_dir=concept_dir,
    examples=examples,
    max_clusters=4,
    image_size=(64, 64),
    save_diff_images=True,
)

print("Clusters:")
for cluster_id, cluster_examples in clusters.items():
    print(f"cluster {cluster_id}: {len(cluster_examples)} examples")