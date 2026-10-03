# Stage-4 similar-patch experiment

This experiment starts with similar patches and asks whether they share highly
ranked prototypes. It does not start with prototypes and retrieve their patches.
No model weights are changed and no extra neural backbone is used.

## Install and run

Place `stage4_similar_patch_groups.py` beside your current
`visual_concept_explanations.py` in the project. The latter supplies the model
architecture, checkpoint loader, dataset loader, collate function, and RGB
preprocessing. Alternatively, pass its full path with `--gallery-script`.
Keep its model architecture arguments consistent with the trained checkpoint.
The new script rejects missing/mismatched model parameter tensors instead of
silently analyzing a partly initialized model. Missing old-checkpoint role
buffers alone do not prevent inference.

```bash
python patch_to_prototype.py \
  --checkpoint /home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/20261001_040032/epoch_068.pt \
  --dataset-dir /data/quantization/zaima/videosal_datasets/dhf1k/train \
  --output-dir /data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/cross_video/concepts_v2 \
  --num-groups 4 \
  --min-group-size 4 \
  --max-group-size 8 \
  --top-k 10 \
  --min-stage4-cosine 0.90 \
  --max-rgb-mae 0.10 \
  --min-rgb-structure-cosine 0.90 \
  --max-windows 512 \
  --device cuda:0
```

Dependencies: your existing project dependencies plus PyTorch, NumPy, and
Pillow. No scikit-learn, additional image encoder, or LLM is required. Use CPU
with `--device cpu`. Pair comparisons default to CPU; optionally use
`--similarity-device cuda:0` if memory permits.

## What defines a group

Capture the **stage-4 feature grid fed into the concept module**, before
`visual_encoder` and prototype matching. This includes any configured stage-4
grid resize. Use its last feature time step, aligned with the target RGB frame.

Randomly sample dataset windows with seed 42 and retain a bounded candidate
pool. Default limits: 512 windows, 12 patches per window, 64 accepted candidates
per video, and 4,000 candidates total. Each selected window is identified by its
original dataset index and video ID. One final target frame is used per window;
earlier frames are saved as temporal context rather than treated as independent
concept-assignment frames.

**Every pair** of patches in a group must pass all these checks:

| Check | Default | Purpose |
|---|---:|---|
| Stage-4 feature cosine | >= 0.90 | Similarity before the concept encoder |
| RGB mean absolute difference | <= 0.10 | Similar colors and spatial appearance |
| Centered RGB structural cosine | >= 0.90 | Similar spatial variation rather than only similar average color |
| Video identities | Different | Avoid near-duplicate frames of one video |

RGB comparisons use unpadded feature-cell crops from the actual backbone-resized
target frame, resampled to 16×16. Structural cosine uses per-channel spatially
centered RGB values. Mostly uniform crops are excluded by default using mean
per-channel spatial standard deviation >= 0.025. Set
`--min-crop-spatial-std 0` if you intentionally want flat/background patches.
Completely zero structural descriptors still cannot establish structural
similarity and are excluded.

A patch joins a group only if it passes the thresholds against **every existing
member**. Similarity to a group center or to just one member is insufficient.
The selection uses a bounded greedy clique search, so it is a practical search,
not proof that all possible groups were found. Different groups have disjoint
patches; this does not guarantee that they represent different semantic concepts.

These are strict **appearance proxies**, not a guarantee of shared semantic
identity. They may reject semantically similar objects with different poses,
lighting, or backgrounds. Inspect the contact sheets and temporal windows.

The script requests four groups of 4–8 patches, each from a different video.
**If fewer groups meet the criteria, it reports fewer groups. It never lowers
the thresholds automatically.** More windows/candidates or another seed may
help. You can deliberately change thresholds for a separate, clearly labeled
run. Use a new or empty output directory for each run.

## Prototype measurements

Only after group formation, inspect each patch's current concept embedding and
compare it with the normalized stage-4 prototype bank. Group selection never
uses prototype identities, ranks, probabilities, or saliency filtering.

For **every prototype**, not only the top ten, report:

| Field | Meaning |
|---|---|
| `cosine_similarity` | Normalized patch embedding versus prototype; -1 to 1 |
| `rank` | Rank within the full bank by cosine; 1 is the highest |
| `scaled_logit` | The model's exported assignment logit, including its temperature scaling |
| `soft_assignment_probability` | Exported softmax probability across the full prototype bank |
| `forward_activation` | Actual `visual_activations` value used in the forward pass |
| `actual_valid_active` | Membership from exported prototype indices and `visual_validity_mask` |
| `in_top8`, `in_top10`, `in_requested_top_k` | Membership among the nearest cosine matches |

With hard/straight-through assignments, a prototype ranked second can have a
large cosine similarity but **zero forward activation**. This experiment keeps
that distinction visible. It tests shared similarity preferences; it does not
change the decoder to activate eight or ten prototypes.

Per-group summaries show each prototype's top-1/top-8/top-10 support, rank,
cosine range, soft probability, and actual active support. They list prototypes
shared by all members in the requested top-k. A separate flag checks whether
all those members also exceed `--prototype-high-cosine` (default 0.70). This
threshold is an experimental convention, not a calibrated probability or
universal definition of a strong match.

The summary also reports top-k frequency across the candidate pool. A prototype
appearing in almost every patch's top-k is weaker evidence of a distinctive
group-specific concept. `topk_overlap_baseline.json` compares within-group
top-k overlap with random pairs from different videos outside the same group.
This is descriptive; pair observations are dependent and no statistical
significance claim is made.

## Saved output

Open `index.html` for a visual overview.

```text
stage4_groups_run1/
  index.html
  experiment_summary.json
  patch_prototype_activations.csv
  topk_overlap_baseline.json
  candidate_features_and_scores.pt
  candidate_metadata.json
  group_00/
    contact_sheet.png
    frame_gallery.png
    activation_heatmap.svg
    pairwise_similarity.csv
    prototype_summary.csv
    patch_metadata.json
    patch_00/
      patch.png
      target_frame.png
      target_frame_native.png
      target_frame_boxed.png
      temporal_window.pt
      temporal_window.gif
      window_frames/frame_000.png
      ...
  group_01/
  ...
```

`temporal_window.pt` stores the full backbone-resized RGB window as float32
`[T,3,H,W]`, before ImageNet normalization. PNGs and the GIF provide visual
previews; GIF playback uses an arbitrary 100 ms per frame and does not claim to
recover the source video's timing. Full windows are saved only for selected
group members, so large unselected windows do not accumulate in memory.

Metadata records video/window identity, original dataset index, target offset,
grid row/column, grid dimensions, and boxes in both model-input and native RGB
coordinates. The existing six-field loader does not expose absolute source
frame numbers, so these are explicitly recorded as unavailable rather than
inferred. The dataset index and saved window establish the sampled instance.
Boxes identify nominal feature-grid cells, not the full receptive field.

Selected windows are reloaded from the dataset for export, and their normalized
native RGB hashes must match the extraction pass. If random augmentation changes
them, export fails rather than silently saving the wrong source window. Use
deterministic evaluation preprocessing for this experiment. Model-input crops
must use the same geometric transformation as the backbone; the adapter currently
assumes full-frame spatial resizing, as in the supplied gallery script.

## How to interpret a positive result

If visually coherent group members share a prototype at ranks 2–8 or 2–10,
despite different top-1 winners, that supports your proposed explanation for
why a winner-only gallery can miss shared similarity structure. Check absolute
cosines, soft probabilities, and candidate-pool frequency as well as rank.

It does not establish that the shared prototype contributes to the current
prediction when its forward activation is zero. It also does not prove semantic
interpretability or causal influence. Stage-4 features come from the trained
model and can themselves contain contextual/temporal information; the independent
RGB gates reduce, but do not eliminate, this ambiguity.

## Validation

15 focused CPU tests passed using synthetic data and lightweight project
substitutes. They cover all-pairs grouping, cross-video membership, appearance
gates, independence from prototype scores, a shared rank-2 prototype with
different top-1 winners, bounded sampling, actual-activation reporting, and
frame/window/coordinate export with source-hash verification. The real custom
backbone/checkpoint, CUDA path, and full DHF1K run remain to be validated on your
machine.
