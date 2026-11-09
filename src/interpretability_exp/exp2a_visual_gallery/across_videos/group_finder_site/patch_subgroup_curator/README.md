# Patchwork — prototype subgroup curator

A local browser interface for the gallery produced by your supplied exporter.
The application is one self-contained Python file. No pip installation, Node,
model loading, GPU, or internet connection is required.

## Run

Requires Python 3.9+ on Linux or macOS (Windows: run the server in WSL).
Run on the machine that can access the gallery:

```bash
python3 patch_subgroup_curator.py \
  --root /data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v4
```

Open **http://127.0.0.1:8765** in your browser. Change `concepts_v4` to the actual
export directory, for example `concepts_v3`, if needed.

If the gallery is on a remote Linux server, run the command there, then run this
in a terminal on your own computer, replacing `USER@SERVER`:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Then open the same localhost URL on your computer. No files need to be copied
from the gallery to your computer.

Optional flags:

```bash
python3 patch_subgroup_curator.py --root /path/to/concepts_v4 \
  --output-dir /path/to/manual_subgroups --port 8765
```

Use an output directory **outside the source gallery**. The supplied gallery
exporter's overwrite mode deletes existing group contents; the default sibling
output directory keeps manual work separate from those exports.

## Expected source layout

```text
concepts_v4/
  group_03/
    patch_metadata.json
    patch_00/
      target_frame_boxed.png
      target_frame.png
      patch.png
      temporal_window.gif
    patch_01/
      ...
  group_04/
    ...
```

`patch_directory` is read from each metadata record. The exporter naming
convention is also supported as a fallback for relocated galleries.
`group_patch_index` is the patch identity; patch 0 is fully selectable.
Group folders are discovered dynamically and sorted numerically; no limit of
eight source groups is imposed.

## Use

1. Choose a source-group tab, or use the arrows on either side of the grid.
2. Inspect each boxed frame **above** its patch. Click either image for an
   enlarged preview; play the temporal GIF if it exists.
3. Toggle the circular control beside a patch's ID. These are independent,
   accessible checkboxes styled as circles, allowing multiple selections and
   deselection. Standard radio buttons would prevent that behavior.
4. The gallery shows **8 columns × up to 8 rows per page (64 items)**. Use the
   pagination buttons for additional patches. Selection survives page and tab
   changes. Draft selections and descriptions also survive a browser reload in
   the same tab, provided the metadata is unchanged.
5. Set **Top-k**, initially 20. The sidebar displays:
   - IDs shared by **every** selected patch, in the first selected patch's rank order;
   - per-patch cosine values for shared IDs;
   - prototype frequency across the selection;
   - each patch's ranked IDs and cosines;
   - pairwise intersections.
6. Add an optional name and visual-pattern description, then click **Form group**.
7. After a successful disk save, the selected patches become disabled and the
   selection clears. They remain disabled after page reloads and app restarts.
   Continue selecting other patches to form another subgroup of the source group.
8. Download any saved JSON from the sidebar. Use **Hide saved** to focus on
   remaining candidates.

The same exported patch record cannot enter two subgroups of one source group.
Locks are scoped to `(source group, group_patch_index)`; independently exported
records in other source groups remain selectable.

## Saved JSON

Default destination:

```text
concepts_v4_manual_subgroups/
  group_03/
    subgroup_0001.json
    subgroup_0002.json
  group_04/
    subgroup_0001.json
```

Every subgroup file contains:

- Source group, subgroup ID, optional name/description, creation time, requested k.
- Selected patch IDs in selection order and distinct-video count.
- Complete, unchanged original metadata for each selected patch in
  `patches[i].metadata`, including all originally stored match and activation fields.
- Selected cosine-ranked matches in `patches[i].selected_top_k_cosine_matches`.
- Shared-by-all IDs, prototype support counts/membership, per-patch cosine scores,
  and pairwise overlaps (including Jaccard similarity).
- Source metadata SHA-256 so saved selections can be traced to the exact export.

For example, read a saved subgroup in Python:

```python
import json
from pathlib import Path

record = json.loads(Path("/path/to/group_03/subgroup_0001.json").read_text())
print(record["patch_ids"])
print(record["shared_by_all_prototype_ids"])
for patch in record["patches"]:
    print(patch["metadata"]["group_patch_index"],
          patch["selected_top_k_cosine_matches"])
```

Saves are atomic. The server rechecks locks and metadata before committing, so
two browser tabs cannot place the same exported patch in different subgroups.
Only one server process may write to a particular output directory at a time.
If the gallery is regenerated, use a fresh output directory: patch numbering
could have changed. Existing saved files are kept intact.

## Interpretation and limits

The ranking deliberately follows your supplied `top_matches` function:
descending `cosine_similarity` from `top_k_cosine_matches`. The supplied export
code distinguishes these matches from actual model forward activations. The
interface preserves that distinction, and retains `actual_valid_active`,
`forward_activation`, and assignment values wherever present in the metadata.
It cannot recover prototype ranks or activations that were not exported.
When k exceeds the stored matches, it uses the available matches and shows a
warning, exactly as in your reporting code.

All-selected and pairwise intersections are included; exponential powerset
enumeration is intentionally not used. Saving a visually coherent subgroup
with no shared prototype IDs is allowed. A single selected patch can be saved,
but its IDs are labeled as that patch's matches rather than a multi-patch overlap.

The app does not train a model, recompute feature similarities, automatically
choose subgroups, or change any source gallery file. Subgroups save references
and metadata, not copies of images. Keep the source gallery available.

## Verification

The accompanying standard-library tests exercise overlap ranking, stored-k
limits, saves/reloads, lock conflicts, changed metadata, and HTTP endpoints:

```bash
python3 -m unittest discover -s tests -v
```
