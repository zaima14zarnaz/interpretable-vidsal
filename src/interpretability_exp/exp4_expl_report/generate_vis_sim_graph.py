#!/usr/bin/env python3
"""Exact visual patch similarity graph, aligned to the prototype graph's node IDs.

Lives beside generate_prot_overlap_graph.py under exp4_expl_report/; finds
src/evaluation.py via SRC_DIR. First finish the prototype graph extraction,
then run this file:

    python generate_vis_sim_graph.py

Edit ACTION and other constants below; no arguments are required. The three
requested flags are also accepted:

    python generate_visual_similarity_graph.py --min-stage4-cosine 0.20 \
        --max-rgb-mae 0.6 --min-rgb-structure-cosine 0.20

Actions:
  extract       Cache descriptors for EVERY stage4 node in the existing graph,
                using the same checkpoint/windows. Generate offline HTML.
                Checkpoints allow interrupted extraction to resume safely.
  view          Regenerate the offline HTML without inference.
  serve         Explore any node, including bounded 1–3-hop neighborhoods,
                at http://127.0.0.1:8766. No inference required after extraction.
  build_edges   OPTIONAL: enumerate all qualifying undirected pairs into SQLite.
                Exact all-pairs construction can be expensive (quadratic time).
  export_edges  Export a completed build_edges database as CSV.
  selftest      Test metrics and graph retrieval without PyTorch or dataset.

An edge exists iff ALL THREE gates pass (inclusive):
    stage4_cosine >= 0.40 AND rgb_mae <= 0.18
    AND rgb_structure_cosine >= 0.50.
No automatic threshold relaxation, top-k neighbor truncation, extra variance
filter, prototype-overlap filter, or approximation is used to define the graph.
Every edge has all three scores; its 'weight' is stage4_cosine, not an invented
combination of scores. Zero-norm structural descriptors have similarity zero.

Metric definitions match the earlier patch-retrieval approach:
  stage4: L2-normalized last-time-step concept-module INPUT, before visual_encoder
          and before prototype matching (normally 768 dimensions).
  RGB MAE: mean absolute difference between [0,1] RGB crops resized to 16x16
           using bilinear interpolation, align_corners=False, in CHW order.
  structure: subtract each RGB channel's spatial mean from that crop descriptor,
             flatten and L2-normalize; similarity is the resulting dot product.
RGB is taken from the live loader tensors and resized like the backbone, NOT
from the JPEG thumbnails. Saliency and prototype evidence are copied by node ID
from the prototype graph; they are never used to determine visual edges.

The scalable default stores the graph IMPLICITLY as memory-mapped float32
descriptors plus exact threshold rules. A query scans every cached node in
blocks. All qualifying edges can be recovered; no global NxN matrix is kept.
Only the interactive DISPLAY is capped. Optional build_edges materializes the
full edge set. All stage4 nodes, including isolates, remain in the cache.

The source prototype database is opened read-only. Its completed manifest and
checkpoint hashes are verified. Existing integer node IDs are never renumbered.
If it includes other stages, this graph contains its stage4 subset only.

Files under /data/.../dhf1k/explanation_graphs/visual_graph_outputs/:
  manifest.json, progress.json, node_ids.npy, stage4_features.npy,
  rgb_descriptors.npy, graph_viewer.html, neighborhood.json.
Optional: edges.sqlite, edges_manifest.json, edges.csv.
Metadata/images continue to reference prototype_graph_outputs in that same
explanation_graphs directory. Keep it available. Full-window replay requires the original dataset,
evaluation.py, checkpoint, and project environment. Browsing afterward needs
only NumPy, Pillow, this script, the companion script, and the two output folders.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import closing
import csv
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
ACTION = "extract"  # extract | view | serve | build_edges | export_edges | selftest
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[1]  # .../src containing evaluation.py
PROTOTYPE_SCRIPT = SCRIPT_DIR / "generate_prot_overlap_graph.py"
GRAPH_ROOT = Path("/data/quantization/zaima/videosal_datasets/dhf1k/explanation_graphs")
PROTOTYPE_GRAPH_DIR = GRAPH_ROOT / "prototype_graph_outputs"
OUTPUT_DIR = GRAPH_ROOT / "visual_graph_outputs"
MIN_STAGE4_COSINE = 0.20
MAX_RGB_MAE = 0.6
MIN_RGB_STRUCTURE_COSINE = 0.2
RGB_DESCRIPTOR_SIZE = 16
GPU_ID = 0
NUM_WORKERS = 0  # replay is one full window per batch
RESUME = True
FLUSH_EVERY_WINDOWS = 25
QUERY_BLOCK_SIZE = 4096
RGB_METRIC_CHUNK = 1024
VIEW_SEED_NODE = None
VIEW_MAX_NODES = 120
VIEW_HOPS = 1
VIEW_CROSS_VIDEO_ONLY = True  # viewer filter; full graph includes same-video edges
VIEW_REQUIRE_DIRECT_ANCHOR_MATCH = True  # useful for the later reporting protocol
GALLERY_TOP_N = 50  # selected patch + up to 49 strongest visual matches
SERVER_PORT = 8766


def load_companion():
    if not Path(PROTOTYPE_SCRIPT).is_file():
        raise FileNotFoundError(f"Missing companion prototype-graph script: {PROTOTYPE_SCRIPT}")
    evaluation_file = SRC_DIR / "evaluation.py"
    if not evaluation_file.is_file():
        raise FileNotFoundError(f"Missing evaluation.py under SRC_DIR. Not found: {evaluation_file}")
    spec = importlib.util.spec_from_file_location("_visual_graph_companion", PROTOTYPE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SRC_DIR = SRC_DIR
    module.EVALUATION_FILE = evaluation_file
    return module


def normalize_rows(a):
    a = np.asarray(a, dtype=np.float32)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    return np.divide(a, np.maximum(norms, np.float32(1e-12)))


def rgb_structure(rgb):
    a = np.asarray(rgb, dtype=np.float32)
    single = a.ndim == 1
    if single:
        a = a[None]
    if a.ndim != 2 or a.shape[1] % 3:
        raise ValueError("RGB descriptors must have shape [N,3*H*W] in CHW order.")
    channels = a.reshape(len(a), 3, -1)
    # Prevent mean-reduction roundoff in an exactly constant channel from
    # becoming a spurious unit-length structure. Nonconstant channels are kept.
    centered = channels - channels.mean(axis=-1, keepdims=True)
    centered = np.where(np.ptp(channels, axis=-1, keepdims=True) == 0, 0, centered)
    result = normalize_rows(centered.reshape(len(a), -1))
    return result[0] if single else result


def validate_thresholds(thresholds):
    f, m, s = thresholds
    if not (-1 <= f <= 1 and 0 <= m <= 1 and -1 <= s <= 1):
        raise ValueError("Cosine thresholds must be in [-1,1]; RGB MAE in [0,1].")
    return tuple(map(float, thresholds))


def default_thresholds():
    return validate_thresholds((MIN_STAGE4_COSINE, MAX_RGB_MAE, MIN_RGB_STRUCTURE_COSINE))


def pair_metrics(feature_a, feature_b, rgb_a, rgb_b):
    fa, fb = normalize_rows(feature_a), normalize_rows(feature_b)
    ra, rb = np.asarray(rgb_a, np.float32), np.asarray(rgb_b, np.float32)
    return {"stage4_cosine": float(np.clip(fa @ fb, -1, 1)),
            "rgb_mae": float(np.mean(np.abs(ra-rb))),
            "rgb_structure_cosine": float(np.clip(rgb_structure(ra) @ rgb_structure(rb), -1, 1))}


def passes(scores, thresholds):
    f, m, s = validate_thresholds(thresholds)
    return scores["stage4_cosine"] >= f and scores["rgb_mae"] <= m and scores["rgb_structure_cosine"] >= s


def metadata_signature(companion):
    companion.require_complete(PROTOTYPE_GRAPH_DIR)
    path = Path(PROTOTYPE_GRAPH_DIR) / "manifest.json"
    manifest = json.loads(path.read_text())
    return manifest, companion.sha256_file(path)


def torch_rgb_descriptors(frame, grid_h, grid_w, size):
    """frame is [3,H,W] float RGB, resized to backbone coordinates."""
    import torch
    import torch.nn.functional as F
    c, h, w = frame.shape
    if c != 3 or h < grid_h or w < grid_w:
        raise ValueError("RGB frame is smaller than its nominal patch grid.")
    if h % grid_h == 0 and w % grid_w == 0:
        ph, pw = h//grid_h, w//grid_w
        crops = frame.reshape(3,grid_h,ph,grid_w,pw).permute(1,3,0,2,4).reshape(-1,3,ph,pw)
        descriptors = F.interpolate(crops.float(), (size,size), mode="bilinear", align_corners=False)
    else:
        crops = []
        for r in range(grid_h):
            for col in range(grid_w):
                x0,x1 = round(col*w/grid_w),round((col+1)*w/grid_w)
                y0,y1 = round(r*h/grid_h),round((r+1)*h/grid_h)
                crop = frame[:,y0:y1,x0:x1]
                crops.append(F.interpolate(crop[None].float(), (size,size), mode="bilinear", align_corners=False)[0])
        descriptors = torch.stack(crops)
    return descriptors.flatten(1).cpu().numpy().astype(np.float32, copy=False)


def extract_descriptors(companion):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    outdir = Path(OUTPUT_DIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    source_manifest, signature = metadata_signature(companion)
    manifest_path = outdir / "manifest.json"
    progress_path = outdir / "progress.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing:
        if not RESUME:
            raise FileExistsError("Output exists and RESUME=False. Choose a new OUTPUT_DIR.")
        if existing["source_manifest_sha256"] != signature or existing["rgb_descriptor_size"] != RGB_DESCRIPTOR_SIZE:
            raise RuntimeError("Source graph or descriptor settings changed. Choose a new OUTPUT_DIR.")
        if existing.get("status") == "complete":
            print("Descriptor cache is already complete; regenerating viewer without inference.", flush=True)
            with VisualGraph(companion) as graph:
                write_viewer(graph)
            return
    ev = companion.load_evaluation()
    eval_hash = companion.sha256_file(ev.__file__)
    if eval_hash != source_manifest["evaluation_sha256"]:
        raise RuntimeError("evaluation.py changed since prototype extraction. Restore that evaluation "
                           "configuration or regenerate the prototype graph before aligning descriptors.")
    checkpoint = source_manifest["checkpoint"]
    if companion.sha256_file(checkpoint) != source_manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint bytes differ from those used for the prototype graph.")
    if torch.cuda.is_available():
        if not 0 <= GPU_ID < torch.cuda.device_count():
            raise ValueError(f"GPU_ID={GPU_ID} is unavailable.")
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")
    ev.train_cfg.set_seed(int(source_manifest["seed"]))
    model = ev.build_model(device,device)
    ev.load_checkpoint(model,checkpoint)
    model.eval()
    if "stage4" not in model.concept_creations:
        raise RuntimeError("Model has no stage4 concept module.")
    amp_enabled = bool(source_manifest.get("amp_enabled",False))
    dtype_name = str(source_manifest.get("amp_dtype") or "torch.float16").split(".")[-1]
    amp_dtype = getattr(torch,dtype_name)
    captured = {}

    def capture_input(module,args):
        tensor = args[0]
        if tensor.ndim != 5 or tensor.shape[0] != 1:
            raise RuntimeError(f"Expected stage4 input [1,C,T,H,W], got {tuple(tensor.shape)}.")
        last = tensor.detach()[0,:,-1].float().permute(1,2,0).contiguous()
        captured["shape"] = tuple(last.shape[:2])
        captured["features"] = normalize_rows(last.reshape(-1,last.shape[-1]).cpu().numpy())

    hook = model.concept_creations["stage4"].register_forward_pre_hook(capture_input)
    source = companion.connect_db(Path(PROTOTYPE_GRAPH_DIR)/"graph.sqlite", readonly=True)
    ids = np.fromiter((r[0] for r in source.execute("SELECT id FROM nodes WHERE stage='stage4' ORDER BY id")),dtype=np.int64)
    if not len(ids):
        hook.remove();source.close()
        raise ValueError("The prototype graph has no stage4 nodes.")
    if existing:
        if not np.array_equal(np.load(outdir/"node_ids.npy"),ids):
            raise RuntimeError("Source stage4 node IDs changed.")
    else:
        np.save(outdir/"node_ids.npy",ids,allow_pickle=False)
    frames = source.execute("""SELECT DISTINCT f.* FROM frames f JOIN nodes n ON n.frame_id=f.id
        WHERE n.stage='stage4' ORDER BY f.id""").fetchall()
    progress = json.loads(progress_path.read_text()) if existing and progress_path.exists() else {"completed_frames":0}
    done = int(progress["completed_frames"])
    feature_cache = rgb_cache = None
    if existing and (outdir/"stage4_features.npy").exists():
        feature_cache = np.load(outdir/"stage4_features.npy",mmap_mode="r+")
        rgb_cache = np.load(outdir/"rgb_descriptors.npy",mmap_mode="r+")
        if feature_cache.shape[0] != len(ids) or rgb_cache.shape != (len(ids),3*RGB_DESCRIPTOR_SIZE**2):
            raise RuntimeError("Descriptor cache shapes disagree with the source graph.")
    elif done:
        raise RuntimeError("Progress exists but descriptor files are missing.")
    manifest = existing or {"created_at":companion.now(),"source_manifest_sha256":signature,
        "prototype_graph_dir":str(Path(PROTOTYPE_GRAPH_DIR).resolve()),
        "source_checkpoint_sha256":source_manifest["checkpoint_sha256"],
        "rgb_descriptor_size":RGB_DESCRIPTOR_SIZE,"dtype":"float32", "num_nodes":len(ids),
        "num_frames":len(frames),"stage":"stage4","descriptor_version":1,
        "feature_space":"L2-normalized last-time-step concept input before visual_encoder",
        "structure":"per-channel mean-centered normalized RGB; exactly constant channels have zero structure",
        "graph_representation":"exact descriptors and all-three-threshold rule; edges queried lazily",
        "node_ids":"identical to corresponding prototype graph stage4 nodes"}
    manifest.update(status="building",last_started_at=companion.now(),thresholds=list(default_thresholds()))
    companion.dump_json(manifest_path,manifest)
    print(f"Replaying {len(frames)-done:,} remaining windows; {len(ids):,} aligned stage4 nodes.",flush=True)
    datasets = {}
    try:
        # Group consecutive frames by dataset root while preserving source frame order.
        cursor = done
        progress = tqdm(total=len(frames), initial=done, desc="windows", unit="window", leave=True)
        try:
            while cursor < len(frames):
                label = frames[cursor]["dataset"]
                end = cursor
                while end < len(frames) and frames[end]["dataset"] == label:
                    end += 1
                root = source_manifest["dataset_roots"][label]
                if label not in datasets:
                    datasets[label] = ev.DatasetLoader(root, window_len=int(source_manifest["window_len"]),
                                                      stride=int(source_manifest["stride"]))
                dataset = datasets[label]
                selected = frames[cursor:end]
                indices = [int(f["dataset_index"]) for f in selected]
                kwargs = {"num_workers":NUM_WORKERS,"collate_fn":ev.video_saliency_collate_fn,
                          "pin_memory":device.type=="cuda"}
                if NUM_WORKERS>0:
                    kwargs.update(persistent_workers=True,prefetch_factor=2)
                loader = DataLoader(Subset(dataset,indices),batch_size=1,shuffle=False,**kwargs)
                for record,batch in zip(selected,loader):
                    names,rgb,sal,lengths,valid = companion.unpack_batch(batch)
                    window_len = int(source_manifest["window_len"])
                    if len(names)!=1 or str(names[0])!=record["video"] or int(lengths[0])!=window_len or not bool(valid.all()):
                        raise RuntimeError("Dataset replay differs from the source window metadata.")
                    di = int(record["dataset_index"])
                    video,start = dataset.windows[di]
                    fnames = companion.frame_names_for_video(dataset,root,str(video))
                    expected = json.loads(record["window_names"])
                    if str(video)!=record["video"] or int(start)!=record["window_start"] or fnames[int(start):int(start)+window_len]!=expected:
                        raise RuntimeError("Frame names/order changed since prototype extraction.")
                    captured.clear()
                    with torch.inference_mode(),torch.autocast(device.type,dtype=amp_dtype,enabled=amp_enabled):
                        prediction = model(rgb.to(device),return_details=False,return_concept_losses=False,
                                           return_decoder_diagnostics=False)
                    if "features" not in captured:
                        raise RuntimeError("stage4 concept pre-hook was never called.")
                    h,w = captured["shape"]
                    rows = source.execute("SELECT id,row,col,grid_h,grid_w FROM nodes "
                        "WHERE frame_id=? AND stage='stage4' ORDER BY row,col",(record["id"],)).fetchall()
                    if len(rows)!=h*w or any((r["grid_h"],r["grid_w"],r["row"],r["col"])!=(h,w,j//w,j%w)
                                             for j,r in enumerate(rows)):
                        raise RuntimeError("stage4 grid no longer agrees with source node IDs.")
                    if rgb.shape[2]==3:
                        frame=rgb[0,-1].detach().float().cpu()
                    elif rgb.shape[-1]==3:
                        frame=rgb[0,-1].detach().float().cpu().permute(2,0,1)
                    else:
                        raise RuntimeError(f"Unsupported RGB layout: {tuple(rgb.shape)}")
                    if tuple(frame.shape[-2:]) != (record["input_h"],record["input_w"]):
                        raise RuntimeError("Loader RGB dimensions changed since prototype extraction.")
                    if frame.max()>2:
                        frame/=255
                    if not torch.isfinite(frame).all() or frame.min() < -0.01 or frame.max()>1.01:
                        raise ValueError("Expected unnormalized RGB in [0,1] or [0,255].")
                    resize = getattr(model.backbone,"resize_to",None)
                    if resize is not None:
                        size=(resize,resize) if isinstance(resize,int) else tuple(resize)
                        frame=F.interpolate(frame[None],size=size,mode="bilinear",align_corners=False)[0]
                    features=captured["features"]
                    descriptors=torch_rgb_descriptors(frame,h,w,RGB_DESCRIPTOR_SIZE)
                    if not np.isfinite(features).all() or not np.isfinite(descriptors).all():
                        raise ValueError("Non-finite descriptors.")
                    if feature_cache is None:
                        feature_cache=np.lib.format.open_memmap(outdir/"stage4_features.npy",mode="w+",
                            dtype=np.float32,shape=(len(ids),features.shape[1]))
                        rgb_cache=np.lib.format.open_memmap(outdir/"rgb_descriptors.npy",mode="w+",
                            dtype=np.float32,shape=(len(ids),descriptors.shape[1]))
                        manifest["feature_dim"]=features.shape[1]
                        companion.dump_json(manifest_path,manifest)
                        gb=(feature_cache.nbytes+rgb_cache.nbytes)/1024**3
                        print(f"Descriptor cache size: {gb:.2f} GiB, float32, disk-backed.",flush=True)
                    if features.shape[1]!=feature_cache.shape[1]:
                        raise RuntimeError("Stage4 feature dimension changed during extraction.")
                    positions=np.searchsorted(ids,np.array([r["id"] for r in rows],dtype=np.int64))
                    feature_cache[positions]=features
                    rgb_cache[positions]=descriptors
                    cursor+=1
                    progress.update(1)
                    if cursor%FLUSH_EVERY_WINDOWS==0 or cursor==len(frames):
                        feature_cache.flush();rgb_cache.flush()
                        companion.dump_json(progress_path,{"completed_frames":cursor,"updated_at":companion.now()})
                    del prediction,frame,descriptors,features,batch,rgb,sal
        finally:
            progress.close()
        manifest.update(status="complete",completed_at=companion.now())
        companion.dump_json(manifest_path,manifest)
    except BaseException as exc:
        manifest.update(status="interrupted",error=str(exc),updated_at=companion.now())
        companion.dump_json(manifest_path,manifest)
        raise
    finally:
        hook.remove();source.close()
        del feature_cache,rgb_cache
    with VisualGraph(companion) as graph:
        write_viewer(graph)


class VisualGraph:
    def __init__(self,companion,directory=None,source_directory=None):
        self.helper=companion
        self.directory=Path(directory or OUTPUT_DIR).resolve()
        self.source_dir=Path(source_directory or PROTOTYPE_GRAPH_DIR).resolve()
        self.manifest=json.loads((self.directory/"manifest.json").read_text())
        if self.manifest.get("status")!="complete":
            raise RuntimeError("Descriptor extraction is incomplete. Run ACTION='extract' to resume.")
        signature=companion.sha256_file(self.source_dir/"manifest.json")
        if signature!=self.manifest["source_manifest_sha256"]:
            raise RuntimeError("Prototype graph manifest changed; the visual cache may no longer align.")
        self.ids=np.load(self.directory/"node_ids.npy",mmap_mode="r")
        self.features=np.load(self.directory/"stage4_features.npy",mmap_mode="r")
        self.rgb=np.load(self.directory/"rgb_descriptors.npy",mmap_mode="r")
        if len(self.ids)!=len(self.features) or len(self.ids)!=len(self.rgb):
            raise RuntimeError("Mismatched descriptor-array lengths.")
        self.con=companion.connect_db(self.source_dir/"graph.sqlite",readonly=True)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.con.close()

    def position(self,node_id):
        p=int(np.searchsorted(self.ids,int(node_id)))
        if p>=len(self.ids) or int(self.ids[p])!=int(node_id):
            raise ValueError(f"Node {node_id} is not in this stage4 graph.")
        return p

    def node(self,node_id):
        self.position(node_id)
        return self.helper.get_node(self.con,node_id)

    def scores(self,a,b):
        ia,ib=self.position(a),self.position(b)
        return pair_metrics(self.features[ia],self.features[ib],self.rgb[ia],self.rgb[ib])

    def neighbors(self,node_id,thresholds=None,limit=None,cross_video=False):
        """Exact all-node scan. A return limit affects retrieval only, not graph definition."""
        thresholds=default_thresholds() if thresholds is None else validate_thresholds(thresholds)
        min_feature,max_mae,min_structure=thresholds
        pos=self.position(node_id)
        f=np.asarray(self.features[pos],np.float32)
        rgb=np.asarray(self.rgb[pos],np.float32)
        structure=rgb_structure(rgb)
        anchor=self.node(node_id) if cross_video else None
        results=[]
        for start in range(0,len(self.ids),QUERY_BLOCK_SIZE):
            end=min(start+QUERY_BLOCK_SIZE,len(self.ids))
            # L2-normalized features were persisted in float32.
            fcos=np.clip(np.asarray(self.features[start:end])@f,-1,1)
            candidates=np.flatnonzero(fcos>=min_feature)
            candidates=candidates[candidates+start!=pos]
            for off in range(0,len(candidates),RGB_METRIC_CHUNK):
                local=candidates[off:off+RGB_METRIC_CHUNK]
                positions=local+start
                rd=np.asarray(self.rgb[positions],np.float32)
                scos=np.clip(rgb_structure(rd)@structure,-1,1)
                mae=np.mean(np.abs(rd-rgb),axis=1)
                keep=(scos>=min_structure)&(mae<=max_mae)
                for j in np.flatnonzero(keep):
                    nid=int(self.ids[positions[j]])
                    if cross_video:
                        other=self.helper.get_node(self.con,nid)
                        if (other["dataset"],other["video"])==(anchor["dataset"],anchor["video"]):
                            continue
                    results.append({"node_id":nid,"stage4_cosine":float(fcos[local[j]]),
                                    "rgb_mae":float(mae[j]),"rgb_structure_cosine":float(scos[j])})
            # Keep bounded result storage without discarding any candidate before scoring.
            if limit is not None and len(results)>max(int(limit)*2,1):
                results.sort(key=sort_key)
                results=results[:int(limit)]
        results.sort(key=sort_key)
        return results if limit is None else results[:max(0,int(limit))]

    def payload(self,seed=None,thresholds=None,limit=VIEW_MAX_NODES,hops=VIEW_HOPS,
                cross_video=VIEW_CROSS_VIDEO_ONLY,direct=VIEW_REQUIRE_DIRECT_ANCHOR_MATCH):
        thresholds=default_thresholds() if thresholds is None else validate_thresholds(thresholds)
        seed=int(self.ids[0]) if seed is None else int(seed)
        self.position(seed)
        limit=max(1,min(int(limit),500));hops=max(1,min(int(hops),3))
        selected=[seed];seen={seed};queue=deque([(seed,0)]);capped=False
        while queue:
            current,depth=queue.popleft()
            if depth>=hops:
                continue
            # Bound candidate expansion. Direct-anchor filtering can leave fewer
            # than `limit` nodes; this is exploration, not exhaustive k-hop search.
            candidates=self.neighbors(current,thresholds,limit=2*limit+1,cross_video=cross_video)
            for hit in candidates:
                nid=hit["node_id"]
                if nid in seen:
                    continue
                if direct and not passes(self.scores(seed,nid),thresholds):
                    continue
                if cross_video:
                    a,b=self.node(seed),self.node(nid)
                    if (a["dataset"],a["video"])==(b["dataset"],b["video"]):
                        continue
                if len(selected)>=limit:
                    capped=True
                    break
                seen.add(nid);selected.append(nid);queue.append((nid,depth+1))
            if capped:
                break
        nodes=[self.node(nid) for nid in selected]
        edges=[]
        for j,a in enumerate(selected):
            for b in selected[j+1:]:
                scores=self.scores(a,b)
                if passes(scores,thresholds):
                    edges.append({"source":a,"target":b,"weight":scores["stage4_cosine"],**scores})
        self.helper.add_preview_images(nodes,self.source_dir)
        for node in nodes:
            node["seed_metrics"]=self.scores(seed,node["id"])
        return {"nodes":nodes,"edges":edges,"seed":seed,"total_nodes":len(self.ids),
                "thresholds":list(thresholds),"display_limit":limit,"hops":hops,
                "cross_video":bool(cross_video),"require_direct_match":bool(direct),
                "capped":capped,"scope":"Bounded exploration; not an exhaustive k-hop set. "
                "All displayed edges pass all three gates. Multi-hop reachability alone "
                "does not establish direct similarity to the seed. Layout distance is schematic."}

    def make_patch_gallery(self, node_id, thresholds=None, cross_video=False,
                           limit=None):
        """Selected node first, then up to GALLERY_TOP_N-1 strongest visual matches.

        Queries the full descriptor cache independently of the displayed neighborhood.
        Ranking uses the same three-gate filters and sort_key as neighbors().
        """
        thresholds = default_thresholds() if thresholds is None else validate_thresholds(thresholds)
        limit = max(0, int(GALLERY_TOP_N - 1 if limit is None else limit))
        anchor = self.node(int(node_id))
        ranked = self.neighbors(int(node_id), thresholds=thresholds, limit=limit,
                                cross_video=bool(cross_video))
        anchor.update(gallery_selected=True, edge_strength=None,
                      stage4_cosine=None, rgb_mae=None, rgb_structure_cosine=None)
        patches = [anchor]
        for hit in ranked:
            node = self.node(hit["node_id"])
            node.update(gallery_selected=False,
                        edge_strength=float(hit["stage4_cosine"]),
                        stage4_cosine=float(hit["stage4_cosine"]),
                        rgb_mae=float(hit["rgb_mae"]),
                        rgb_structure_cosine=float(hit["rgb_structure_cosine"]))
            patches.append(node)
        self.helper.add_preview_images(patches, self.source_dir)
        return {"selected_node": int(node_id), "patches": patches,
                "thresholds": list(thresholds), "cross_video": bool(cross_video),
                "grid_columns": 10, "grid_rows": 5, "total_slots": int(GALLERY_TOP_N),
                "num_neighbors": len(ranked), "scope": "complete_database",
                "ranking": "stage4_cosine descending; then lower rgb_mae; then "
                           "higher structure cosine; node ID ascending for ties"}


def sort_key(hit):
    return (-hit["stage4_cosine"],hit["rgb_mae"],-hit["rgb_structure_cosine"],hit["node_id"])


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual patch similarity graph</title><style>
*{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;background:#f4f6fa;color:#19283c}header{padding:20px 24px;background:white;border-bottom:1px solid #d5deea}h1{font-size:22px;margin:0 0 7px}p{line-height:1.5;margin:8px 0}.muted{font-size:13px;color:#5c6b7f}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end;margin-top:15px}label{display:grid;gap:5px;font-size:12px}input,select,button{padding:8px;border:1px solid #bbc8d8;border-radius:6px;background:white;color:#19283c}input[type=number]{width:100px}button{cursor:pointer}.primary{background:#244f7e;color:white}#status{padding:12px 24px}.shell{display:grid;grid-template-columns:minmax(0,1fr) 350px;background:white}#scene{height:65vh;min-height:420px}canvas{width:100%;height:100%;touch-action:none}aside{padding:18px;border-left:1px solid #d5deea;max-height:75vh;overflow:auto}.frame{position:relative;margin:10px 0}.frame img{width:100%;display:block}.box{position:absolute;border:3px solid #e76e33}.crop{max-width:150px;max-height:110px}.pill{display:inline-block;padding:4px 7px;margin:2px;border-radius:5px;background:#e7eef7;font-size:12px}.legend{padding:12px 24px;font-size:13px}.gradient{display:inline-block;width:140px;height:10px;background:linear-gradient(90deg,#427ca8,#e29e58,#d15a45);margin:0 8px}#rows{max-height:260px;overflow:auto;margin:12px 24px}table{width:100%;border-collapse:collapse;font-size:12px}td,th{text-align:left;padding:8px;border-bottom:1px solid #d5deea}tr[data-id]{cursor:pointer}tr[data-id]:hover{background:#edf3fb}code{font-size:12px;word-break:break-all}@media(max-width:850px){.shell{grid-template-columns:1fr}aside{max-height:500px;border-left:0}}
.patch-gallery{width:min(1500px,97vw);max-width:97vw;max-height:92vh;padding:0;border:1px solid #bbc8d8;border-radius:12px;color:#19283c;background:#f7f9fc;box-shadow:0 20px 80px #0005}
.patch-gallery::backdrop{background:rgba(15,25,42,.65)}
.gallery-header{padding:16px 20px;background:white;border-bottom:1px solid #d5deea;position:sticky;top:0;z-index:2}
.gallery-heading{display:flex;justify-content:space-between;align-items:center;gap:16px}.gallery-heading h2{font-size:19px;margin:0}
.gallery-scroll{overflow:auto;padding:14px}.patch-grid{display:grid;grid-template-columns:repeat(10,minmax(104px,1fr));grid-template-rows:repeat(5,auto);gap:10px;list-style:none;padding:0;margin:0;min-width:1130px}
.gallery-card{min-width:0;border:1px solid #d8e0eb;border-radius:8px;background:white;overflow:hidden}.gallery-card.is-selected{border:3px solid #e76e33;background:#fff5ec}
.gallery-card button{display:block;width:100%;height:100%;padding:7px;border:0;border-radius:0;background:transparent;text-align:left;color:inherit}
.gallery-card button:hover{background:#eef4fc}.gallery-card button:focus-visible{outline:3px solid #244f7e;outline-offset:-3px}
.gallery-rank{display:block;min-height:20px;font-size:11px;font-weight:700;color:#244f7e}.is-selected .gallery-rank{color:#a94416}
.gallery-crop{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#e9edf3;border-radius:4px}
.gallery-no-image{display:flex;align-items:center;justify-content:center;font-size:12px;color:#657287;text-align:center}
.gallery-node{display:block;font-weight:650;font-size:12px;margin-top:6px}.gallery-info{display:block;font-size:10px;color:#5c6b7f;line-height:1.4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.gallery-empty{min-height:150px;border:1px dashed #d0d8e5;border-radius:8px;background:#eef2f7}.gallery-note{font-size:12px;color:#5c6b7f;margin-bottom:0}
</style></head><body><header><h1>Visual patch similarity graph</h1><p class="muted">Each edge passes all three thresholds. Patch IDs match the prototype graph. Edge weight is stage-4 cosine; all three metrics are available on inspection. Select a node and open Top 50 patches for a ranked visual-match gallery.</p>
<div class="controls"><label>Minimum stage-4 cosine<input id="feature" type="number" min="-1" max="1" step=".01"></label><label>Maximum RGB MAE<input id="mae" type="number" min="0" max="1" step=".01"></label><label>Minimum RGB structure cosine<input id="structure" type="number" min="-1" max="1" step=".01"></label><label>Seed node<input id="seed" type="number" min="0"></label><label>Max displayed nodes<input id="limit" type="number" min="1" max="500"></label><label>Hops<select id="hops"><option>1</option><option>2</option><option>3</option></select></label><label>Neighbors<select id="cross"><option value="1">Other videos</option><option value="0">Any video</option></select></label><label>Direct match to seed<select id="direct"><option value="1">Required</option><option value="0">Allow transitive matches</option></select></label><button id="load" class="primary">Load neighborhood</button><button id="reset">Reset layout</button></div>
<div class="controls"><label>Search video / frame / node ID<input id="query"></label><button id="search">Search nodes</button><label>Edges<select id="edgeMode"><option value="selected">Touching selected node</option><option value="all">All visible</option><option value="none">Hidden</option></select></label><label>Edge color<select id="edgeColor" title="Colors existing visible edges only; does not change edge visibility."><option value="uniform">Original</option><option value="prototype">Prototype overlap (top-k)</option></select></label><span id="prototypeEdgeLegend" class="muted" hidden title="Jaccard overlap of unique top-k candidate IDs: shared IDs / union of IDs. Missing or empty evidence is gray.">Prototype overlap: 0% <span style="display:inline-block;width:100px;height:10px;background:linear-gradient(90deg,rgb(37,99,235),rgb(220,38,38));vertical-align:middle;margin:0 5px"></span> 100% <span style="color:#6b7280">· gray = unavailable</span></span><span class="muted" id="mode"></span></div></header><div id="status"></div><div class="shell"><div id="scene"><canvas id="canvas"></canvas></div><aside id="details"></aside></div><div class="legend">Within-frame predicted score percentile: low<span class="gradient"></span>high. Scroll to zoom; drag nodes/background. Layout proximity is not a similarity measurement.</div><div id="rows"></div>
<dialog id="patchGallery" class="patch-gallery" aria-labelledby="galleryTitle" aria-describedby="gallerySummary">
<div class="gallery-header"><div class="gallery-heading"><h2 id="galleryTitle">Top 50 patches</h2><button id="closeGallery" type="button" aria-label="Close patch gallery">Close</button></div><p id="gallerySummary" class="gallery-note" role="status" aria-live="polite"></p></div>
<div class="gallery-scroll"><ol id="galleryGrid" class="patch-grid" aria-label="Selected patch followed by strongest visual matches"></ol></div>
</dialog><script>
const INITIAL=__PAYLOAD__;
const ONLINE=false;
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data=INITIAL,nodes=[],edges=[],byId=new Map(),selected=null,zoom=1,panX=0,panY=0,drag=null,hover=null,settle=0;
const canvas=$('canvas'),ctx=canvas.getContext('2d');let width=800,height=550;
$('mode').textContent=ONLINE?'Local explorer: exact queries over all cached nodes.':'Offline snapshot: use ACTION="serve" to search the complete graph.';
const fmt=x=>Number(x).toFixed(4),color=p=>p<.5?`rgb(${66+Math.round(p*320)},${124+Math.round(p*68)},${168-Math.round(p*160)})`:`rgb(${226-Math.round((p-.5)*34)},${158-Math.round((p-.5)*136)},${88-Math.round((p-.5)*38)})`;

// Display metadata only: never used by retrieval, filters, weights or layout.
function prototypeIdSet(node){
  return Array.isArray(node.topk_ids)
    ? new Set(node.topk_ids.filter(id=>id!=null).map(String)) : null;
}
function prototypeOverlap(a,b){
  if(!a||!b)return null;
  let shared=0;for(const id of a)if(b.has(id))shared++;
  const union=a.size+b.size-shared;
  return union ? shared/union : null;
}
function edgeStroke(edge,byOverlap){
  if(!byOverlap)return 'rgba(60,90,130,.25)';
  const overlap=edge.prototypeOverlap;
  if(overlap==null||!Number.isFinite(overlap))return 'rgba(107,114,128,.65)';
  const t=Math.max(0,Math.min(1,overlap));
  return `rgba(${Math.round(37+183*t)},${Math.round(99-61*t)},${Math.round(235-197*t)},.85)`;
}
$('edgeColor').onchange=()=>{
  $('prototypeEdgeLegend').hidden=$('edgeColor').value!=='prototype';
};

function loadData(d){data=d;['seed','hops'].forEach(k=>$(k).value=d[k]);$('limit').value=d.display_limit;$('feature').value=d.thresholds[0];$('mae').value=d.thresholds[1];$('structure').value=d.thresholds[2];$('cross').value=d.cross_video?'1':'0';$('direct').value=d.require_direct_match?'1':'0';
nodes=d.nodes.map((n,i)=>({...n,x:Math.cos(i*2.399)*Math.sqrt(i)*16,y:Math.sin(i*2.399)*Math.sqrt(i)*16,vx:0,vy:0}));byId=new Map(nodes.map(n=>[n.id,n]));const prototypeSets=new Map(nodes.map(n=>[n.id,prototypeIdSet(n)]));edges=d.edges.map(e=>({...e,a:byId.get(e.source),b:byId.get(e.target),prototypeOverlap:prototypeOverlap(prototypeSets.get(e.source),prototypeSets.get(e.target))}));selected=byId.get(d.seed)||nodes[0];zoom=1;panX=panY=0;settle=160;
$('status').textContent=`${nodes.length} of ${d.total_nodes.toLocaleString()} nodes shown · ${edges.length} displayed edges${d.capped?' · display capped':''}. ${d.scope}`;details();table(nodes);}
function details(){const n=selected;if(!n)return;const ns=edges.filter(e=>e.a===n||e.b===n).sort((a,b)=>b.stage4_cosine-a.stage4_cosine).slice(0,16);const sm=n.seed_metrics;
$('details').innerHTML=`<h2 style="font-size:18px;margin-top:0">Patch ${n.id}</h2><p>${esc(n.dataset)} / ${esc(n.video)} / ${esc(n.target_name)}</p><p class="muted">Stage 4 · cell ${n.row},${n.col} · grid ${n.grid_h}×${n.grid_w}<br>Window ${n.window_start}…${n.target_index} (zero-based)</p>${n.crop_image?`<img class="crop" alt="Patch crop" src="${n.crop_image}">`:''}${n.frame_image?`<div class="frame"><img alt="Full input frame" src="${n.frame_image}"><div class="box" style="left:${100*n.col/n.grid_w}%;top:${100*n.row/n.grid_h}%;width:${100/n.grid_w}%;height:${100/n.grid_h}%"></div></div>`:''}<p>Predicted score: <b>${fmt(n.pred_score)}</b><br>Within-frame percentile: <b>${(100*n.pred_percentile).toFixed(1)}%</b>${n.gt_percentile==null?'':`<br>GT percentile: ${(100*n.gt_percentile).toFixed(1)}%`}</p>${sm?`<p><b>Direct metrics against seed</b><br>Stage-4 cosine: ${fmt(sm.stage4_cosine)}<br>RGB MAE: ${fmt(sm.rgb_mae)}<br>Structure cosine: ${fmt(sm.rgb_structure_cosine)}</p>`:''}<p><b>Prototype candidates</b></p>${n.topk_ids.map(p=>`<span class="pill">${p}</span>`).join('')}<p><b>Forward-valid prototypes</b></p>${n.active_ids.map(p=>`<span class="pill">${p}</span>`).join('')||'None'}<p class="muted">Prototypes are metadata only; they do not determine visual edges.</p><button id="recenter">Use this patch as seed</button> <button id="topPatches" type="button">${ONLINE?'Top 50 patches':'Top 50 in snapshot'}</button><p><b>Displayed visual neighbors</b></p>${ns.map(e=>{const o=e.a===n?e.b:e.a;return `<p><button data-select="${o.id}">Node ${o.id}</button><br>Feature ${fmt(e.stage4_cosine)} · MAE ${fmt(e.rgb_mae)} · structure ${fmt(e.rgb_structure_cosine)}</p>`;}).join('')}<p class="muted"><code>${esc(n.source_path)}</code></p>`;
$('recenter').onclick=()=>{$('seed').value=n.id;fetchGraph();};
$('topPatches').onclick=()=>openPatchGallery(n);
$('details').querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>{selected=byId.get(Number(b.dataset.select));details();});}
let galleryRequest=0,galleryReturnFocus=null;
const galleryDialog=$('patchGallery');
function closePatchGallery(){galleryRequest++;if(galleryDialog.open)galleryDialog.close();if(galleryReturnFocus&&galleryReturnFocus.isConnected)galleryReturnFocus.focus();}
$('closeGallery').onclick=closePatchGallery;
galleryDialog.addEventListener('cancel',()=>{galleryRequest++;});
galleryDialog.addEventListener('click',event=>{if(event.target===galleryDialog){const r=galleryDialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)closePatchGallery();}});
function rankSnapshotGallery(anchor,feature,mae,structure,cross){
 const ranked=INITIAL.edges.filter(e=>(e.source===anchor.id||e.target===anchor.id)&&e.stage4_cosine>=feature&&e.rgb_mae<=mae&&e.rgb_structure_cosine>=structure)
  .map(e=>{const id=e.source===anchor.id?e.target:e.source,n=INITIAL.nodes.find(x=>x.id===id);return n?{...n,gallery_selected:false,edge_strength:e.stage4_cosine,stage4_cosine:e.stage4_cosine,rgb_mae:e.rgb_mae,rgb_structure_cosine:e.rgb_structure_cosine}:null;})
  .filter(n=>n&&(!cross||n.dataset!==anchor.dataset||n.video!==anchor.video))
  .sort((a,b)=>b.stage4_cosine-a.stage4_cosine||a.rgb_mae-b.rgb_mae||b.rgb_structure_cosine-a.rgb_structure_cosine||a.id-b.id).slice(0,49);
 return{selected_node:anchor.id,patches:[{...anchor,gallery_selected:true,edge_strength:null,stage4_cosine:null,rgb_mae:null,rgb_structure_cosine:null},...ranked],thresholds:[feature,mae,structure],cross_video:cross,scope:'offline_snapshot',num_neighbors:ranked.length,total_slots:50};
}
function renderPatchGallery(result){const scope=result.scope==='complete_database'?'Ranked across the complete descriptor cache.':'Ranked only within the original offline snapshot; use ACTION="serve" for complete-graph ranking.';
 $('galleryTitle').textContent=`Top 50 patches · selected node ${result.selected_node}`;
 $('gallerySummary').textContent=`Selected patch first, followed by ${result.num_neighbors} neighbors ranked by stage-4 cosine (ties: lower RGB MAE, then higher structure cosine, then node ID). Thresholds feature≥${Number(result.thresholds[0]).toFixed(2)}, MAE≤${Number(result.thresholds[1]).toFixed(2)}, structure≥${Number(result.thresholds[2]).toFixed(2)}; ${result.cross_video?'other videos only':'any video'}. ${scope} ${result.patches.length<50?'Empty cells indicate fewer matching patches.':''}`;
 const cards=result.patches.map((n,i)=>`<li class="gallery-card${i===0?' is-selected':''}"><button type="button" data-gallery-node="${n.id}" title="${esc(n.dataset+' / '+n.video+' / '+n.target_name)}"><span class="gallery-rank">${i===0?'Selected patch':`#${i} · cos ${fmt(n.stage4_cosine)} · MAE ${fmt(n.rgb_mae)}`}</span>${n.crop_image?`<img class="gallery-crop" loading="lazy" src="${n.crop_image}" alt="Patch ${n.id}, ${esc(n.video)}, cell ${n.row},${n.col}">`:'<span class="gallery-crop gallery-no-image">No thumbnail</span>'}<span class="gallery-node">Node ${n.id}</span><span class="gallery-info">${esc(n.video)} / ${esc(n.target_name)}</span><span class="gallery-info">cell ${n.row},${n.col} · pred ${(100*n.pred_percentile).toFixed(1)}%</span>${i===0?'':`<span class="gallery-info">structure ${fmt(n.rgb_structure_cosine)}</span>`}</button></li>`);
 while(cards.length<50)cards.push('<li class="gallery-empty" aria-hidden="true"></li>');
 $('galleryGrid').innerHTML=cards.join('');
 $('galleryGrid').querySelectorAll('[data-gallery-node]').forEach(button=>button.onclick=()=>{const id=Number(button.dataset.galleryNode);closePatchGallery();if(byId.has(id)){selected=byId.get(id);details();}else{$('seed').value=id;fetchGraph();}});}
async function openPatchGallery(anchor){const request=++galleryRequest;galleryReturnFocus=document.activeElement;$('galleryTitle').textContent=`Top 50 patches · selected node ${anchor.id}`;$('gallerySummary').textContent='Loading strongest visual matches…';$('galleryGrid').innerHTML='';if(!galleryDialog.open)galleryDialog.showModal();
 try{const feature=Number($('feature').value),mae=Number($('mae').value),structure=Number($('structure').value),cross=$('cross').value==='1';let result;
 if(ONLINE){const q=new URLSearchParams({node:anchor.id,feature,mae,structure,cross:Number(cross)});const response=await fetch('/api/top-patches?'+q);result=await response.json();if(!response.ok)throw Error(result.error||'Could not load patch gallery.');}
 else result=rankSnapshotGallery(anchor,feature,mae,structure,cross);
 if(request===galleryRequest&&galleryDialog.open)renderPatchGallery(result);
 }catch(error){if(request===galleryRequest&&galleryDialog.open)$('gallerySummary').textContent=error.message;}}
function table(list){$('rows').innerHTML=`<table><thead><tr><th>Node</th><th>Video / frame</th><th>Cell</th><th>Predicted percentile</th><th>Prototype candidates</th></tr></thead><tbody>${list.slice(0,200).map(n=>`<tr data-id="${n.id}"><td>${n.id}</td><td>${esc(n.video)} / ${esc(n.target_name)}</td><td>${n.row},${n.col}</td><td>${(100*n.pred_percentile).toFixed(1)}%</td><td>${n.topk_ids.join(', ')}</td></tr>`).join('')}</tbody></table>`;$('rows').querySelectorAll('[data-id]').forEach(r=>r.onclick=()=>{const id=Number(r.dataset.id);if(byId.has(id)){selected=byId.get(id);details();}else{$('seed').value=id;fetchGraph();}});}
async function fetchGraph(){try{const q={seed:Number($('seed').value),feature:Number($('feature').value),mae:Number($('mae').value),structure:Number($('structure').value),limit:Number($('limit').value),hops:Number($('hops').value),cross:$('cross').value,direct:$('direct').value};
if(ONLINE){$('status').textContent='Scanning cached descriptors against all nodes…';const r=await fetch('/api/graph?'+new URLSearchParams(q));const d=await r.json();if(!r.ok)throw Error(d.error);loadData(d);}
else{const t=INITIAL.thresholds;if(q.feature<t[0]||q.mae>t[1]||q.structure<t[2])throw Error('Looser thresholds require ACTION="serve"; this snapshot only contains edges passing its original thresholds.');const all=INITIAL.nodes,root=all.find(n=>n.id===q.seed);if(!root)throw Error('Seed is outside the snapshot. Use ACTION="serve".');const valid=INITIAL.edges.filter(e=>e.stage4_cosine>=q.feature&&e.rgb_mae<=q.mae&&e.rgb_structure_cosine>=q.structure);const seen=new Set([q.seed]),queue=[[q.seed,0]];while(queue.length&&seen.size<q.limit){const [cur,dep]=queue.shift();if(dep>=q.hops)continue;const ee=valid.filter(e=>e.source===cur||e.target===cur).sort((a,b)=>b.stage4_cosine-a.stage4_cosine);for(const e of ee){const id=e.source===cur?e.target:e.source,n=all.find(x=>x.id===id);if(seen.has(id))continue;if(q.cross==='1'&&n.dataset===root.dataset&&n.video===root.video)continue;if(q.direct==='1'&&!valid.some(z=>(z.source===q.seed&&z.target===id)||(z.target===q.seed&&z.source===id)))continue;seen.add(id);queue.push([id,dep+1]);if(seen.size>=q.limit)break;}}loadData({...INITIAL,nodes:all.filter(n=>seen.has(n.id)).map(n=>({...n,seed_metrics:q.seed===INITIAL.seed?n.seed_metrics:INITIAL.edges.find(e=>(e.source===q.seed&&e.target===n.id)||(e.target===q.seed&&e.source===n.id))||null})),edges:valid.filter(e=>seen.has(e.source)&&seen.has(e.target)),seed:q.seed,thresholds:[q.feature,q.mae,q.structure],display_limit:q.limit,hops:q.hops,cross_video:q.cross==='1',require_direct_match:q.direct==='1',scope:'Filtered within the original offline snapshot only; use serve mode for full retrieval.'});}
}catch(e){$('status').textContent=e.message;}}
$('load').onclick=fetchGraph;$('reset').onclick=()=>loadData(data);$('search').onclick=async()=>{try{const q=$('query').value.trim();if(ONLINE){const r=await fetch('/api/search?q='+encodeURIComponent(q));const d=await r.json();if(!r.ok)throw Error(d.error);table(d.nodes);$('status').textContent='Search results (at most 100). Select a row to load its neighborhood.';}else table(INITIAL.nodes.filter(n=>(n.video+' '+n.target_name+' '+n.id).includes(q)));}catch(e){$('status').textContent=e.message;}};
function point(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left-width/2-panX)/zoom,y:(e.clientY-r.top-height/2-panY)/zoom};}function hit(p){return nodes.find(n=>Math.hypot(n.x-p.x,n.y-p.y)<10/zoom+4);}
canvas.onpointerdown=e=>{const n=hit(point(e));drag={n,x:e.clientX,y:e.clientY};if(n){selected=n;details();}canvas.setPointerCapture(e.pointerId);};canvas.onpointermove=e=>{hover=hit(point(e));canvas.style.cursor=hover?'pointer':'grab';if(drag){if(drag.n){Object.assign(drag.n,point(e));drag.n.vx=drag.n.vy=0;}else{panX+=e.clientX-drag.x;panY+=e.clientY-drag.y;}drag.x=e.clientX;drag.y=e.clientY;}};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;canvas.onwheel=e=>{e.preventDefault();const p=point(e);zoom=Math.max(.12,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));const r=canvas.getBoundingClientRect();panX=e.clientX-r.left-width/2-p.x*zoom;panY=e.clientY-r.top-height/2-p.y*zoom;};
function tick(){const dpr=window.devicePixelRatio||1,r=canvas.getBoundingClientRect();width=r.width;height=r.height;if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}if(settle>0){settle--;for(let i=0;i<nodes.length;i++){const a=nodes[i];a.vx-=a.x*.0006;a.vy-=a.y*.0006;for(let j=i+1;j<nodes.length;j++){const b=nodes[j],dx=a.x-b.x,dy=a.y-b.y,f=140/(dx*dx+dy*dy+35);a.vx+=dx*f;a.vy+=dy*f;b.vx-=dx*f;b.vy-=dy*f;}}for(const e of edges){const dx=e.b.x-e.a.x,dy=e.b.y-e.a.y,d=Math.hypot(dx,dy)||1,f=(d-75)*.001;e.a.vx+=dx/d*f;e.a.vy+=dy/d*f;e.b.vx-=dx/d*f;e.b.vy-=dy/d*f;}for(const n of nodes){if(drag?.n===n)continue;n.vx*=.75;n.vy*=.75;n.x+=Math.max(-8,Math.min(8,n.vx));n.y+=Math.max(-8,Math.min(8,n.vy));}}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);ctx.translate(width/2+panX,height/2+panY);ctx.scale(zoom,zoom);const mode=$('edgeMode').value,byOverlap=$('edgeColor').value==='prototype';for(const e of edges){if(mode==='none'||(mode==='selected'&&e.a!==selected&&e.b!==selected))continue;ctx.strokeStyle=edgeStroke(e,byOverlap);ctx.lineWidth=(.5+2*e.stage4_cosine)/Math.sqrt(zoom);ctx.beginPath();ctx.moveTo(e.a.x,e.a.y);ctx.lineTo(e.b.x,e.b.y);ctx.stroke();}for(const n of nodes){ctx.beginPath();ctx.arc(n.x,n.y,n===selected?8:6,0,2*Math.PI);ctx.fillStyle=color(n.pred_percentile);ctx.fill();ctx.strokeStyle=n.id===data.seed?'#ed6d35':n===selected?'#19283c':'#fff';ctx.lineWidth=n.id===data.seed||n===selected?2.5:1;ctx.stroke();if(n===selected||n===hover||nodes.length<35){ctx.fillStyle='#19283c';ctx.font='11px system-ui';ctx.fillText(n.id,n.x+10,n.y-6);}}requestAnimationFrame(tick);}
loadData(INITIAL);tick();
</script></body></html>'''


def write_viewer(graph):
    seed=VIEW_SEED_NODE
    if seed is None:
        row=graph.con.execute("SELECT id FROM nodes WHERE stage='stage4' AND pred_percentile>=0.95 ORDER BY id LIMIT 1").fetchone()
        seed=int(row[0]) if row else int(graph.ids[0])
    payload=graph.payload(seed)
    graph.helper.dump_json(graph.directory/"neighborhood.json",payload)
    encoded=json.dumps(payload,allow_nan=False).replace("<","\\u003c")
    (graph.directory/"graph_viewer.html").write_text(HTML.replace("__PAYLOAD__",encoded),encoding="utf-8")
    print(f"Open {graph.directory/'graph_viewer.html'}",flush=True)


def serve(companion):
    directory=Path(OUTPUT_DIR).resolve()
    # Refresh the HTML shell after generator updates; descriptors are not recomputed.
    with VisualGraph(companion) as graph:
        write_viewer(graph)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                path=urlparse(self.path);q=parse_qs(path.query)
                if path.path in ("/","/graph_viewer.html"):
                    content=(directory/"graph_viewer.html").read_text().replace("const ONLINE=false;","const ONLINE=true;").encode()
                    mime="text/html; charset=utf-8"
                elif path.path in ("/api/graph","/api/search","/api/top-patches"):
                    with VisualGraph(companion) as graph:
                        if path.path=="/api/graph":
                            thresholds=(float(q.get("feature",[MIN_STAGE4_COSINE])[0]),
                                float(q.get("mae",[MAX_RGB_MAE])[0]),float(q.get("structure",[MIN_RGB_STRUCTURE_COSINE])[0]))
                            result=graph.payload(seed=int(q.get("seed",[graph.ids[0]])[0]),thresholds=thresholds,
                                limit=int(q.get("limit",[VIEW_MAX_NODES])[0]),hops=int(q.get("hops",[1])[0]),
                                cross_video=q.get("cross",["1"])[0]=="1",direct=q.get("direct",["1"])[0]=="1")
                        elif path.path=="/api/top-patches":
                            thresholds=(float(q.get("feature",[MIN_STAGE4_COSINE])[0]),
                                float(q.get("mae",[MAX_RGB_MAE])[0]),float(q.get("structure",[MIN_RGB_STRUCTURE_COSINE])[0]))
                            result=graph.make_patch_gallery(
                                node_id=int(q.get("node",[graph.ids[0]])[0]),
                                thresholds=thresholds,
                                cross_video=q.get("cross",["1"])[0]=="1")
                        else:
                            term=q.get("q",[""])[0][:200]
                            rows=graph.con.execute("""SELECT n.id FROM nodes n JOIN frames f ON f.id=n.frame_id
                                WHERE n.stage='stage4' AND (f.video LIKE ? OR f.target_name LIKE ? OR CAST(n.id AS TEXT)=?)
                                ORDER BY n.id LIMIT 100""",(f"%{term}%",f"%{term}%",term)).fetchall()
                            result={"nodes":[graph.node(r[0]) for r in rows]}
                    content=json.dumps(result,allow_nan=False).encode();mime="application/json"
                else:
                    self.send_error(404);return
                self.send_response(200);self.send_header("Content-Type",mime)
                self.send_header("Content-Length",str(len(content)));self.send_header("Cache-Control","no-store")
                self.end_headers();self.wfile.write(content)
            except (ValueError,KeyError,RuntimeError,sqlite3.Error) as exc:
                content=json.dumps({"error":str(exc)}).encode()
                self.send_response(400);self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(content)));self.end_headers();self.wfile.write(content)
    server=ThreadingHTTPServer(("127.0.0.1",SERVER_PORT),Handler)
    print(f"Open http://127.0.0.1:{SERVER_PORT} — Ctrl+C to stop.",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_edges(companion):
    directory=Path(OUTPUT_DIR).resolve();path=directory/"edges.sqlite"
    thresholds=default_thresholds();meta_path=directory/"edges_manifest.json"
    prior=json.loads(meta_path.read_text()) if meta_path.exists() else None
    if prior and prior["thresholds"]!=list(thresholds):
        raise RuntimeError("Existing edge build used different thresholds. Use a new edge output.")
    if prior and prior.get("status")=="complete":
        print("Full edge database is already complete.");return
    with VisualGraph(companion) as graph,closing(sqlite3.connect(path)) as con:
        con.execute("CREATE TABLE IF NOT EXISTS edges(source INTEGER,target INTEGER,stage4_cosine REAL,"
                    "rgb_mae REAL,rgb_structure_cosine REAL,PRIMARY KEY(source,target))")
        start=int(prior.get("completed_source_rows",0)) if prior else 0
        meta={"status":"building","thresholds":list(thresholds),"num_nodes":len(graph.ids),
              "source_manifest_sha256":graph.manifest["source_manifest_sha256"],"completed_source_rows":start}
        if prior and prior.get("source_manifest_sha256")!=meta["source_manifest_sha256"]:
            raise RuntimeError("Edge build belongs to a different node collection.")
        companion.dump_json(meta_path,meta)
        for pos in range(start,len(graph.ids)):
            source=int(graph.ids[pos]);hits=graph.neighbors(source,thresholds,limit=None,cross_video=False)
            con.executemany("INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?)",
                [(source,h["node_id"],h["stage4_cosine"],h["rgb_mae"],h["rgb_structure_cosine"])
                 for h in hits if h["node_id"]>source])
            con.commit()
            meta["completed_source_rows"]=pos+1
            companion.dump_json(meta_path,meta)
            if pos%25==0:
                print(f"Full edge scan {pos+1:,}/{len(graph.ids):,} source nodes",flush=True)
        con.execute("CREATE INDEX IF NOT EXISTS edges_by_target ON edges(target)");con.commit()
        meta.update(status="complete",completed_at=companion.now(),num_edges=con.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        companion.dump_json(meta_path,meta)


def export_edges(companion):
    directory=Path(OUTPUT_DIR).resolve()
    meta=json.loads((directory/"edges_manifest.json").read_text())
    if meta.get("status")!="complete":
        raise RuntimeError("Finish ACTION='build_edges' before CSV export.")
    path=directory/"edges.csv"
    with closing(sqlite3.connect((directory/"edges.sqlite").as_uri()+"?mode=ro",uri=True)) as con,path.open("x",newline="") as f:
        writer=csv.writer(f);writer.writerow(["source","target","stage4_cosine","rgb_mae","rgb_structure_cosine"])
        writer.writerows(con.execute("SELECT source,target,stage4_cosine,rgb_mae,rgb_structure_cosine FROM edges ORDER BY source,target"))
    companion.dump_json(directory/"edges_csv_metadata.json",meta)
    print(f"Exported {meta['num_edges']:,} edges to {path}")


def selftest(companion):
    # The test uses deliberately gapped node IDs to detect accidental renumbering.
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);source=root/"prototype";dest=root/"visual";source.mkdir();dest.mkdir()
        companion.dump_json(source/"manifest.json",{"status":"complete"})
        con=companion.connect_db(source/"graph.sqlite");companion.create_schema(con)
        rng=np.random.default_rng(17);n=25;ids=np.arange(n,dtype=np.int64)*3+2
        features=normalize_rows(rng.normal(size=(n,12)))
        rgb=rng.uniform(0,1,size=(n,3*4*4)).astype(np.float32)
        features[1]=features[0];rgb[1]=rgb[0]
        rgb[2]=.2  # uniform crops have no structural cosine
        for i,nid in enumerate(ids):
            con.execute("INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (i,"val",f"v{i%4}",i,0,15,f"{i}.jpg","none",None,"[]",224,384))
            con.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (int(nid),i,"stage4",0,0,1,1,i/n,i/n,None,None,"[1,2]","[1,0.5]","[1]","[1]"))
            con.executemany("INSERT INTO memberships VALUES (?,?,?,?)",[(int(nid),"stage4",k,1) for k in ("topk","active")])
        con.commit();con.close()
        np.save(dest/"node_ids.npy",ids);np.save(dest/"stage4_features.npy",features);np.save(dest/"rgb_descriptors.npy",rgb)
        companion.dump_json(dest/"manifest.json",{"status":"complete","source_manifest_sha256":companion.sha256_file(source/"manifest.json")})
        assert np.allclose(rgb_structure(np.full(48,.2,dtype=np.float32)),0)
        assert passes({"stage4_cosine":.4,"rgb_mae":.18,"rgb_structure_cosine":.5},(.4,.18,.5))
        with VisualGraph(companion,dest,source) as graph:
            for thresholds in ((.4,.18,.5),(-1,1,-1),(.2,.5,.1)):
                for seed in ids[:5]:
                    for cross in (False,True):
                        oracle=[]
                        for other in ids:
                            if seed==other:
                                continue
                            if cross and graph.node(seed)["video"]==graph.node(other)["video"]:
                                continue
                            scores=graph.scores(seed,other)
                            if passes(scores,thresholds):
                                oracle.append({"node_id":int(other),**scores})
                        oracle.sort(key=sort_key)
                        actual=graph.neighbors(seed,thresholds,None,cross)
                        assert [r["node_id"] for r in actual]==[r["node_id"] for r in oracle]
                        for a,b in zip(actual,oracle):
                            assert all(abs(a[k]-b[k])<2e-6 for k in ("stage4_cosine","rgb_mae","rgb_structure_cosine"))
                        assert [r["node_id"] for r in graph.neighbors(seed,thresholds,3,cross)]==[r["node_id"] for r in oracle[:3]]
            payload=graph.payload(int(ids[0]),(-1,1,-1),limit=10,hops=3,cross_video=True,direct=False)
            assert all(passes(e,(-1,1,-1)) for e in payload["edges"])
            assert all(n["id"] in ids for n in payload["nodes"])
            assert len(payload["nodes"])<=10
            try:
                graph.position(0)
                raise AssertionError("Unknown node ID accepted")
            except ValueError:
                pass
    print("PASS: all-three-gate thresholds, exact block queries vs brute force, gapped node IDs, "
          "video filters, retrieval limits, zero structure and bounded multi-hop payloads.")


def main():
    global MIN_STAGE4_COSINE,MAX_RGB_MAE,MIN_RGB_STRUCTURE_COSINE,ACTION
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-stage4-cosine",type=float,default=MIN_STAGE4_COSINE)
    parser.add_argument("--max-rgb-mae",type=float,default=MAX_RGB_MAE)
    parser.add_argument("--min-rgb-structure-cosine",type=float,default=MIN_RGB_STRUCTURE_COSINE)
    parser.add_argument("--action",choices=("extract","view","serve","build_edges","export_edges","selftest"),default=ACTION)
    args=parser.parse_args()
    MIN_STAGE4_COSINE,MAX_RGB_MAE,MIN_RGB_STRUCTURE_COSINE=validate_thresholds((args.min_stage4_cosine,args.max_rgb_mae,args.min_rgb_structure_cosine))
    ACTION=args.action
    companion=load_companion()
    if ACTION=="extract":
        extract_descriptors(companion)
    elif ACTION=="view":
        with VisualGraph(companion) as graph:
            write_viewer(graph)
    elif ACTION=="serve":
        serve(companion)
    elif ACTION=="build_edges":
        build_edges(companion)
    elif ACTION=="export_edges":
        export_edges(companion)
    else:
        selftest(companion)


if __name__=="__main__":
    main()
