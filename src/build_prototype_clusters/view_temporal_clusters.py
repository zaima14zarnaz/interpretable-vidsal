#!/usr/bin/env python3
"""Host a top-k temporal cluster clip gallery (Pillow + system FFmpeg required).

    pip install pillow
    python view_temporal_clusters.py --cluster-dir /path/to/temporal_clusters
    python view_temporal_clusters.py --cluster-dir /path/to/bank --top-k 12 --no-browser

Open http://127.0.0.1:8768. For a remote machine:
    ssh -L 8768:127.0.0.1:8768 user@server

Reads fixed_stage4_temporal_clusters_v1 banks from form_temporal_clusters.py.
Top-k means nearest to the fixed medoid, not strongest model activation. Clips
are made on demand from source RGB frame directories, with a fixed patch box
(not an object track). Extra context is not part of the flow measurement.
Preview FPS is user supplied; frame-rate metadata is not stored in the bank.
The bank is opened read-only. Generated silent H.264 clips go in a separate
cache, never in the bank. No model inference, PyTorch, or external web assets.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import webbrowser
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore, Timer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

FORMAT_VERSION = "fixed_stage4_temporal_clusters_v1"
DEFAULT_CLUSTER_DIR = Path(
    "/data/quantization/zaima/videosal_datasets/dhf1k/concept_clusters/temporal_finch"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def readonly(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


class TemporalClusterStore:
    def __init__(
        self,
        directory,
        *,
        image_roots=(),
        cache_dir=None,
        fps=30,
        clip_frames=0,
        ffmpeg="ffmpeg",
    ):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if (
            self.manifest.get("status") != "complete"
            or self.manifest.get("format_version") != FORMAT_VERSION
        ):
            raise ValueError(
                "Expected a completed temporal bank from form_temporal_clusters.py."
            )
        self.db_path = self.directory / "temporal_clusters.sqlite"
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Missing database: {self.db_path}")
        self.fps, self.clip_frames = float(fps), int(clip_frames)
        if not math.isfinite(self.fps) or not 1 <= self.fps <= 120:
            raise ValueError("Preview FPS must be in 1..120.")
        if not 0 <= self.clip_frames <= 120:
            raise ValueError(
                "clip-frames must be in 0..120; 0 means the full backbone window."
            )
        self.ffmpeg = shutil.which(str(ffmpeg))
        if not self.ffmpeg:
            raise ValueError(
                "FFmpeg is required. Install it and put it on PATH, or use --ffmpeg /path/to/ffmpeg."
            )
        self.cache_dir = (
            Path(cache_dir or Path(tempfile.gettempdir()) / "temporal_cluster_clips_v1")
            .expanduser()
            .resolve()
        )
        if self.cache_dir == self.directory or self.directory in self.cache_dir.parents:
            raise ValueError(
                "The clip cache must be outside the cluster bank directory."
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_roots = []
        for mapping in image_roots:
            old, sep, new = mapping.partition("=")
            if not sep or not old or not new:
                raise ValueError("--image-root must be OLD_ROOT=NEW_ROOT.")
            self.image_roots.append((Path(old).expanduser(), Path(new).expanduser()))
        self.image_roots.sort(key=lambda pair: len(str(pair[0])), reverse=True)
        self.encoders = BoundedSemaphore(2)
        with closing(readonly(self.db_path)) as con:
            self.prototypes = [
                dict(row) for row in con.execute("SELECT * FROM prototypes ORDER BY id")
            ]
        if (
            not self.prototypes
            or len(self.prototypes) != self.manifest["num_prototypes"]
        ):
            raise ValueError(
                "Temporal prototype table does not match the completed manifest."
            )
        self.by_id = {p["id"]: p for p in self.prototypes}

    def resolve_image(self, value):
        original = Path(value).expanduser()
        for old, new in self.image_roots:
            try:
                mapped = new / original.relative_to(old)
            except ValueError:
                continue
            if mapped.is_file():
                return mapped
        if original.is_file():
            return original
        raise FileNotFoundError(
            f"Source frame unavailable: {original}. Use --image-root OLD_ROOT=NEW_ROOT if it moved."
        )

    def patch(self, node):
        with closing(readonly(self.db_path)) as con:
            row = con.execute(
                """SELECT p.*,f.dataset,f.video,f.window_start,f.target_index,
                f.target_name,f.previous_index,f.previous_name,f.previous_slice,f.current_slice,
                f.input_h,f.input_w,f.source_path,f.previous_source_path
                FROM patches p JOIN frames f ON f.id=p.frame_id WHERE p.node_id=?""",
                (node,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown patch node {node}.")
        return dict(row)

    def clip_sources(self, patch):
        target = self.resolve_image(patch["source_path"])
        previous = self.resolve_image(patch["previous_source_path"])
        if previous.parent != target.parent:
            raise ValueError(
                "Descriptor endpoint frames must be in the same source directory."
            )
        files = sorted(
            (
                p
                for p in target.parent.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=natural_key,
        )
        # Reproduce DatasetLoader's natural order and verify recorded indices.
        end, before, start = (
            patch["target_index"],
            patch["previous_index"],
            patch["window_start"],
        )
        if not (0 <= start <= before < end < len(files)):
            raise ValueError("Invalid recorded frame indices or missing source frames.")
        if (
            files[end].name != patch["target_name"]
            or files[before].name != patch["previous_name"]
        ):
            raise ValueError(
                "Source frame order no longer matches the bank. Restore the original frame directory."
            )
        if self.clip_frames:
            start = max(start, end - self.clip_frames + 1)
        # Always keep both descriptor anchors, even when --clip-frames is short.
        start = min(start, before)
        if end - start + 1 > 120:
            raise ValueError(
                "Preview exceeds 120 frames; set --clip-frames to shorten the context."
            )
        return files[start : end + 1], start

    def cluster(self, prototype, top_k=12, unique_videos=False):
        if prototype not in self.by_id:
            raise ValueError("Unknown temporal prototype ID.")
        if not 1 <= top_k <= 64:
            raise ValueError("top_k must be in 1..64.")
        summary = self.by_id[prototype]
        order = "CASE WHEN p.node_id=? THEN 0 ELSE 1 END,p.distance,p.node_id"
        select = """SELECT p.*,f.dataset,f.video,f.window_start,f.target_index,f.target_name,
            f.previous_index,f.previous_name,f.previous_slice,f.current_slice
            FROM patches p JOIN frames f ON f.id=p.frame_id WHERE p.prototype_id=?"""
        with closing(readonly(self.db_path)) as con:
            if unique_videos:
                sql = f"""WITH ranked AS (SELECT p.node_id,
                    ROW_NUMBER() OVER (PARTITION BY f.dataset,f.video ORDER BY {order}) AS rn
                    FROM patches p JOIN frames f ON f.id=p.frame_id WHERE p.prototype_id=?)
                    {select} AND p.node_id IN (SELECT node_id FROM ranked WHERE rn=1)
                    ORDER BY {order} LIMIT ?"""
                params = (
                    summary["medoid_node_id"],
                    prototype,
                    prototype,
                    summary["medoid_node_id"],
                    top_k,
                )
            else:
                sql, params = (
                    select + f" ORDER BY {order} LIMIT ?",
                    (prototype, summary["medoid_node_id"], top_k),
                )
            rows = [dict(row) for row in con.execute(sql, params)]
        for p in rows:
            p["is_medoid"] = p["node_id"] == summary["medoid_node_id"]
        return {
            "prototype_id": prototype,
            "summary": summary,
            "clips": rows,
            "top_k": top_k,
            "unique_videos": unique_videos,
        }

    def render_frame(self, path, patch, frame_index, view):
        width, height = patch["input_w"], patch["input_h"]
        if not (1 <= width <= 2048 and 1 <= height <= 2048):
            raise ValueError("Unsupported canvas dimensions.")
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        x0 = round(patch["col"] * width / patch["grid_w"])
        x1 = round((patch["col"] + 1) * width / patch["grid_w"])
        y0 = round(patch["row"] * height / patch["grid_h"])
        y1 = round((patch["row"] + 1) * height / patch["grid_h"])
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Invalid stored patch geometry.")
        if view == "crop":
            image = image.crop((x0, y0, x1, y1)).resize(
                (192, 192), Image.Resampling.NEAREST
            )
        elif view == "full":
            ImageDraw.Draw(image).rectangle(
                (x0, y0, x1 - 1, y1 - 1), outline="#ff573b", width=2
            )
        else:
            raise ValueError("view must be full or crop.")
        # yuv420p needs even dimensions; do not change the grid's aspect ratio.
        if image.width % 2 or image.height % 2:
            padded = Image.new(
                "RGB", (image.width + image.width % 2, image.height + image.height % 2)
            )
            padded.paste(image)
            image = padded
        anchor = (
            "PREVIOUS"
            if frame_index == patch["previous_index"]
            else "TARGET"
            if frame_index == patch["target_index"]
            else "context"
        )
        text = f"{anchor} | frame {frame_index}"
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((5, 5), text)
        draw.rectangle((0, 0, box[2] + 5, box[3] + 5), fill="#142337")
        draw.text((5, 5), text, fill="#ffda79" if anchor == "TARGET" else "white")
        return image

    def poster(self, node, view):
        patch = self.patch(node)
        image = self.render_frame(
            self.resolve_image(patch["source_path"]), patch, patch["target_index"], view
        )
        stream = io.BytesIO()
        image.save(stream, "JPEG", quality=88)
        return stream.getvalue()

    def clip(self, node, view="full"):
        if view not in ("full", "crop"):
            raise ValueError("view must be full or crop.")
        patch = self.patch(node)
        sources, start = self.clip_sources(patch)
        identity = {
            "renderer": 1,
            "bank": str(self.directory),
            "patch": patch,
            "fps": self.fps,
            "view": view,
            "start": start,
            "files": [
                (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources
            ],
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        destination = self.cache_dir / f"{key}.mp4"
        if destination.is_file() and destination.stat().st_size:
            return destination
        with self.encoders:
            if destination.is_file() and destination.stat().st_size:
                return destination
            images = [
                self.render_frame(p, patch, start + i, view)
                for i, p in enumerate(sources)
            ]
            width, height = images[0].size
            with tempfile.NamedTemporaryFile(
                suffix=".mp4", dir=self.cache_dir, delete=False
            ) as handle:
                temporary = Path(handle.name)
            try:
                command = [
                    self.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(self.fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ]
                result = subprocess.run(
                    command,
                    check=False,
                    input=b"".join(image.tobytes() for image in images),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
                if result.returncode:
                    raise RuntimeError(
                        "FFmpeg encoding failed: "
                        + result.stderr.decode(errors="replace")[-1500:]
                    )
                if temporary.stat().st_size == 0:
                    raise RuntimeError("FFmpeg produced an empty clip.")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Temporal prototype clips</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#17263b;background:#f3f6fa}*{box-sizing:border-box}
body{margin:0}header{padding:24px 32px;background:white;border-bottom:1px solid #dae2ed}h1{font-size:25px;margin:0 0 7px}
p{line-height:1.6;margin:6px 0}.muted{font-size:13px;color:#5c6c81}.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-top:18px}
label{display:grid;gap:6px;font-size:12px;font-weight:600}input,select,button{font:inherit;padding:9px 12px;border:1px solid #bcc9da;border-radius:7px;background:#fff;color:inherit}
button{cursor:pointer}.primary{background:#214e78;color:#fff;border-color:#214e78}input[type=number]{width:85px}.check{display:flex;align-items:center;padding-bottom:10px}
main{max-width:1800px;margin:auto;padding:22px 32px}.layout{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:24px}.grid{display:grid;grid-template-columns:repeat(__COLUMNS__,minmax(0,1fr));gap:16px}
.card,aside{border:1px solid #d7e1ed;border-radius:10px;background:white;overflow:hidden}.card.medoid{border:2px solid #d2922b}.media{position:relative;background:#142337;aspect-ratio:12/7}
video{display:block;width:100%;height:100%;object-fit:contain}.caption{padding:12px;font-size:12px;line-height:1.65}.caption strong{font-size:14px}.line{display:block;overflow-wrap:anywhere}
.tag{display:inline-block;background:#fff0cc;color:#835100;border-radius:4px;padding:2px 6px;font-size:11px;font-weight:700}.clip-status{font-size:12px;color:#5c6c81;margin-top:6px}.error{color:#a02b36}
.toolbar{display:flex;justify-content:space-between;align-items:center;gap:15px;margin-bottom:16px;flex-wrap:wrap}aside{padding:18px;position:sticky;top:18px;height:fit-content}aside h2{margin:0 0 12px;font-size:18px}
dl{margin:0}dt{font-size:12px;color:#65758a;margin-top:14px}dd{margin:4px 0;font-size:18px;font-weight:650}.footnote{border-top:1px solid #e1e7ef;padding-top:12px;margin-top:18px;font-size:12px;line-height:1.65;color:#607086}
@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.layout{grid-template-columns:minmax(0,1fr) 230px}}
@media(max-width:750px){header,main{padding:18px}.layout{display:flex;flex-direction:column}aside{position:static;order:-1}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:500px){.grid{grid-template-columns:1fr}}
</style></head><body><header><h1>Temporal prototype clips</h1>
<p class="muted">Select a prototype to see its closest reference clips. The red box marks a fixed stage4 cell across frames; it is not an object track.</p>
<form id="controls" class="controls"><label>Temporal prototype<select id="prototype" aria-label="Temporal prototype"></select></label>
<label>Top-k clips<input id="topK" type="number" min="1" max="64" value="__TOP_K__" required></label>
<label>Preview<select id="view"><option value="full">Full frame + patch box</option><option value="crop">Patch crop</option></select></label>
<label class="check"><input id="unique" type="checkbox">One clip per video</label>
<label class="check"><input id="autoplay" type="checkbox" checked>Autoplay visible clips</label>
<button class="primary" type="submit">Load clips</button></form>
<p id="coverage" class="muted"></p></header><main><div class="toolbar"><span id="status" role="status" aria-live="polite">Loading prototypes…</span>
<div><button id="playAll">Play visible</button> <button id="pauseAll">Pause all</button></div></div>
<div class="layout"><section id="grid" class="grid" aria-label="Temporal cluster clips"></section>
<aside><h2 id="summaryTitle">Cluster summary</h2><dl id="summary"></dl>
<p class="footnote">Ranked by composite temporal distance to the fixed medoid, not trained-model activation. Mean distance describes closeness to the medoid, not pairwise coherence.</p>
<p class="footnote">PREVIOUS and TARGET mark the descriptor’s RGB endpoints. Other frames provide viewing context. Feature differences also contain the backbone’s temporal context.</p>
</aside></div></main><script>
'use strict';
const $=id=>document.getElementById(id);let requestId=0,controller=null,observer=null;
const fmt=(x,n=4)=>x==null?'—':Number(x).toFixed(n);
function text(tag,value,cls){const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e;}
async function json(url,signal){const r=await fetch(url,{signal});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data;}
function cleanup(){if(observer)observer.disconnect();for(const v of document.querySelectorAll('video')){v.pause();v.removeAttribute('src');v.load();}}
function play(v){if(v.getAttribute('src'))v.play().catch(()=>{});}
function setupObserver(){observer=new IntersectionObserver(entries=>{for(const {target:v,isIntersecting} of entries){
 if(isIntersecting){if(!v.getAttribute('src')){v.src=v.dataset.src;v.load();}if($('autoplay').checked)play(v);}else v.pause();
 }},{rootMargin:'80px'});}
function card(p,rank,view){const article=text('article','',p.is_medoid?'card medoid':'card');const media=text('div','','media');
 const v=document.createElement('video');v.controls=true;v.loop=true;v.muted=true;v.defaultMuted=true;v.playsInline=true;v.preload='none';
 const params=new URLSearchParams({node:p.node_id,view});v.poster='/poster?'+params;v.dataset.src='/clip?'+params;
 v.setAttribute('aria-label',`Prototype clip ${rank}, video ${p.video}, patch ${p.row},${p.col}`);media.append(v);article.append(media);
 const cap=text('div','','caption');cap.append(text('strong',`#${rank} · ${p.video}`));if(p.is_medoid)cap.append(text('span','Reference medoid','tag'));
 cap.append(text('span',`Distance ${fmt(p.distance)} · patch (${p.row}, ${p.col})`,'line'));
 cap.append(text('span',`Descriptor frames ${p.previous_name} → ${p.target_name}`,'line'));
 cap.append(text('span',`Flow ${fmt(p.flow_magnitude,2)} px · feature change ${fmt(p.change_magnitude,3)}`,'line'));
 cap.append(text('span',`Flow coverage ${fmt(p.flow_valid_fraction*100,1)}% · flow ${p.flow_valid?'valid':'unknown'} · feature ${p.change_valid?(p.change_static?'static':'changing'):'unknown'}`,'line'));
 const link=text('a','Download MP4');link.href='/clip?'+params+'&download=1';cap.append(link);
 const state=text('div','Clip loads when visible.','clip-status');cap.append(state);
 v.addEventListener('loadeddata',()=>{state.textContent=`${fmt(v.duration,2)} s · silent preview`;const r=v.getBoundingClientRect();if($('autoplay').checked&&!document.hidden&&r.top<innerHeight&&r.bottom>0)play(v);});
 v.addEventListener('error',async()=>{if(!v.getAttribute('src'))return;state.classList.add('error');state.textContent='Clip unavailable.';
 try{const r=await fetch('/api/clip-status?'+params);const d=await r.json();state.textContent=d.error||'Clip unavailable.';}catch{}});
 article.append(cap);$('grid').append(article);observer.observe(v);
}
async function load(){const id=++requestId;if(controller)controller.abort();controller=new AbortController();cleanup();$('grid').replaceChildren();$('status').className='';$('status').textContent='Loading nearest clips…';
 const params=new URLSearchParams({prototype:$('prototype').value,top_k:$('topK').value,unique:$('unique').checked?'1':'0'});
 try{const data=await json('/api/cluster?'+params,controller.signal);if(id!==requestId)return;
 $('summaryTitle').textContent=`Prototype ${data.prototype_id}`;$('summary').replaceChildren();const s=data.summary;
 for(const [name,val] of [['Assigned patches',s.patch_count.toLocaleString()],['Source videos',s.video_count.toLocaleString()],['Mean medoid distance',fmt(s.mean_distance)],['Mean residual flow',fmt(s.mean_flow_magnitude,2)+' px'],['Mean feature change',fmt(s.mean_change_magnitude,3)]]){
 $('summary').append(text('dt',name),text('dd',val));}
 $('status').textContent=`Showing ${data.clips.length} nearest clip${data.clips.length===1?'':'s'}${data.unique_videos?' from distinct videos':''}. Patch coordinates and frame indices are zero-based.`;
 setupObserver();data.clips.forEach((p,i)=>card(p,i+1,$('view').value));if(!data.clips.length)$('grid').append(text('p','This cluster has no members.'));
 }catch(e){if(e.name==='AbortError')return;$('status').textContent=e.message;$('status').className='error';}}
$('controls').addEventListener('submit',e=>{e.preventDefault();load();});$('prototype').addEventListener('change',load);
$('playAll').onclick=()=>{for(const v of document.querySelectorAll('video')){const r=v.getBoundingClientRect();if(r.top<innerHeight&&r.bottom>0)play(v);}};
$('pauseAll').onclick=()=>{for(const v of document.querySelectorAll('video'))v.pause();$('autoplay').checked=false;};
$('autoplay').onchange=()=>{if(!$('autoplay').checked)for(const v of document.querySelectorAll('video'))v.pause();else $('playAll').click();};
document.addEventListener('visibilitychange',()=>{if(document.hidden)for(const v of document.querySelectorAll('video'))v.pause();});
(async()=>{try{const init=await json('/api/init');for(const p of init.prototypes){const o=text('option',`Prototype ${p.id} · ${p.patch_count.toLocaleString()} patches`);o.value=p.id;$('prototype').append(o);}
 $('prototype').value=String(__PROTOTYPE__);$('coverage').textContent=`${init.num_prototypes} temporal prototypes · ${init.num_patches.toLocaleString()} patches · ${init.num_videos} videos. Preview: ${init.fps} FPS (assumed). ${init.clip_frames?'Up to '+init.clip_frames+' trailing frames, extended to retain both endpoints.':'Full recorded backbone window.'}`;await load();
 }catch(e){$('status').textContent=e.message;$('status').className='error';}})();
</script></body></html>"""


def byte_range(value, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()):
        raise ValueError("Invalid byte range.")
    left, right = match.groups()
    if not left:
        length = int(right)
        if length <= 0:
            raise ValueError("Invalid suffix range.")
        return max(0, size - length), size - 1
    start, end = int(left), min(int(right), size - 1) if right else size - 1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable byte range.")
    return start, end


def make_server(store, port=8768, prototype=0, top_k=12, columns=3):
    if prototype not in store.by_id:
        raise ValueError("Initial prototype is not present in the bank.")
    if not 1 <= top_k <= 64 or not 1 <= columns <= 8:
        raise ValueError("top-k must be in 1..64 and columns in 1..8.")
    html = (
        HTML.replace("__PROTOTYPE__", str(prototype))
        .replace("__TOP_K__", str(top_k))
        .replace("__COLUMNS__", str(columns))
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, content, mime, headers=()):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(content)

        def json(self, status, payload):
            self.reply(
                status,
                json.dumps(payload, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def video(self, path, download, node):
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            if self.headers.get("Range"):
                try:
                    start, end = byte_range(self.headers["Range"], size)
                    status = 206
                except ValueError:
                    self.reply(
                        416, b"", "video/mp4", [("Content-Range", f"bytes */{size}")]
                    )
                    return
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "private, max-age=3600")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="temporal_patch_{node}.mp4"',
                )
            self.end_headers()
            if self.command == "HEAD":
                return
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = stream.read(min(65536, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                url = urlparse(self.path)
                q = parse_qs(url.query)
                if url.path in ("/", "/index.html"):
                    self.reply(200, html, "text/html; charset=utf-8")
                elif url.path == "/api/init":
                    self.json(
                        200,
                        {
                            "prototypes": store.prototypes,
                            "fps": store.fps,
                            "clip_frames": store.clip_frames,
                            **{
                                k: store.manifest[k]
                                for k in ("num_prototypes", "num_patches", "num_videos")
                            },
                        },
                    )
                elif url.path == "/api/cluster":
                    self.json(
                        200,
                        store.cluster(
                            int(q.get("prototype", ["0"])[0]),
                            int(q.get("top_k", [str(top_k)])[0]),
                            q.get("unique", ["0"])[0] == "1",
                        ),
                    )
                elif url.path in ("/clip", "/poster", "/api/clip-status"):
                    node, view = (
                        int(q.get("node", ["-1"])[0]),
                        q.get("view", ["full"])[0],
                    )
                    if url.path == "/poster":
                        self.reply(200, store.poster(node, view), "image/jpeg")
                    elif url.path == "/api/clip-status":
                        # Validation only; never repeat a failed encoding request.
                        p = store.patch(node)
                        store.clip_sources(p)
                        self.json(
                            200,
                            {
                                "error": "Clip could not be played. Check the terminal for the FFmpeg error; your FFmpeg/browser needs H.264 support."
                            },
                        )
                    else:
                        self.video(
                            store.clip(node, view),
                            q.get("download", ["0"])[0] == "1",
                            node,
                        )
                elif url.path == "/favicon.ico":
                    self.reply(204, b"", "image/x-icon")
                else:
                    self.json(404, {"error": "Not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (
                ValueError,
                KeyError,
                OSError,
                sqlite3.Error,
                RuntimeError,
                subprocess.TimeoutExpired,
            ) as exc:
                print(f"Viewer request failed: {exc}", file=sys.stderr)
                try:
                    self.json(400, {"error": str(exc)})
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, fmt, *args):
            if args and any(
                str(args[0]).startswith("GET " + prefix)
                for prefix in ("/clip", "/poster")
            ):
                return
            super().log_message(fmt, *args)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--prototype", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Assumed preview FPS, not inferred from the bank.",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=0,
        help="Trailing context length; 0=full backbone window. Both descriptor anchors are always kept.",
    )
    parser.add_argument(
        "--image-root", action="append", default=[], metavar="OLD_ROOT=NEW_ROOT"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Separate generated MP4 cache (default: system temp directory).",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be in 1..65535.")
    try:
        store = TemporalClusterStore(
            args.cluster_dir,
            image_roots=args.image_root,
            cache_dir=args.cache_dir,
            fps=args.fps,
            clip_frames=args.clip_frames,
            ffmpeg=args.ffmpeg,
        )
        server = make_server(store, args.port, args.prototype, args.top_k, args.columns)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{server.server_port}"
    print(
        f"Open {url} | {len(store.prototypes)} temporal prototypes | nearest-medoid clips | stop with Ctrl+C",
        flush=True,
    )
    if not args.no_browser:
        timer = Timer(0.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
