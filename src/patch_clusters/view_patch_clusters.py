#!/usr/bin/env python3
"""Browse fixed patch clusters produced by form_patch_clusters.py.

Dependency: python -m pip install pillow
    python view_patch_clusters.py
    python view_patch_clusters.py --prototype 12 --port 8767 --columns 8
    python view_patch_clusters.py --cluster-dir /path/to/patch_cluster --no-browser

Open http://127.0.0.1:8767 and enter a prototype/cluster ID in the text field.
IDs are the NEW fixed reference IDs, not the original trained model's IDs.
The gallery is paginated over ALL members; it is not a top-N activation gallery.
Click a patch for its full frame and grid box. The default ordering is proximity
to the fixed medoid; source-node order is also available. The statistics report
mean distance/similarity to the reference medoid, NOT pairwise cluster coherence.

Images are read on demand from original dataset paths stored in the cluster DB,
falling back to source graph thumbnails. Keep those source images accessible;
--image-root OLD_ROOT=NEW_ROOT can remap paths after moving a dataset/cache.
The server binds to loopback only. On a remote server, forward the port with:
    ssh -L 8767:127.0.0.1:8767 user@server

Cluster data and source graphs are opened read-only. No model inference,
prototype reassignment, training, external CDNs, or public hosting is involved.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import sqlite3
import sys
from threading import Timer
from urllib.parse import parse_qs, urlparse
import webbrowser

from PIL import Image, ImageDraw, ImageOps

DEFAULT_CLUSTER_DIR = Path('/data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster')
FORMAT_VERSION = 'fixed_stage4_patch_clusters_v1'


def readonly(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    return con


class ClusterStore:
    def __init__(self, directory, image_roots=()):
        self.directory = Path(directory).expanduser().resolve()
        self.db_path = self.directory / 'patch_clusters.sqlite'
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('status') != 'complete' or self.manifest.get('format_version') != FORMAT_VERSION:
            raise ValueError('Cluster generation is incomplete or incompatible. Run form_patch_clusters.py successfully first.')
        if not self.db_path.is_file():
            raise FileNotFoundError(f'Cluster database missing: {self.db_path}')
        self.image_roots = []
        for mapping in image_roots:
            old, separator, new = mapping.partition('=')
            if not separator or not old or not new:
                raise ValueError('--image-root must have the form OLD_ROOT=NEW_ROOT.')
            self.image_roots.append((Path(old).expanduser(), Path(new).expanduser()))
        self.image_roots.sort(key=lambda pair: len(str(pair[0])), reverse=True)

    def resolve_image(self, path):
        if not path:
            return None
        original = Path(path).expanduser()
        for old, new in self.image_roots:
            try:
                relative = original.relative_to(old)
            except ValueError:
                continue
            mapped = new / relative
            if mapped.is_file():
                return mapped
        return original if original.is_file() else None

    def validate_prototype(self, prototype):
        if not 0 <= prototype < self.manifest['num_prototypes']:
            raise ValueError(f'Prototype must be between 0 and {self.manifest["num_prototypes"] - 1}.')

    def cluster(self, prototype, offset=0, limit=80, order='distance'):
        self.validate_prototype(prototype)
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError('offset must be >= 0 and limit must be in 1..200.')
        if order not in ('distance', 'node'):
            raise ValueError('order must be distance or node.')
        with closing(readonly(self.db_path)) as con:
            summary_row = con.execute('SELECT * FROM prototypes WHERE id=?', (prototype,)).fetchone()
            if summary_row is None:
                raise ValueError('Prototype data is missing from the cluster database.')
            summary = dict(summary_row)
            sort = ('CASE WHEN p.node_id=? THEN 0 ELSE 1 END,p.distance,p.node_id'
                    if order == 'distance' else 'p.node_id')
            params = [prototype]
            if order == 'distance':
                params.append(summary['medoid_node_id'])
            params.extend([limit, offset])
            rows = con.execute(f'''SELECT p.*,f.dataset,f.video,f.dataset_index,f.window_start,
                     f.target_index,f.target_name,f.input_h,f.input_w
                     FROM patches p JOIN frames f ON f.id=p.frame_id
                     WHERE p.prototype_id=? ORDER BY {sort} LIMIT ? OFFSET ?''', params).fetchall()
            mean = con.execute('''SELECT AVG(feature_cosine) AS feature_cosine,AVG(rgb_mae) AS rgb_mae,
                     AVG(structure_cosine) AS structure_cosine FROM patches WHERE prototype_id=?''', (prototype,)).fetchone()
            summary.update(dict(mean))
            summary['mean_visual_similarity_to_medoid'] = mean['feature_cosine'] + (1 - mean['rgb_mae']) + mean['structure_cosine']
        patches = []
        for row in rows:
            patch = dict(row)
            patch['is_medoid'] = patch['node_id'] == summary['medoid_node_id']
            patch['visual_similarity_to_medoid'] = patch['feature_cosine'] + (1 - patch['rgb_mae']) + patch['structure_cosine']
            patches.append(patch)
        return {'prototype_id': prototype, 'offset': offset, 'limit': limit, 'order': order,
                'total': summary['patch_count'], 'summary': summary, 'patches': patches}

    def patch(self, node_id):
        with closing(readonly(self.db_path)) as con:
            row = con.execute('''SELECT p.*,f.dataset,f.video,f.target_name,f.source_path,
                     f.thumb_path,f.input_h,f.input_w FROM patches p JOIN frames f ON f.id=p.frame_id
                     WHERE p.node_id=?''', (node_id,)).fetchone()
            if row is None:
                raise ValueError(f'Patch node {node_id} does not exist.')
            return dict(row)

    @lru_cache(maxsize=16)
    def frame(self, frame_id):
        with closing(readonly(self.db_path)) as con:
            row = con.execute('SELECT * FROM frames WHERE id=?', (frame_id,)).fetchone()
            if row is None:
                raise ValueError('Frame does not exist.')
        for candidate in (row['source_path'], row['thumb_path']):
            path = self.resolve_image(candidate)
            if path is None:
                continue
            try:
                with Image.open(path) as image:
                    rgb = image.convert('RGB')
                # Match the graph's uniform input-canvas grid for previews.
                # JPEG previews do not participate in any similarity metric.
                rgb = rgb.resize((row['input_w'], row['input_h']), Image.Resampling.BILINEAR)
                rgb.thumbnail((1024, 1024))
                return rgb
            except (OSError, ValueError):
                continue
        return None

    @lru_cache(maxsize=512)
    def image(self, node_id, full=False):
        patch = self.patch(node_id)
        frame = self.frame(patch['frame_id'])
        if frame is None:
            image = Image.new('RGB', (240, 180), '#e8edf4')
            ImageDraw.Draw(image).multiline_text((18, 65), 'Source image unavailable\nCheck --image-root mapping', fill='#41526a', spacing=8)
        elif full:
            image = frame.copy()
        else:
            w, h = frame.size
            box = (round(patch['col'] * w / patch['grid_w']), round(patch['row'] * h / patch['grid_h']),
                   round((patch['col'] + 1) * w / patch['grid_w']), round((patch['row'] + 1) * h / patch['grid_h']))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Patch crop is empty at the stored canvas size.')
            image = frame.crop(box)
            image = ImageOps.contain(image, (192, 192), Image.Resampling.BILINEAR)
        stream = io.BytesIO()
        image.save(stream, 'JPEG', quality=90)
        return stream.getvalue(), frame is not None


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fixed patch clusters</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#17263b;background:#f3f6fa}*{box-sizing:border-box}
body{margin:0}header{background:#fff;border-bottom:1px solid #dae2ed;padding:24px 32px}
h1{margin:0 0 7px;font-size:25px}p{line-height:1.55;margin:6px 0}.muted{color:#5c6c81;font-size:13px}
.controls{display:flex;gap:14px;align-items:end;flex-wrap:wrap;margin-top:18px}label{display:grid;gap:6px;font-size:12px;font-weight:600}
input,select,button{font:inherit;padding:9px 12px;border:1px solid #bcc9da;border-radius:7px;background:white;color:inherit}
input{width:150px}button{cursor:pointer}button:disabled{cursor:default;opacity:.45}.primary{background:#214e78;color:#fff;border-color:#214e78}
main{max-width:1900px;margin:auto;padding:22px 32px}.layout{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:24px}
.toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:16px}
#status{font-size:14px;min-height:24px}.pager{display:flex;align-items:center;gap:10px;font-size:13px}
.grid{--cols:8;display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:12px}
.card{background:white;border:1px solid #d7e1ed;border-radius:10px;overflow:hidden;min-width:0}
.card.medoid{border:2px solid #d2922b}.card button{padding:0;border:0;border-radius:0;width:100%;text-align:left;background:transparent}
.card button:focus-visible{outline:3px solid #214e78;outline-offset:-3px}.crop{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#e9eef5}
.caption{padding:10px;font-size:11px;line-height:1.55}.caption strong{display:block;font-size:12px}.line{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tag{display:inline-block;color:#835100;background:#fff0cc;border-radius:4px;padding:2px 5px;margin-top:4px;font-size:10px;font-weight:700}
aside{background:#fff;border:1px solid #d7e1ed;border-radius:11px;padding:18px;height:fit-content;position:sticky;top:18px}aside h2{font-size:18px;margin:0 0 12px}
dl{margin:0}dt{font-size:11px;color:#65758a;margin-top:13px}dd{margin:4px 0 0;font-size:16px;font-weight:650}.reference{width:100%;max-height:170px;object-fit:contain;background:#edf2f8;border-radius:6px;margin-top:12px}
.footer{padding-top:16px;border-top:1px solid #e1e7ef;margin-top:16px;font-size:11px;color:#607086;line-height:1.6}
dialog{width:min(950px,94vw);border:1px solid #ccd6e3;border-radius:12px;padding:20px;color:#17263b}dialog::backdrop{background:#0b1c30aa}
.modal-head{display:flex;align-items:center;justify-content:space-between;gap:15px;margin-bottom:12px}.modal-head h2{font-size:18px;margin:0}
.frame{position:relative;background:#e9eef5;line-height:0}.frame img{display:block;width:100%;height:auto}.box{position:absolute;border:3px solid #ee753a;pointer-events:none}
#empty{padding:40px;background:white;border:1px solid #d7e1ed;border-radius:10px}.error{color:#a02b36}
@media(max-width:1200px){.grid{grid-template-columns:repeat(4,minmax(0,1fr))}.layout{grid-template-columns:minmax(0,1fr) 250px}}
@media(max-width:750px){header{padding:20px}main{padding:18px}.layout{display:flex;flex-direction:column}aside{position:static;order:-1;width:100%}.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:450px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body>
<header><h1>Fixed patch clusters</h1><p class="muted">Browse the appearance-based reference group for a prototype. These cluster IDs define a new vocabulary for training; they do not use the previous model’s prototype activations.</p>
<form id="controls" class="controls"><label>Prototype / cluster ID<input id="prototype" type="text" inputmode="numeric" pattern="[0-9]+" required value="__PROTOTYPE__" aria-label="Prototype ID"></label>
<label>Order<select id="order"><option value="distance">Nearest reference first</option><option value="node">Source node ID</option></select></label>
<label>Patches per page<select id="limit"><option>40</option><option selected>80</option><option>160</option></select></label>
<button class="primary" type="submit">Load patches</button><span id="coverage" class="muted"></span></form></header>
<main><div class="toolbar"><div id="status" role="status" aria-live="polite">Loading clusters…</div><div class="pager"><button id="previous" disabled>Previous</button><span id="page"></span><button id="next" disabled>Next</button></div></div>
<div class="layout"><section><div id="grid" class="grid" style="--cols:__COLUMNS__"></div><div id="empty" hidden>No patches on this page.</div></section><aside id="summary"><h2>Reference group</h2><p class="muted">Select a prototype to inspect its reference patches.</p></aside></div></main>
<dialog id="detail"><div class="modal-head"><h2 id="detailTitle"></h2><button id="close" autofocus>Close</button></div><div class="frame"><img id="frame" alt="Source frame"><div id="box" class="box"></div></div><p id="detailInfo" class="muted"></p></dialog>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const score=x=>x==null?'unavailable':Number(x).toFixed(4),count=x=>Number(x).toLocaleString();
let offset=0,result=null,manifest=null,requestId=0;
async function json(url){const response=await fetch(url);const body=await response.json();if(!response.ok)throw Error(body.error||'Request failed');return body;}
function setSummary(s){$('summary').innerHTML=`<h2>Prototype ${s.id}</h2><p class="muted">Fixed reference medoid</p><img class="reference" src="/image?node=${s.medoid_node_id}" alt="Reference patch for prototype ${s.id}"><p class="muted">Source node ${s.medoid_node_id}</p><dl><dt>Assigned patches</dt><dd>${count(s.patch_count)}</dd><dt>Distinct videos</dt><dd>${count(s.video_count)}</dd><dt>Mean distance to reference</dt><dd>${score(s.mean_distance)}</dd><dt>Mean visual similarity to reference</dt><dd>${score(s.mean_visual_similarity_to_medoid)}</dd><dt>Mean saliency percentile ± standard deviation</dt><dd>${s.mean_saliency_percentile==null?'unavailable':(100*s.mean_saliency_percentile).toFixed(1)+'% ± '+(100*s.std_saliency_percentile).toFixed(1)+' pp'}</dd></dl><div class="footer">Saliency is the source graph’s within-frame predicted rank; it was not used for clustering. Standard deviation is across cluster members (population).<br><br>Distance uses feature cosine, RGB MAE and centered RGB structure cosine. Similarity to the medoid is not pairwise cluster coherence.</div>`;}
function showPatch(p){$('detailTitle').textContent=`Patch ${p.node_id} · prototype ${p.prototype_id}`;$('frame').src='/image?node='+p.node_id+'&full=1';const b=$('box');b.style.left=100*p.col/p.grid_w+'%';b.style.top=100*p.row/p.grid_h+'%';b.style.width=100/p.grid_w+'%';b.style.height=100/p.grid_h+'%';$('detailInfo').textContent=`${p.dataset} / ${p.video} / ${p.target_name} · stage4 cell ${p.row},${p.col} · distance ${score(p.distance)} · feature cosine ${score(p.feature_cosine)} · RGB MAE ${score(p.rgb_mae)} · structure cosine ${score(p.structure_cosine)}`;$('detail').showModal();}
async function load(){const request=++requestId;try{const text=$('prototype').value.trim();if(!/^\d+$/.test(text))throw Error('Enter a nonnegative integer prototype ID.');const id=Number(text);if(!Number.isSafeInteger(id)||id>=manifest.num_prototypes)throw Error(`Prototype must be between 0 and ${manifest.num_prototypes-1}.`);$('status').className='';$('status').textContent='Loading reference patches…';const q=new URLSearchParams({prototype:id,offset,limit:$('limit').value,order:$('order').value});const d=await json('/api/cluster?'+q);if(request!==requestId)return;result=d;$('status').textContent=`Prototype ${id} · ${count(d.total)} patches · showing ${d.patches.length?count(d.offset+1):'0'}–${count(d.offset+d.patches.length)}`;$('page').textContent=`Page ${Math.floor(d.offset/d.limit)+1} of ${Math.max(1,Math.ceil(d.total/d.limit))}`;$('previous').disabled=d.offset===0;$('next').disabled=d.offset+d.patches.length>=d.total;
$('grid').innerHTML=d.patches.map((p,i)=>`<article class="card${p.is_medoid?' medoid':''}"><button type="button" data-index="${i}" aria-label="Open source frame for patch ${p.node_id}"><img class="crop" loading="lazy" src="/image?node=${p.node_id}" alt="Patch ${p.node_id}"><div class="caption"><strong>Patch ${p.node_id}</strong><span class="line" title="${esc(p.dataset+' / '+p.video+' / '+p.target_name)}">${esc(p.video)} / ${esc(p.target_name)}</span><span class="line">Cell ${p.row},${p.col} · ${p.grid_h}×${p.grid_w}</span><span class="line">Distance ${score(p.distance)}</span><span class="line">Saliency ${p.pred_percentile==null?'unavailable':(100*p.pred_percentile).toFixed(1)+'%'}</span>${p.is_medoid?'<span class="tag">REFERENCE MEDOID</span>':''}</div></button></article>`).join('');$('grid').querySelectorAll('[data-index]').forEach(button=>button.onclick=()=>showPatch(d.patches[Number(button.dataset.index)]));$('empty').hidden=d.patches.length>0;setSummary(d.summary);
}catch(error){if(request===requestId){$('status').className='error';$('status').textContent=error.message;}}}
$('controls').onsubmit=e=>{e.preventDefault();offset=0;load();};$('previous').onclick=()=>{offset=Math.max(0,result.offset-result.limit);load();};$('next').onclick=()=>{offset=result.offset+result.limit;load();};$('close').onclick=()=>$('detail').close();
json('/api/manifest').then(d=>{manifest=d;$('coverage').textContent=`IDs 0–${d.num_prototypes-1} · ${count(d.num_patches)} stage4 patches`;load();}).catch(e=>{$('status').className='error';$('status').textContent=e.message;});
</script></body></html>'''


def make_server(store, port=8767, prototype=0, columns=8):
    store.validate_prototype(prototype)
    html = HTML.replace('__PROTOTYPE__', str(prototype)).replace('__COLUMNS__', str(columns)).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, content, content_type, extra_headers=()):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in extra_headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            try:
                url = urlparse(self.path)
                q = parse_qs(url.query)
                if url.path in ('/', '/index.html'):
                    self.reply(200, html, 'text/html; charset=utf-8')
                    return
                if url.path == '/api/manifest':
                    payload = {k: store.manifest[k] for k in ('format_version', 'num_prototypes', 'num_patches', 'num_videos', 'weights', 'rgb_descriptor_size')}
                elif url.path == '/api/cluster':
                    payload = store.cluster(int(q.get('prototype', ['0'])[0]), int(q.get('offset', ['0'])[0]),
                                            int(q.get('limit', ['80'])[0]), q.get('order', ['distance'])[0])
                elif url.path == '/image':
                    content, found = store.image(int(q.get('node', ['-1'])[0]), q.get('full', ['0'])[0] == '1')
                    self.reply(200, content, 'image/jpeg', [('X-Source-Image-Available', '1' if found else '0')])
                    return
                elif url.path == '/favicon.ico':
                    self.reply(204, b'', 'image/x-icon')
                    return
                else:
                    self.reply(404, b'{"error":"Not found"}', 'application/json')
                    return
                content = json.dumps(payload, allow_nan=False).encode('utf-8')
                self.reply(200, content, 'application/json; charset=utf-8')
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, KeyError, OSError, sqlite3.Error) as exc:
                self.reply(400, json.dumps({'error': str(exc)}).encode(), 'application/json; charset=utf-8')

        def log_message(self, fmt, *args):
            # Gallery image requests can otherwise bury useful server messages.
            if args and str(args[0]).startswith('GET /image'):
                return
            super().log_message(fmt, *args)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cluster-dir', type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument('--prototype', type=int, default=0)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--columns', type=int, default=8)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--image-root', action='append', default=[], metavar='OLD_ROOT=NEW_ROOT')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.columns <= 16:
        parser.error('Port must be in 1..65535 and columns in 1..16.')
    try:
        store = ClusterStore(args.cluster_dir, args.image_root)
        server = make_server(store, args.port, args.prototype, args.columns)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Open {url} — enter prototype IDs 0..{store.manifest["num_prototypes"]-1}; stop with Ctrl+C.', flush=True)
    if not args.no_browser:
        timer = Timer(.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
