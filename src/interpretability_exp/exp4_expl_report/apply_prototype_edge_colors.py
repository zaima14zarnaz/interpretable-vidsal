#!/usr/bin/env python3
"""Add prototype-overlap edge coloring to the existing visual graph viewer.

Run beside your generator:
    python apply_prototype_edge_colors.py generate_vis_sim_graph.py

Then regenerate with --action view, or restart --action serve. No extraction
is needed. Only the embedded HTML is changed; a .bak copy is kept once.
"""

import argparse
import ast
from pathlib import Path
import re
import sys


CONTROLS = '''<label>Edge color<select id="edgeColor" title="Colors existing visible edges only; does not change edge visibility."><option value="uniform">Original</option><option value="prototype">Prototype overlap (top-k)</option></select></label><span id="prototypeEdgeLegend" class="muted" hidden title="Jaccard overlap of unique top-k candidate IDs: shared IDs / union of IDs. Missing or empty evidence is gray.">Prototype overlap: 0% <span style="display:inline-block;width:100px;height:10px;background:linear-gradient(90deg,rgb(37,99,235),rgb(220,38,38));vertical-align:middle;margin:0 5px"></span> 100% <span style="color:#6b7280">· gray = unavailable</span></span>'''

HELPERS = '''
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
'''


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one matching viewer fragment, found {count}: {old[:90]!r}. No file changed.")
    return text.replace(old, new, 1)


def patch_html(html):
    if 'id="edgeColor"' in html:
        if all(marker in html for marker in ('function prototypeOverlap(', 'function edgeStroke(', 'ctx.strokeStyle=edgeStroke(e,byOverlap);')):
            return html
        raise ValueError('An existing edgeColor control differs from this patch. No file changed.')
    html = replace_once(html, '<span class="muted" id="mode">', CONTROLS+'<span class="muted" id="mode">')
    html = replace_once(html, 'function loadData(d){', HELPERS+'\nfunction loadData(d){')
    old = 'byId=new Map(nodes.map(n=>[n.id,n]));edges=d.edges.map(e=>({...e,a:byId.get(e.source),b:byId.get(e.target)}));'
    new = 'byId=new Map(nodes.map(n=>[n.id,n]));const prototypeSets=new Map(nodes.map(n=>[n.id,prototypeIdSet(n)]));edges=d.edges.map(e=>({...e,a:byId.get(e.source),b:byId.get(e.target),prototypeOverlap:prototypeOverlap(prototypeSets.get(e.source),prototypeSets.get(e.target))}));'
    html = replace_once(html, old, new)
    html = replace_once(html, "const mode=$('edgeMode').value;for(const e of edges)", "const mode=$('edgeMode').value,byOverlap=$('edgeColor').value==='prototype';for(const e of edges)")
    html = replace_once(html, "ctx.strokeStyle='rgba(60,90,130,.25)';", 'ctx.strokeStyle=edgeStroke(e,byOverlap);')
    return html


def patch_source(source, filename='<generator>'):
    # Parse without importing: no companion script, NumPy, Torch or dataset needed.
    tree = ast.parse(source, filename=filename)
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'HTML' for target in node.targets)]
    if len(matches) != 1 or not isinstance(matches[0], ast.Constant) or not isinstance(matches[0].value, str):
        raise ValueError('Expected one top-level HTML string in the generator. No file changed.')
    value = matches[0]
    updated = patch_html(value.value)
    if updated == value.value:
        return source
    # AST columns use UTF-8 byte offsets. Replace only the literal, preserving
    # every byte of the Python extraction/query/build/export implementation.
    lines = source.encode('utf-8').splitlines(keepends=True)
    start = sum(map(len, lines[:value.lineno-1]))+value.col_offset
    end = sum(map(len, lines[:value.end_lineno-1]))+value.end_col_offset
    quote = next((q for q in ("'''", '"""') if q not in updated), None)
    literal = ('r'+quote+updated+quote) if quote and not updated.endswith('\\') else repr(updated)
    result = (source.encode('utf-8')[:start]+literal.encode('utf-8')+source.encode('utf-8')[end:]).decode('utf-8')
    ast.parse(result, filename=filename)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('generator', nargs='?', type=Path, help='Path to the original visual similarity graph generator')
    args = parser.parse_args()
    path = args.generator
    if path is None:
        candidates = [Path(name) for name in ('generate_vis_sim_graph.py', 'generate_visual_similarity_graph.py') if Path(name).is_file()]
        if len(candidates) != 1:
            parser.error('Pass the path to your original visual graph generator .py file.')
        path = candidates[0]
    path = path.resolve()
    if path.suffix != '.py':
        parser.error('Use the original .py generator, not the pasted Markdown attachment.')
    try:
        with path.open(encoding='utf-8', newline='') as handle:
            original = handle.read()
        updated = patch_source(original, str(path))
        if updated == original:
            print(f'Already patched: {path}')
            return
        backup = path.with_name(path.name+'.prototype_edge_colors.bak')
        if not backup.exists():
            with backup.open('x', encoding='utf-8', newline='') as handle:
                handle.write(original)
        with path.open('w', encoding='utf-8', newline='') as handle:
            handle.write(updated)
    except (OSError, SyntaxError, ValueError) as error:
        sys.exit(str(error))
    print(f'Updated: {path}\nBackup: {backup}\nRegenerate using --action view, or restart --action serve.\nChoose Edge color > Prototype overlap (top-k) in the top panel.')


if __name__ == '__main__':
    main()
