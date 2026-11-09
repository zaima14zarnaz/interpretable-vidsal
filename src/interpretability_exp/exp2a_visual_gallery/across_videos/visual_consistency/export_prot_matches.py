#!/usr/bin/env python3
"""Export every patch whose top-N similarity candidates contain prototype_group.

Edit prototype_group below, then run:
  python export_prototype_group_patches.py --src-root /path/to/ExplainableSaliency/src
Or: python export_prototype_group_patches.py --group 154 133 180 --stage stage4 --top-n 20 --stride 5

Input: DATA_DIR/<video_filename>/images/<frame_filename> (DHF1K/UCF layout).
Scan the first MAX_FRAMES physical RGB frames globally, with natural filename
ordering, without saliency/fixation filtering. For each frame, use a causal
window ending at that frame: up to train.WINDOW_LEN preceding/current frames.
Early frames use shorter windows; windows never cross videos. Every source
frame is counted once, including nonmatching frames. This deliberately covers
frames 1..5000, rather than only last frames of nonoverlapping eval windows.

Defaults match the reviewed best_dhf1k.pth architecture, with topk_soft and k=8.
Only backbone + selected stage concept creation are run: these are the exact
pre-decoder operations used to construct assignments in model.forward. Decoder
saliency and GT annotations are not needed. Requires the updated topk_soft code.
Prototype IDs are local to --stage; ALL listed IDs must occur in a patch's top-N.
--top-n ranges from 1 to 20 (default 8). At N=8 use original valid assigned IDs;
otherwise rank the full-bank pre-softmax similarity logits. N>8 candidates are
not all active evidence: the model itself still uses its original top-8 slots.
Neither checkpoint weights nor model top_k are changed by the search setting.

--stride / --frame-stride samples frame offsets 0,S,2S,... within EACH video.
--max-frames caps the first source frames globally BEFORE stride sampling; with
5000 source frames and stride=5, roughly 1000 frames are evaluated (video starts
reset stride). Every source frame still supplies contiguous temporal context.
Metadata distinguishes scanned source frames and evaluated sampled frames.

OUT_DIR contains only prototype_group.json and <vid_fname>_<frame_no>.jpg files.
Each matching frame gets ONE sheet: the original frame with ALL matching cells
boxed/numbered in red, plus EVERY matching patch crop with corresponding numbers.
Boxes show concept-grid cell footprints, not full backbone receptive fields.
No separate crops, per-patch JPEGs, CSVs, or saliency maps. Existing files and
folders in OUT_DIR are deleted before export to avoid mixing groups/checkpoints/runs.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import sys
from collections import deque
from pathlib import Path

# EDIT THESE SETTINGS. The group must contain 1..TOP_N distinct IDs from one stage.
prototype_group = [100]
# [88] is for blue patches especially water bodies
OUT_DIR = Path('OUT_DIR')
DATA_DIR = Path('/data/quantization/zaima/videosal_datasets/dhf1k/train')
CHECKPOINT = Path('/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/20261019_104444/epoch_050.pth')
MAX_FRAMES = 100000
STAGE = 'stage4'
TOP_N = 8  # Search ranking size, 1..20; does not change model top_k=8.
FRAME_STRIDE = 1
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def natural_key(path):
    return tuple((0,int(x)) if x.isdigit() else (1,x.casefold())
                 for x in re.split(r'(\d+)',Path(path).name))


def discover_videos(root):
    videos=[];names=set()
    for folder in sorted(root.iterdir(),key=natural_key):
        if not folder.is_dir() or not (folder/'images').is_dir():
            continue
        frames=sorted((p for p in (folder/'images').iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS),key=natural_key)
        if not frames:
            continue
        # Preserve video filename (including .mp4 if present); sanitize only unsafe characters.
        video=re.sub(r'[^\w.-]','_',folder.name)
        if video in names:
            raise ValueError('Video filenames collide after output sanitization: '+video)
        names.add(video)
        stems=[p.stem for p in frames]
        if len(set(stems))!=len(stems):
            raise ValueError('Duplicate frame stems would overwrite JPEGs in '+str(folder))
        videos.append((video,frames))
    if not videos:
        raise ValueError('No RGB frames: expected DATA_DIR/<video_filename>/images/.')
    return videos


def validate_group(group,top_n=TOP_N):
    if type(top_n) is not int or not 1<=top_n<=20:
        raise ValueError('--top-n must be an integer in [1,20].')
    if not isinstance(group,list) or not 1<=len(group)<=top_n:
        raise ValueError(f'Set prototype_group to 1..{top_n} IDs, or increase --top-n (maximum 20).')
    if any(type(x) is not int or not 0<=x<512 for x in group) or len(set(group))!=len(group):
        raise ValueError('Prototype IDs must be distinct integers in [0,511].')
    return list(group)


def matching_patches(concept,group,bank_size=512,top_n=TOP_N):
    import torch
    validate_group(group,top_n)
    if top_n>bank_size or any(p>=bank_size for p in group):
        raise ValueError('Search ranking/group exceeds the stage prototype bank size.')
    ids=concept['active_visual_prototype_indices']
    valid=concept['visual_validity_mask'].bool()
    print(ids)
    if ids.ndim!=5 or ids.shape[:2]!=(1,1) or ids.shape[-1]!=8 or valid.shape!=ids.shape or not valid.all():
        raise ValueError('Expected actual last-frame [1,1,H,W,8] assignments, all valid. Install updated topk_soft code.')
    if (ids<0).any() or (ids>=bank_size).any() or not (ids.sort(-1).values.diff(dim=-1)>0).all():
        raise ValueError('Invalid/duplicate prototype IDs in top-8 assignments.')
    h,w=ids.shape[2:4]
    # Use original slots; do not recompute rankings after selection.
    logits=concept['visual_concept_logits'].reshape(h*w,bank_size)
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite prototype similarity logits.')
    if not torch.equal(ids.reshape(-1,8),logits.topk(8,dim=-1).indices):
        raise ValueError('Active slots differ from the model top-8 similarity ranking.')
    candidates=ids[0,0] if top_n==8 else logits.topk(top_n,dim=-1).indices.reshape(h,w,top_n)
    selected=torch.ones((h,w),dtype=torch.bool,device=ids.device)
    for proto in group:
        selected &= candidates.eq(proto).any(-1)
    return selected.nonzero().cpu().tolist(),(h,w)


def patch_box(row,col,grid,size):
    h,w=grid;image_w,image_h=size
    left=col*image_w//w;top=row*image_h//h
    right=min(image_w,max(left+1,math.ceil((col+1)*image_w/w)))
    bottom=min(image_h,max(top+1,math.ceil((row+1)*image_h/h)))
    return left,top,right,bottom


def contact_sheet(frame,patches,grid,video,frame_no,group,stage,tile_size=160,columns=6,top_n=TOP_N):
    from PIL import Image,ImageDraw,ImageOps,ImageFont
    if not patches:
        raise ValueError('Do not render a frame with no matching patches.')
    gap=10;header_h=64;caption_h=24
    try:font=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:font=ImageFont.load_default()
    # Expand columns if necessary to stay within JPEG's 65,500 pixel dimension limit.
    max_rows=max(1,(60000-1000)//(tile_size+caption_h+gap))
    columns=min(len(patches),max(columns,math.ceil(len(patches)/max_rows)))
    width=max(640,columns*(tile_size+gap)+gap)
    if width>60000:
        raise ValueError('Requested sheet exceeds JPEG size limits; use smaller --tile-size.')
    overview=frame.copy()
    overview.thumbnail((width-2*gap,720),Image.Resampling.LANCZOS)
    # Draw boxes AFTER resizing so every red box remains visible in the overview.
    mark=ImageDraw.Draw(overview)
    sx=overview.width/frame.width;sy=overview.height/frame.height
    boxes=[]
    for number,(row,col) in enumerate(patches,1):
        box=patch_box(row,col,grid,frame.size);boxes.append(box)
        x0,y0,x1,y1=box
        marked=(math.floor(x0*sx),math.floor(y0*sy),max(math.floor(x0*sx),math.ceil(x1*sx)-1),max(math.floor(y0*sy),math.ceil(y1*sy)-1))
        mark.rectangle(marked,outline='red',width=2)
        mark.text((marked[0]+2,marked[1]+2),str(number),font=font,fill='red',stroke_width=1,stroke_fill='white')
    crop_start=header_h+overview.height+gap
    rows=math.ceil(len(patches)/columns)
    height=crop_start+rows*(tile_size+caption_h+gap)+gap
    if height>65000:
        raise ValueError('Contact sheet exceeds JPEG height limit.')
    sheet=Image.new('RGB',(width,height),(32,32,32));draw=ImageDraw.Draw(sheet)
    draw.text((gap,8),f'{video} | frame {frame_no} | {stage} | top-{top_n} | IDs: {group}',font=font,fill='white')
    draw.text((gap,34),f'{len(patches)} matching cells | grid {grid[0]}x{grid[1]} | red numbers identify crops',font=font,fill='white')
    sheet.paste(overview,((width-overview.width)//2,header_h))
    for i,((row,col),box) in enumerate(zip(patches,boxes)):
        x=gap+(i%columns)*(tile_size+gap);y=crop_start+(i//columns)*(tile_size+caption_h+gap)
        # Preserve aspect ratio, enlarge the actual crop, and pad; never distort it.
        crop=ImageOps.pad(frame.crop(box),(tile_size,tile_size),method=Image.Resampling.LANCZOS,color=(0,0,0))
        sheet.paste(crop,(x,y));draw.rectangle((x,y,x+tile_size-1,y+tile_size-1),outline='white')
        draw.text((x,y+tile_size+4),f'#{i+1}  row={row} col={col}',font=font,fill='white')
    return sheet


def build_model(backbone_device, head_device):
    import train as train_cfg
    from model.model import ExplainableVidSalModel

    # Matches the user's supplied baseline construction, including deprecated args.
    return ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=False, freeze_backbone=train_cfg.FREEZE_BACKBONE,
        backbone_gradient_checkpointing=False, input_format="BTCHW", resize_to=(224, 384),
        concept_dim=128, num_concepts=512, concept_hidden_dim=256, saliency_hidden_dim=256,
        top_k=8, max_source_patches=64, tau_pi=0.5, tau_alpha=0.07, tau_concept=0.07,
        concept_residual_weight=0.0, last_transition_only=True,
        use_rgb_refinement=False, use_feature_refinement=False, output_activation="none",
        return_details=True, use_subpatch_head=True, subpatch_factor=4,
        subpatch_residual_scale=0.5, use_temporal_transition_aggregation=True,
        temporal_aggregation_hidden_channels=128, temporal_aggregation_temperature=1.0,
        visual_concept_on=train_cfg.VISUAL_CONCEPT_ON,
        temporal_concepts_on=train_cfg.TEMPORAL_CONCEPTS_ON,
        visual_concept_logit_scale=train_cfg.VISUAL_CONCEPT_LOGIT_SCALE,
        visual_concept_residual_weight=1.0, use_temporal_feature_infusion=True,
        visual_assignment_mode="topk_soft",
        use_shared_concept_activations=True,
        prototype_bottleneck_strength=train_cfg.PROTOTYPE_BOTTLENECK_STRENGTH,
        prototype_application_position=train_cfg.PROTOTYPE_APPLICATION_POSITION,
        fine_unary_mask_strength=getattr(train_cfg, "FINE_UNARY_MASK_STRENGTH", 1.0),
        stage2_backbone_fusion_enabled=getattr(train_cfg, "STAGE2_BACKBONE_FUSION_ENABLED", True),
    ).to_split_devices(backbone_device, head_device)


def write_metadata(path,data):
    temp=path.with_suffix('.json.part')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def clear_out_dir(args):
    out=args.out_dir.resolve()
    if out.exists() and not out.is_dir():
        raise ValueError(f'OUT_DIR exists but is not a directory: {out}')
    protected=[Path('/').resolve(),Path.home().resolve(),Path(args.src_root).resolve(),
               args.data_dir.resolve(),args.checkpoint.resolve(),Path(__file__).resolve()]
    if any(out==p or out in p.parents for p in protected):
        raise ValueError(f'Refusing to clear OUT_DIR {out}: it is or contains code, data, checkpoint, or home.')
    if out.exists():
        entries=list(out.iterdir())
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():shutil.rmtree(entry)
            else:entry.unlink()
        if entries:print(f'Removed {len(entries)} existing entries from {out}')


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    import train as cfg
    group=validate_group(args.group if args.group is not None else prototype_group,args.top_n)
    if type(args.frame_stride) is not int or args.frame_stride<1:
        raise ValueError('--stride must be a positive integer.')
    clear_out_dir(args)
    videos=discover_videos(args.data_dir)
    total=min(args.max_frames,sum(len(frames) for _,frames in videos))
    cfg.set_seed(args.seed)
    if torch.cuda.is_available():
        if args.gpu_id>=torch.cuda.device_count():raise ValueError('Requested GPU is not visible.')
        device=torch.device(f'cuda:{args.gpu_id}')
    else:device=torch.device('cpu')
    model=build_model(device,device)
    before=args.checkpoint.stat()
    checkpoint=torch.load(str(args.checkpoint),map_location='cpu',weights_only=False)
    if not isinstance(checkpoint,dict) or 'model_state_dict' not in checkpoint:
        raise ValueError('Checkpoint must contain model_state_dict.')
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    del checkpoint
    model.eval();model.allow_eval_concept_losses=False
    concept_module=model.concept_creations[args.stage]
    if concept_module.top_k!=8 or concept_module.visual_assignment_mode!='topk_soft':
        raise ValueError('Requires topk_soft with eight valid slots.')
    window_len=cfg.WINDOW_LEN if args.window_len is None else args.window_len
    if window_len<1:raise ValueError('Window length must be positive.')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    meta={'prototype_group':group,'stage':args.stage,'top_k':8,'model_top_k':8,
          'match_top_n':args.top_n,'frame_stride':args.frame_stride,'assignment_mode':'topk_soft',
          'matching_policy':'ALL group IDs must occur in the top-N full-bank similarity ranking; N=8 uses actual valid slots',
          'search_changes_model_assignments':False,
          'stride_policy':'frame offsets 0,S,2S,... per video; global source-frame cap applied before stride; contiguous context',
          'checkpoint':str(args.checkpoint.resolve()),'data_dir':str(args.data_dir.resolve()),
          'frame_order':'natural video-folder order, then natural RGB-frame filename order',
          'frame_number':'source RGB filename stem, preserved in output name',
          'temporal_context':'causal trailing window including current frame; short prefixes at video start',
          'window_len':window_len,'max_frames':args.max_frames,'available_frames_scanned':total,
          'scanned_frames':0,'processed_frames':0,'matched_frames':0,'matched_patches':0,'status':'incomplete',
          'box_definition':'concept-grid cell mapped to original RGB coordinates; not full backbone receptive field',
          'source_files':{name:str(Path(args.src_root)/name) for name in ['model/model.py','model/concept_creation.py']}}
    meta_path=args.out_dir/'prototype_group.json';write_metadata(meta_path,meta)
    enabled=not args.no_amp and cfg._amp_enabled(device)
    dtype=cfg._amp_dtype(device)
    with tqdm(total=total,desc='RGB frames',unit='frame') as bar:
        for video,frames in videos:
            context=deque(maxlen=window_len);shape=None
            for frame_offset,frame_path in enumerate(frames):
                if meta['scanned_frames']>=total:break
                with Image.open(frame_path) as source:frame=source.convert('RGB')
                array=np.array(frame,dtype=np.uint8,copy=True)
                if shape is not None and shape!=array.shape:
                    raise ValueError('Frame dimensions change within video '+video)
                shape=array.shape;context.append(array)
                meta['scanned_frames']+=1;bar.update(1)
                if frame_offset%args.frame_stride:
                    if meta['scanned_frames']%100==0:write_metadata(meta_path,meta)
                    continue
                # Match project collate: RGB float32 in [0,1], no ImageNet
                # normalization here (the backbone performs its own normalization).
                rgb=torch.from_numpy(np.stack(context)).float().div_(255).permute(0,3,1,2).unsqueeze(0)
                with torch.inference_mode(),torch.autocast(device.type,enabled=enabled,dtype=dtype):
                    x=model._normalize_video_layout(rgb).to(model.input_device)
                    features=model.backbone.forward_features(x)
                    selected_features=features[args.stage].to(model.output_device)
                    selected_features=model._resize_feature_for_concepts(selected_features,args.stage)
                    result=concept_module(selected_features,return_losses=False,collect_gate_debug=False)
                    patches,grid=matching_patches(result,group,int(concept_module.visual_concepts.shape[0]),args.top_n)
                if patches:
                    name=f'{video}_{frame_path.stem}.jpg'
                    destination=args.out_dir/name
                    if destination.exists():raise ValueError('Output filename collision: '+name)
                    sheet=contact_sheet(frame,patches,grid,video,frame_path.stem,group,args.stage,args.tile_size,args.columns,args.top_n)
                    sheet.save(destination,format='JPEG',quality=95,subsampling=0)
                    meta['matched_frames']+=1;meta['matched_patches']+=len(patches)
                    del sheet
                meta['processed_frames']+=1
                bar.set_postfix(matches=meta['matched_patches'],sheets=meta['matched_frames'])
                if meta['scanned_frames']%100==0:write_metadata(meta_path,meta)
                del features,selected_features,result,x,rgb
            if meta['scanned_frames']>=total:break
    after=args.checkpoint.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during export; metadata remains incomplete.')
    meta['status']='complete';write_metadata(meta_path,meta)
    print(f"Scanned {meta['scanned_frames']} source frames, evaluated {meta['processed_frames']} sampled frames (stride={args.frame_stride}, top-N={args.top_n}); saved ALL {meta['matched_patches']} matching patches in {meta['matched_frames']} sheets to {args.out_dir.resolve()}")
    return meta


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src-root',type=Path)
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--data-dir',type=Path,default=DATA_DIR)
    p.add_argument('--out-dir',type=Path,default=OUT_DIR)
    p.add_argument('--group',nargs='+',type=int,help='Override prototype_group = [] in this file.')
    p.add_argument('--stage',choices=['stage1','stage2','stage3','stage4'],default=STAGE)
    p.add_argument('--max-frames',type=int,default=MAX_FRAMES)
    p.add_argument('--top-n',type=int,default=TOP_N,help='Compare the group against top 1..20 similarity candidates; model top_k remains 8.')
    p.add_argument('--stride','--frame-stride',dest='frame_stride',type=int,default=FRAME_STRIDE,
                   help='Evaluate every S-th source frame per video, within the first --max-frames source frames.')
    p.add_argument('--window-len',type=int)
    p.add_argument('--gpu-id',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tile-size',type=int,default=160)
    p.add_argument('--columns',type=int,default=6)
    p.add_argument('--no-amp',action='store_true')
    a=p.parse_args()
    if min(a.max_frames,a.tile_size,a.columns,a.frame_stride)<1 or a.gpu_id<0 or (a.window_len is not None and a.window_len<1):
        p.error('Frame/window/tile/column/stride counts must be positive; GPU ID nonnegative.')
    try:validate_group(a.group if a.group is not None else prototype_group,a.top_n)
    except ValueError as exc:p.error(str(exc))
    if not a.data_dir.is_dir() or not a.checkpoint.is_file():p.error('Dataset directory/checkpoint does not exist.')
    if a.src_root is None:
        a.src_root=next((r for r in [*Path(__file__).resolve().parents,Path.cwd(),Path.cwd()/'src'] if (r/'train.py').is_file()),None)
    if a.src_root is None or not (a.src_root/'train.py').is_file():p.error('Pass --src-root /path/to/ExplainableSaliency/src.')
    sys.path.insert(0,str(a.src_root.resolve()))
    return a


if __name__=='__main__':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    run(parse_args())
