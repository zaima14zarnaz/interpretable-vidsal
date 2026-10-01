import torch
from types import SimpleNamespace

import patch_to_prototype_assign as p

args = p.parse_args()
args.checkpoint = args.checkpoint.expanduser().resolve()
args.repo_root = args.repo_root.expanduser().resolve()
args.output_root = args.output_root.expanduser().resolve()
if args.projected_checkpoint is None:
    args.projected_checkpoint = args.output_root / "projection" / "projected_best_dhf1k.pth"
else:
    args.projected_checkpoint = args.projected_checkpoint.expanduser().resolve()

state_path = args.output_root / "projection" / "search_state.pt"
saved = p.torch_load(state_path)
if saved.get("partial", False):
    raise RuntimeError(
        f"{state_path} is partial: {saved['windows_processed']}/{saved['total_windows']} windows"
    )

model, checkpoint, container_key, had_module_prefix, _ = p.load_model(args)
states = {
    stage: SimpleNamespace(embeddings=payload["embeddings"])
    for stage, payload in saved["states"].items()
}
p.apply_projection(model, states)
p.save_projected_checkpoint(
    model=model,
    original_checkpoint=checkpoint,
    container_key=container_key,
    had_module_prefix=had_module_prefix,
    destination=args.projected_checkpoint,
)
print(f"Projected checkpoint: {args.projected_checkpoint}")