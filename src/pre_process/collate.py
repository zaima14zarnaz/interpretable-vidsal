import torch


def _as_float_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(x).float()


def _rgb_clip_to_tchw(rgb) -> torch.Tensor:
    """
    Accept DatasetLoader rgb clips as:
        [T, H, W, 3] uint8/float
        [T, 3, H, W] float
    and return [T, 3, H, W] float.
    """
    rgb = _as_float_tensor(rgb)
    if rgb.dim() != 4:
        raise ValueError(
            f"Expected rgb clip to be 4D [T, H, W, 3] or [T, 3, H, W], got {tuple(rgb.shape)}"
        )

    if rgb.shape[1] == 3:
        rgb = rgb
    elif rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            f"Expected rgb clip channels on dim 1 or -1, got {tuple(rgb.shape)}"
        )

    if rgb.numel() > 0 and rgb.max() > 2.0:
        rgb = rgb / 255.0
    return rgb.contiguous()


def _supervision_map_to_hw(x) -> torch.Tensor:
    """Accept [H, W] or [1, H, W] and return [H, W]."""
    x = _as_float_tensor(x)
    if x.dim() == 2:
        return x
    if x.dim() == 3 and x.shape[0] == 1:
        return x.squeeze(0)
    raise ValueError(
        f"Expected supervision map shape [H, W] or [1, H, W], got {tuple(x.shape)}"
    )


def _rgb_clip_hw(rgb) -> tuple[int, int]:
    rgb = _as_float_tensor(rgb)
    if rgb.dim() != 4:
        raise ValueError(f"Expected 4D rgb clip, got {tuple(rgb.shape)}")
    if rgb.shape[1] == 3:
        return int(rgb.shape[2]), int(rgb.shape[3])
    if rgb.shape[-1] == 3:
        return int(rgb.shape[1]), int(rgb.shape[2])
    raise ValueError(f"Unsupported rgb clip shape {tuple(rgb.shape)}")


def video_saliency_collate_fn(batch):
    """
    Collate batches from DatasetLoader.

    DatasetLoader returns:
        rgb_frame_set: [T, H, W, 3] uint8/float32 in [0, 255] or [0, 1]
        final_sal_map: [H, W] float32 in [0, 1]
        final_fix_map: [H, W] float32 binary {0, 1}

    Also supports pre-normalized rgb clips shaped [T, 3, H, W].

    Returns:
        video_filenames: list[str]
        rgb_batch:       [B, T, 3, H, W] float32 in [0, 1] unless already normalized
        sal_batch:       [B, 1, H, W] float32 in [0, 1]
        fix_batch:       [B, 1, H, W] float32 binary {0, 1}
        n_frames:        [B] int64, valid frame count per sample
        valid_mask:      [B, T] bool, True for real frames
    """
    video_filenames, rgb_list, sal_list, fix_list, n_frames_list = zip(*batch)

    b = len(batch)
    t_max = max(int(n) for n in n_frames_list)
    h_max = max(_rgb_clip_hw(rgb)[0] for rgb in rgb_list)
    w_max = max(_rgb_clip_hw(rgb)[1] for rgb in rgb_list)

    rgb_batch = torch.zeros(b, t_max, 3, h_max, w_max, dtype=torch.float32)
    sal_batch = torch.zeros(b, 1, h_max, w_max, dtype=torch.float32)
    fix_batch = torch.zeros(b, 1, h_max, w_max, dtype=torch.float32)
    valid_mask = torch.zeros(b, t_max, dtype=torch.bool)
    n_frames = torch.tensor(n_frames_list, dtype=torch.int64)

    for i, (rgb, sal, fix, n) in enumerate(
        zip(rgb_list, sal_list, fix_list, n_frames_list)
    ):
        n = int(n)
        rgb = _rgb_clip_to_tchw(rgb)
        t, _, h, w = rgb.shape
        rgb_batch[i, :n, :, :h, :w] = rgb[:n]

        sal_map = _supervision_map_to_hw(sal)
        fix_map = _supervision_map_to_hw(fix)
        sh, sw = sal_map.shape
        fh, fw = fix_map.shape
        if (sh, sw) != (fh, fw):
            raise ValueError(
                f"Saliency/fixation shape mismatch in sample {i}: "
                f"{(sh, sw)} vs {(fh, fw)}"
            )

        sal_batch[i, 0, :sh, :sw] = sal_map
        fix_batch[i, 0, :fh, :fw] = fix_map
        valid_mask[i, :n] = True

    fix_batch = (fix_batch > 0).float()

    if fix_batch.shape != sal_batch.shape:
        raise ValueError(
            f"fix_batch shape {tuple(fix_batch.shape)} != "
            f"sal_batch shape {tuple(sal_batch.shape)}"
        )
    if fix_batch.numel() > 0:
        fix_min = float(fix_batch.min())
        fix_max = float(fix_batch.max())
        if fix_min < 0.0 or fix_max > 1.0:
            raise ValueError(
                f"fix_batch values must be in [0, 1], got min={fix_min}, max={fix_max}"
            )
        unique_fix = torch.unique(fix_batch)
        if not torch.all(torch.isin(unique_fix, unique_fix.new_tensor([0.0, 1.0]))):
            raise ValueError(
                f"fix_batch must be binary {{0, 1}}, got unique values {unique_fix.tolist()}"
            )

    return (
        list(video_filenames),
        rgb_batch,
        sal_batch,
        fix_batch,
        n_frames,
        valid_mask,
    )
