import torch


def video_saliency_collate_fn(batch):
    """
    Collate batches from DatasetLoader.

    Pads shorter windows to the max ``n_frames`` in the batch (<= window_len).

    Returns:
        video_filenames: list[str]
        rgb_frame_set:   [B, T, H, W, 3] float32 in [0, 1]
        sal_map_set:     [B, T, H, W] float32 in [0, 1]
        n_frames:        [B] int64, valid frame count per sample
        valid_mask:      [B, T] bool, True for real frames
    """
    video_filenames, rgb_list, sal_list, n_frames_list = zip(*batch)

    b = len(batch)
    t_max = max(int(n) for n in n_frames_list)
    h_max = max(rgb.shape[1] for rgb in rgb_list)
    w_max = max(rgb.shape[2] for rgb in rgb_list)

    rgb_batch = torch.zeros(b, t_max, h_max, w_max, 3, dtype=torch.float32)
    sal_batch = torch.zeros(b, t_max, h_max, w_max, dtype=torch.float32)
    valid_mask = torch.zeros(b, t_max, dtype=torch.bool)
    n_frames = torch.tensor(n_frames_list, dtype=torch.int64)

    for i, (rgb, sal, n) in enumerate(zip(rgb_list, sal_list, n_frames_list)):
        n = int(n)
        t, h, w, _ = rgb.shape
        rgb_batch[i, :n, :h, :w] = torch.from_numpy(rgb).float() / 255.0
        sal_batch[i, :n, :h, :w] = torch.from_numpy(sal).float()
        valid_mask[i, :n] = True

    return (
        list(video_filenames),
        rgb_batch,
        sal_batch,
        n_frames,
        valid_mask,
    )
