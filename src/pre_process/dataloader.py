import os
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    """
    Video saliency dataset loader for UCF-style layouts.

    Each video lives in ``dataset_dir/<video_filename>/`` with:
      - ``images/``  RGB frames
      - ``maps/``    saliency maps (same filenames as images)

    Each sample is a temporal window of up to ``window_len`` consecutive frames.
    Videos shorter than ``window_len`` are skipped.

    Missing saliency maps: if a non-final frame in the window has no map, a
    zero dummy map is used. If the last frame in the window has no map, that
    frame is omitted from the window (windows of length 1 with a missing map
    are skipped entirely).

    Returns per index:
      video_filename, rgb_frame_set, sal_map_set, n_frames
    """

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(
        self,
        dataset_dir: str,
        window_len: int,
        stride: int = 1,
        transform_rgb: Optional[Callable] = None,
        transform_sal: Optional[Callable] = None,
    ):
        if window_len < 1:
            raise ValueError(f"window_len must be >= 1, got {window_len}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.dataset_dir = os.path.abspath(dataset_dir)
        self.window_len = window_len
        self.stride = stride
        self.transform_rgb = transform_rgb
        self.transform_sal = transform_sal

        self.video_dirs: List[str] = []
        self.windows: List[Tuple[str, int]] = []  # (video_filename, start_index)

        for name in sorted(os.listdir(self.dataset_dir)):
            video_path = os.path.join(self.dataset_dir, name)
            if not os.path.isdir(video_path):
                continue

            images_dir = os.path.join(video_path, "images")
            maps_dir = os.path.join(video_path, "maps")
            if not os.path.isdir(images_dir) or not os.path.isdir(maps_dir):
                continue

            frame_names = self._list_image_names(images_dir)
            if not frame_names:
                continue

            self.video_dirs.append(name)
            n_frames = len(frame_names)
            if n_frames < window_len:
                continue

            for start in range(0, n_frames - window_len + 1, stride):
                last_fname = frame_names[start + window_len - 1]
                last_has_map = self._has_sal_map(maps_dir, last_fname)
                if not last_has_map and window_len == 1:
                    continue
                self.windows.append((name, start))

        if not self.windows:
            raise RuntimeError(
                f"No valid video windows found under {self.dataset_dir}. "
                "Expected subdirs with 'images/' and 'maps/'."
            )

    @staticmethod
    def _list_image_names(images_dir: str) -> List[str]:
        image_files = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith(DatasetLoader.IMAGE_EXTS)
        ]
        image_files.sort()
        return image_files

    @staticmethod
    def _has_sal_map(maps_dir: str, frame_name: str) -> bool:
        return os.path.isfile(os.path.join(maps_dir, frame_name))

    def _dummy_sal_map(self, height: int, width: int) -> np.ndarray:
        return np.zeros((height, width), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def _load_rgb(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        if self.transform_rgb is not None:
            arr = self.transform_rgb(arr)
        return arr

    def _load_sal(self, path: str) -> np.ndarray:
        sal = Image.open(path).convert("L")
        arr = np.asarray(sal, dtype=np.float32) / 255.0
        if self.transform_sal is not None:
            arr = self.transform_sal(arr)
        return arr

    def __getitem__(self, idx: int):
        video_filename, start = self.windows[idx]
        video_path = os.path.join(self.dataset_dir, video_filename)
        images_dir = os.path.join(video_path, "images")
        maps_dir = os.path.join(video_path, "maps")

        frame_names = self._list_image_names(images_dir)
        end = min(start + self.window_len, len(frame_names))
        window_frames = frame_names[start:end]

        rgb_frames = []
        sal_maps = []
        for offset, fname in enumerate(window_frames):
            is_last_in_window = offset == len(window_frames) - 1
            has_map = self._has_sal_map(maps_dir, fname)

            if not has_map and is_last_in_window:
                continue

            rgb = self._load_rgb(os.path.join(images_dir, fname))
            rgb_frames.append(rgb)
            if has_map:
                sal_maps.append(self._load_sal(os.path.join(maps_dir, fname)))
            else:
                sal_maps.append(self._dummy_sal_map(rgb.shape[0], rgb.shape[1]))

        if not rgb_frames:
            raise RuntimeError(
                f"Window {idx} for video '{video_filename}' produced no frames "
                f"(start={start}, window_len={self.window_len})."
            )

        n_frames = len(rgb_frames)
        rgb_frame_set = np.stack(rgb_frames, axis=0)  # [T, H, W, 3]
        sal_map_set = np.stack(sal_maps, axis=0)    # [T, H, W]

        return video_filename, rgb_frame_set, sal_map_set, n_frames
