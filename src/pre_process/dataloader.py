import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    """
    DHF1K / UCF-style video saliency dataset loader.

    Expected layout:
        dataset_dir/<video_name>/images/
        dataset_dir/<video_name>/maps/
        dataset_dir/<video_name>/fixation/

    Each sample returns:
        video_filename, rgb_frame_set, final_sal_map, final_fix_map, n_frames

    Shapes:
        rgb_frame_set: [T, H, W, 3]
        final_sal_map: [H, W]
        final_fix_map: [H, W]

    Important:
        This loader assumes last-frame saliency prediction.
        It loads a window of RGB frames, but only the final frame's saliency
        and fixation maps as supervision.

    HSFI-Net-style sampling:
        If random_train_sampling=True, __len__ is the number of videos and each
        __getitem__ samples one random valid clip start from that video. This
        matches the original HSFI-Net training loader's one-random-clip-per-video
        behavior, while keeping this loader's paths, transforms, and return type.
    """

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(
        self,
        dataset_dir: str,
        window_len: int,
        stride: int = 1,
        joint_transform: Optional[Callable] = None,
        transform_rgb: Optional[Callable] = None,
        transform_sal: Optional[Callable] = None,
        random_train_sampling: bool = False,
    ):
        if window_len < 1:
            raise ValueError(f"window_len must be >= 1, got {window_len}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.dataset_dir = os.path.abspath(dataset_dir)
        self.window_len = window_len
        self.stride = stride
        self.random_train_sampling = random_train_sampling

        # Preferred: one synchronized transform for RGB window + saliency + fixation.
        self.joint_transform = joint_transform

        # Kept for compatibility. Avoid random crop/flip/resize here unless
        # the same operation is already handled by joint_transform.
        self.transform_rgb = transform_rgb
        self.transform_sal = transform_sal

        self.video_dirs: List[str] = []
        self.windows: List[Tuple[str, int]] = []

        # Used only when random_train_sampling=True. Each index corresponds to
        # one video, then __getitem__ randomly selects a valid start for it.
        self.video_random_starts: Dict[str, List[int]] = {}

        # Fix 4: cache frame names instead of listing directory in every __getitem__.
        self.video_frame_names: Dict[str, List[str]] = {}

        for name in sorted(os.listdir(self.dataset_dir)):
            video_path = os.path.join(self.dataset_dir, name)
            if not os.path.isdir(video_path):
                continue

            images_dir = os.path.join(video_path, "images")
            maps_dir = os.path.join(video_path, "maps")
            fixations_dir = os.path.join(video_path, "fixation")

            if (
                not os.path.isdir(images_dir)
                or not os.path.isdir(maps_dir)
                or not os.path.isdir(fixations_dir)
            ):
                continue

            frame_names = self._list_image_names(images_dir)
            if len(frame_names) < window_len:
                continue

            n_frames = len(frame_names)
            valid_starts: List[int] = []

            # HSFI-Net training samples random starts at frame-level granularity,
            # not at the deterministic sliding-window stride.
            candidate_stride = 1 if random_train_sampling else stride
            for start in range(0, n_frames - window_len + 1, candidate_stride):
                final_fname = frame_names[start + window_len - 1]

                # Only index windows with valid final-frame supervision.
                if not self._has_sal_map(maps_dir, final_fname):
                    continue
                if not self._has_fixation_map(fixations_dir, final_fname):
                    continue

                valid_starts.append(start)
                if not random_train_sampling:
                    self.windows.append((name, start))

            if not valid_starts:
                continue

            self.video_dirs.append(name)
            self.video_frame_names[name] = frame_names
            if random_train_sampling:
                self.video_random_starts[name] = valid_starts

        if random_train_sampling:
            if not self.video_dirs:
                raise RuntimeError(
                    f"No valid videos found under {self.dataset_dir}. "
                    "Expected subdirs with 'images/', 'maps/', and 'fixation/'."
                )
        elif not self.windows:
            raise RuntimeError(
                f"No valid video windows found under {self.dataset_dir}. "
                "Expected subdirs with 'images/', 'maps/', and 'fixation/'."
            )

    @staticmethod
    def _natural_key(filename: str):
        # Safer than plain sort if filenames are not zero-padded.
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", filename)
        ]

    @classmethod
    def _list_image_names(cls, images_dir: str) -> List[str]:
        image_files = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith(cls.IMAGE_EXTS)
        ]
        image_files.sort(key=cls._natural_key)
        return image_files

    @staticmethod
    def _has_sal_map(maps_dir: str, frame_name: str) -> bool:
        return os.path.isfile(os.path.join(maps_dir, frame_name))

    @staticmethod
    def _has_fixation_map(fixations_dir: str, frame_name: str) -> bool:
        return os.path.isfile(os.path.join(fixations_dir, frame_name))

    @staticmethod
    def _load_rgb_pil(path: str) -> Image.Image:
        # Fix 6: close file handle immediately after copying image into memory.
        with Image.open(path) as img:
            return img.convert("RGB").copy()

    @staticmethod
    def _load_gray_pil(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("L").copy()

    def _rgb_to_array(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img, dtype=np.uint8)
        if self.transform_rgb is not None:
            arr = self.transform_rgb(arr)
        return arr

    def _sal_to_array(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.transform_sal is not None:
            arr = self.transform_sal(arr)
        return arr

    @staticmethod
    def _fix_to_array(img: Image.Image) -> np.ndarray:
        arr = np.asarray(img, dtype=np.uint8)
        return (arr > 0).astype(np.float32)

    def __len__(self) -> int:
        if self.random_train_sampling:
            return len(self.video_dirs)
        return len(self.windows)

    def __getitem__(self, idx: int):
        if self.random_train_sampling:
            video_filename = self.video_dirs[idx]
            starts = self.video_random_starts[video_filename]
            start = int(np.random.choice(starts))
        else:
            video_filename, start = self.windows[idx]

        video_path = os.path.join(self.dataset_dir, video_filename)
        images_dir = os.path.join(video_path, "images")
        maps_dir = os.path.join(video_path, "maps")
        fixations_dir = os.path.join(video_path, "fixation")

        frame_names = self.video_frame_names[video_filename]
        window_frames = frame_names[start : start + self.window_len]

        if len(window_frames) != self.window_len:
            raise RuntimeError(
                f"Window for video '{video_filename}' has "
                f"{len(window_frames)} frames, expected {self.window_len}."
            )

        final_fname = window_frames[-1]
        final_sal_path = os.path.join(maps_dir, final_fname)
        final_fix_path = os.path.join(fixations_dir, final_fname)

        # Fix 5: this should never fail because __init__ filtered invalid windows.
        # If it does fail, raise instead of silently returning a shorter sample.
        if not os.path.isfile(final_sal_path):
            raise FileNotFoundError(
                f"Missing final saliency map for indexed window: "
                f"{video_filename}/{final_fname}"
            )
        if not os.path.isfile(final_fix_path):
            raise FileNotFoundError(
                f"Missing final fixation map for indexed window: "
                f"{video_filename}/{final_fname}"
            )

        rgb_imgs = [
            self._load_rgb_pil(os.path.join(images_dir, fname))
            for fname in window_frames
        ]
        final_sal = self._load_gray_pil(final_sal_path)
        final_fix = self._load_gray_pil(final_fix_path)

        # Fix 1: one synchronized transform for the whole window and both targets.
        if self.joint_transform is not None:
            rgb_imgs, final_sal, final_fix = self.joint_transform(
                rgb_imgs,
                final_sal,
                final_fix,
            )

        rgb_frames = [self._rgb_to_array(img) for img in rgb_imgs]
        final_sal_map = self._sal_to_array(final_sal)
        final_fix_map = self._fix_to_array(final_fix)

        rgb_frame_set = np.stack(rgb_frames, axis=0)  # [T, H, W, 3]
        n_frames = rgb_frame_set.shape[0]

        if n_frames != self.window_len:
            raise RuntimeError(
                f"Expected {self.window_len} frames, got {n_frames} "
                f"for video '{video_filename}', start={start}."
            )

        if final_sal_map.shape != final_fix_map.shape:
            raise ValueError(
                f"final_sal_map shape {final_sal_map.shape} != "
                f"final_fix_map shape {final_fix_map.shape} "
                f"for video '{video_filename}/{final_fname}'"
            )

        if final_fix_map.max() <= 0.0:
            print(
                f"Warning: empty final fixation map for "
                f"'{video_filename}/{final_fname}'"
            )

        return video_filename, rgb_frame_set, final_sal_map, final_fix_map, n_frames

