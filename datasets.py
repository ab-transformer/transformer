import pickle
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch as th

GT_NAMES = {
    "train": "annotation_training.pkl",
    "valid": "annotation_validation.pkl",
    "test": "annotation_test.pkl",
}

SET_SIZE = {"train": 6000, "valid": 2000, "test": 2000}

IMPRESSIONV2_DIR = Path("/impressionv2")
EMBEDDING_DIR = Path("/mbalazsdb")


def load_impressionv2_dataset_all() -> List[th.utils.data.Dataset]:
    """Loads 3 datasets containing embeddings for ImpressionV2 for a specific split. The dataset returns tensors of
    embeddings in the following order: audio, face, text, label. The label is not an embedding but the ground truth
    value. All three datasets will take up 13.3 GB space in the RAM.

    :return: train, valid and test datasets
    """
    return [load_impressionv2_dataset_split(split) for split in ["train", "valid", "test"]]


def load_impressionv2_dataset_split(split: str) -> th.utils.data.Dataset:
    """Loads a dataset containing embeddings for ImpressionV2 for a specific split. The dataset returns tensors of
    embeddings in the following order: audio, face, text, label. The label is not an embedding but the ground truth
    value.

    :param split: Can be either "train", "valid" or "test".
    :return:
    """
    split_dir = IMPRESSIONV2_DIR / split
    gt, target_names = _get_gt(split)
    videos = sorted(gt.keys())
    video_dirs = [split_dir / video for video in videos]
    assert len(videos) == SET_SIZE[split]

    audio_norm = _get_audio(split, video_dirs)
    face_np = _get_face(split, video_dirs)
    text_np = _get_text(split, videos)

    audio_th = th.tensor(audio_norm)
    face_th = th.tensor(face_np)
    text_th = th.tensor(text_np)
    label_th = th.tensor([gt[video] for video in videos])
    return th.utils.data.TensorDataset(audio_th, face_th, text_th, label_th)


def _get_gt(split: str) -> Tuple[Dict, List[str]]:
    gt_file = IMPRESSIONV2_DIR / GT_NAMES[split]

    with open(gt_file, "rb") as f:
        gt_dict = pickle.load(f, encoding="latin1")
    target_names = list(gt_dict.keys())
    target_names.pop(-2)  # remove interview
    sample_names = sorted(list(gt_dict[target_names[0]].keys()))
    gt = {
        Path(sample_name).stem: [
            gt_dict[target_name][sample_name] for target_name in target_names
        ]
        for sample_name in sample_names
    }
    return gt, target_names


def _get_audio(split: str, video_dirs: List[Path]) -> np.ndarray:
    file = IMPRESSIONV2_DIR / f"{split}_audio.pkl"
    if not file.exists():
        audio_np = _create_audio(video_dirs)
        audio_norm = _normalize_audio(audio_np, file, split)
    else:
        with open(file, "rb") as f:
            gt_dict = pickle.load(f, encoding="latin1")
            audio_norm = gt_dict["audio_norm"]
    return audio_norm


def _create_audio(video_dirs: List[Path]) -> np.ndarray:
    audio_list = [
        pd.read_csv(video / "egemaps" / "lld.csv", sep=";") for video in video_dirs
    ]
    audio_list_pad = [
        np.pad(a.to_numpy(), [(0, 1526 - a.shape[0]), (0, 0)]) for a in audio_list
    ]
    audio_np = np.stack(audio_list_pad)
    return audio_np


def _normalize_audio(audio_np: np.ndarray, file: Path, split: str) -> np.ndarray:
    if split == "train":
        mean = np.mean(audio_np, (0, 1))
        std = np.std(audio_np, (0, 1))
        audio_norm = (audio_np - mean) / std
        with open(file, "wb") as f:
            pickle.dump({"audio_norm": audio_norm, "mean": mean, "std": std}, f)
    else:
        with open(IMPRESSIONV2_DIR / f"train_audio.pkl", "rb") as f:
            gt_dict = pickle.load(f, encoding="latin1")
        mean = gt_dict["mean"]
        std = gt_dict["std"]
        audio_norm = (audio_np - mean) / std
        with open(file, "wb") as f:
            pickle.dump({"audio_norm": audio_norm}, f)
    return audio_norm


def _get_text(split: str, videos: List[str]) -> np.ndarray:
    file = IMPRESSIONV2_DIR / f"{split}_text.npy"
    if not file.exists():
        text_np = _create_text(split, videos)
        np.save(file, text_np)
    else:
        text_np = np.load(file)
    return text_np


def _create_text(split: str, videos: List[str]) -> np.ndarray:
    split_dir = EMBEDDING_DIR / "text" / split
    text_list = [np.load(split_dir / f"{video}_bertemd.npy") for video in videos]
    text_np = np.concatenate(text_list)
    return text_np


def _get_face(split: str, video_dirs: List[Path]) -> np.ndarray:
    file = IMPRESSIONV2_DIR / f"{split}_face.npy"
    if not file.exists():
        face_np = _creat_face(video_dirs)
        np.save(file, face_np)
    else:
        face_np = np.load(file)
    return face_np


def _creat_face(video_dirs: List[Path]) -> np.ndarray:
    face_list = [
        np.load(video / "fi_face_resnet18" / "features.npy") for video in video_dirs
    ]
    face_list_pad = [np.pad(a, [(0, 459 - a.shape[0]), (0, 0)]) for a in face_list]
    face_np = np.stack(face_list_pad)
    return face_np
