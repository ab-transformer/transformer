import pickle
from pathlib import Path
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


def load_gt(split):
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


def load_split(split):
    split_dir = IMPRESSIONV2_DIR / split
    gt, target_names = load_gt(split)
    videos = sorted(gt.keys())
    video_dirs = [split_dir / video for video in videos]
    assert len(videos) == SET_SIZE[split]

    file = IMPRESSIONV2_DIR / f"{split}_audio.pkl"
    if not file.exists():
        audio_list = [
            pd.read_csv(video / "egemaps" / "lld.csv", sep=";") for video in video_dirs
        ]
        audio_list_pad = [
            np.pad(a.to_numpy(), [(0, 1526 - a.shape[0]), (0, 0)]) for a in audio_list
        ]
        audio_np = np.stack(audio_list_pad)
        mean = np.mean(audio_np, (0, 1))
        std = np.std(audio_np, (0, 1))
        audio_norm = (audio_np - mean) / std
        with open(file, "wb") as f:
            pickle.dump({"audio_norm": audio_norm, "mean": mean, "std": std}, f)
    else:
        with open(file, "rb") as f:
            gt_dict = pickle.load(f, encoding="latin1")
            audio_norm = gt_dict["audio_norm"]

    file = IMPRESSIONV2_DIR / f"{split}_face.npy"
    if not file.exists():
        face_list = [
            np.load(video / "fi_face_resnet18" / "features.npy") for video in video_dirs
        ]
        face_list_pad = [np.pad(a, [(0, 459 - a.shape[0]), (0, 0)]) for a in face_list]
        face_np = np.stack(face_list_pad)
        np.save(file, face_np)
    else:
        face_np = np.load(file)

    file = IMPRESSIONV2_DIR / f"{split}_text.npy"
    if not file.exists():
        split_dir = EMBEDDING_DIR / "text" / split
        text_list = [np.load(split_dir / f"{video}_bertemd.npy") for video in videos]
        text_np = np.concatenate(text_list)
    else:
        text_np = np.load(file)

    audio_th = th.tensor(audio_norm)
    face_th = th.tensor(face_np)
    text_th = th.tensor(text_np)
    label_th = th.tensor([gt[video] for video in videos])
    return th.utils.data.TensorDataset(audio_th, face_th, text_th, label_th)
