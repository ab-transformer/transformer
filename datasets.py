import pickle
import pickle5
from pathlib import Path
from typing import Tuple, Dict, List, Callable

import numpy as np
import pandas as pd
import torch as th
import zarr

GT_NAMES = {
    "train": "annotation_training.pkl",
    "valid": "annotation_validation.pkl",
    "test": "annotation_test.pkl",
}

SET_SIZE = {"train": 6000, "valid": 2000, "test": 2000}

IMPRESSIONV2_DIR = Path("/impressionv2")
EMBEDDING_DIR = Path("/mbalazsdb")

REPORT_IMPRESSIONV2_DIR = Path("/workspace/lld_au_bert")


class ReportImpressionV2DataSet(th.utils.data.Dataset):
    def __init__(self, data, trfs):
        self.trfs = trfs
        self.data = self.call_preprocess(data)

    def call_preprocess(self, data):
        return list(map(self.process_data, data))

    def process_data(self, item):
        a, v, t, label = self.unpack(item)
        if self.trfs is not None:
            a, v, t = self.trfs(a, v, t)
        a = a.astype("float32")
        v = v.astype("float32")
        t = t.astype("float32")
        label = label.astype("float32")
        return a, v, t, label

    def unpack(self, item):
        (a, v, t), label = item
        return a, v, t, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ReportMOSIDataSet(ReportImpressionV2DataSet):
    def call_preprocess(self, data):
        return list(
            map(
                self.process_data,
                zip(data["audio"], data["vision"], data["text"], data["labels"]),
            )
        )

    def unpack(self, item):
        return item[0], item[1], item[2], item[3].flatten()


class Pipeline:
    def __init__(self, callables: List[Callable]):
        self.callables = callables

    def __call__(self, *args):
        data = args
        for c in self.callables:
            data = c(*data)
        return data


class NormAVModalities:
    def __init__(self, a_mean, a_std, v_mean, v_std):
        super().__init__()
        self.a_norm = Normalizing(a_mean, a_std)
        self.v_norm = Normalizing(v_mean, v_std)

    def __call__(self, a, v, t):
        return self.a_norm(a), self.v_norm(v), t


class Normalizing:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean) / self.std


class Padd3Modalities:
    def __init__(self):
        self.a_pad = Padding(1526)
        self.v_pad = Padding(459)
        self.t_pad = Padding(116)

    def __call__(self, a, v, t):
        return self.a_pad(a), self.v_pad(v), self.t_pad(t)


class Padding:
    def __init__(self, pad_to_size):
        self.pad_to_size = pad_to_size

    def __call__(self, data):
        return np.pad(data, [(0, self.pad_to_size - data.shape[0]), (0, 0)])


def load_report_impressionv2_dataset_all(is_norm: bool) -> List[th.utils.data.Dataset]:
    train_ds = load_report_impressionv2_dataset_split("train", is_norm)
    valid_ds = load_report_impressionv2_dataset_split("valid", is_norm)
    test_ds = load_report_impressionv2_dataset_split("test", is_norm)
    return [train_ds, valid_ds, test_ds]


def load_report_impressionv2_dataset_split(
    split: str, is_norm: bool
) -> th.utils.data.Dataset:
    file_name = f"fi_{split}_lld_au_bert.pkl"
    with open(REPORT_IMPRESSIONV2_DIR / file_name, "rb") as f:
        data = pickle.load(f)

    if is_norm:
        norms = np.load("norms.npz")
        trfs = Pipeline([NormAVModalities(**norms), Padd3Modalities()])
    else:
        trfs = Padd3Modalities()
    return ReportImpressionV2DataSet(data, trfs)


def load_report_mosi_dataset_all(is_norm: bool) -> List[th.utils.data.Dataset]:
    file_name = "mosi_of_os_bert.pkl"
    with open(REPORT_IMPRESSIONV2_DIR / file_name, "rb") as f:
        data = pickle5.load(f)

    if is_norm:
        norms = np.load("mosi_norms.npz")
        trfs = Pipeline([NormAVModalities(**norms)])
    else:
        trfs = None
    return [
        ReportMOSIDataSet(data[split], trfs) for split in ["train", "valid", "test"]
    ]


def load_report_mosei_dataset_all(is_norm: bool) -> List[th.utils.data.Dataset]:
    file_name = "mosei_of_os_bert_emotions_18k.pkl"
    with open(REPORT_IMPRESSIONV2_DIR / file_name, "rb") as f:
        data = pickle5.load(f)

    if is_norm:
        norms = np.load("mosei_norms.npz")
        trfs = Pipeline([NormAVModalities(**norms)])
    else:
        trfs = None
    return [
        ReportMOSIDataSet(data[split], trfs) for split in ["train", "valid", "test"]
    ]


def load_report_mosei_sent_dataset_all(is_norm: bool) -> List[th.utils.data.Dataset]:
    file_name = "mosei_of_os_bert_sentiment_18k.pkl"
    with open(REPORT_IMPRESSIONV2_DIR / file_name, "rb") as f:
        data = pickle5.load(f)

    if is_norm:
        norms = np.load("mosei_norms.npz")
        trfs = Pipeline([NormAVModalities(**norms)])
    else:
        trfs = None
    return [
        ReportMOSIDataSet(data[split], trfs) for split in ["train", "valid", "test"]
    ]


class TensorDatasetWithTransformer(th.utils.data.Dataset):
    def __init__(self, tensor_dataset, transform=None):
        self.tensor_dataset = tensor_dataset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.tensor_dataset[index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.tensor_dataset)


class SamplerTransform:
    def __init__(self, srA, srF, srT, is_random=False):
        self.srA = srA
        self.srF = srF
        self.srT = srT
        self.is_random = is_random

    def __call__(self, x):
        audio, face, text, label = x
        al = audio.shape[0]
        fl = face.shape[0]
        tl = text.shape[0]
        assert al == 1526
        assert fl == 459
        assert tl == 60
        if self.srA is None:
            self.srA = al
        if self.srF is None:
            self.srF = fl
        if self.srT is None:
            self.srT = tl
        assert self.srA <= al
        assert self.srF <= fl
        assert self.srT <= tl

        if not self.is_random:
            a_idx = np.linspace(0, al - 1, self.srA, dtype=int)
            f_idx = np.linspace(0, fl - 1, self.srF, dtype=int)
            t_idx = np.linspace(0, tl - 1, self.srT, dtype=int)
        else:
            a_idx = np.random.choice(al - 1, self.srA, replace=False)
            f_idx = np.random.choice(fl - 1, self.srF, replace=False)
            t_idx = np.random.choice(tl - 1, self.srT, replace=False)
            a_idx.sort()
            f_idx.sort()
            t_idx.sort()
        audio_s = audio[
            a_idx,
        ]
        face_s = face[
            f_idx,
        ]
        text_s = text[
            t_idx,
        ]  # 60x768 -> 10x768

        return audio_s, face_s, text_s, label


def load_impressionv2_dataset_all(
    srA=None,
    srF=None,
    srT=None,
    is_random=False,
    audio_emb: str = "lld",
    face_emb: str = "resnet18",
    text_emb: str = "bert",
) -> Tuple[List[th.utils.data.Dataset], List[str]]:
    """Loads 3 datasets containing embeddings for ImpressionV2 for a specific split. The dataset returns tensors of
    embeddings in the following order: audio, face, text, label. The label is not an embedding but the ground truth
    value. All three datasets will take up 13.3 GB space in the RAM.

    :return: train, valid and test datasets in the first list and the target names in the second list
    """
    train_ds, train_target_names = load_impressionv2_dataset_split(
        "train", audio_emb, face_emb, text_emb
    )
    valid_ds, test_target_names = load_impressionv2_dataset_split(
        "valid", audio_emb, face_emb, text_emb
    )
    test_ds, valid_target_names = load_impressionv2_dataset_split(
        "test", audio_emb, face_emb, text_emb
    )

    if (srA is not None) or (srF is not None) or (srT is not None):
        train_ds = TensorDatasetWithTransformer(
            train_ds, SamplerTransform(srA, srF, srT, is_random)
        )
        valid_ds = TensorDatasetWithTransformer(
            valid_ds, SamplerTransform(srA, srF, srT, False)
        )
        test_ds = TensorDatasetWithTransformer(
            test_ds, SamplerTransform(srA, srF, srT, False)
        )

    assert train_target_names == valid_target_names
    assert train_target_names == test_target_names
    return [train_ds, valid_ds, test_ds], train_target_names


def load_impressionv2_dataset_split(
    split: str, audio_emb: str, face_emb: str, text_emb: str
) -> Tuple[th.utils.data.Dataset, List[str]]:
    """Loads a dataset containing embeddings for ImpressionV2 for a specific split. The dataset returns tensors of
    embeddings in the following order: audio, face, text, label. The label is not an embedding but the ground truth
    value.

    :param split: Can be either "train", "valid" or "test".
    :return:
    """
    gt, target_names = _get_gt(split)
    videos = sorted(gt.keys())
    assert len(videos) == SET_SIZE[split]

    audio_embs = {"lld": _get_lld_audio, "wav2vec2": _get_wav2vec2_audio}
    face_embs = {"resnet18": _get_resnet18_face, "ig65m": _get_ig65m_face}
    text_embs = {"bert": _get_bert_text}
    audio_norm = audio_embs[audio_emb](split, videos)
    face_np = face_embs[face_emb](split, videos)
    text_np = text_embs[text_emb](split, videos)

    audio_th = th.tensor(audio_norm)
    face_th = th.tensor(face_np)
    text_th = th.tensor(text_np)
    label_th = th.tensor([gt[video] for video in videos])
    return (
        th.utils.data.TensorDataset(audio_th, face_th, text_th, label_th),
        target_names,
    )


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


def videos2videodirs(split, videos):
    split_dir = IMPRESSIONV2_DIR / split
    return [split_dir / video for video in videos]


# region lld audio
def _get_lld_audio(split: str, videos: List[str]) -> np.ndarray:
    video_dirs = videos2videodirs(split, videos)
    file = IMPRESSIONV2_DIR / f"{split}_audio.pkl"
    if not file.exists():
        audio_np = _create_lld_audio(video_dirs)
        audio_norm = _normalize_lld_audio(audio_np, file, split)
    else:
        with open(file, "rb") as f:
            gt_dict = pickle.load(f, encoding="latin1")
            audio_norm = gt_dict["audio_norm"]
    return audio_norm.astype(np.float32)


def _create_lld_audio(video_dirs: List[Path]) -> np.ndarray:
    audio_list = [
        pd.read_csv(video / "egemaps" / "lld.csv", sep=";") for video in video_dirs
    ]
    audio_list_pad = [
        np.pad(a.to_numpy(), [(0, 1526 - a.shape[0]), (0, 0)]) for a in audio_list
    ]
    audio_np = np.stack(audio_list_pad)
    return audio_np


def _normalize_lld_audio(audio_np: np.ndarray, file: Path, split: str) -> np.ndarray:
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


# endregion

# region wav2vec2 audio
def _get_wav2vec2_audio(split: str, videos: List[str]) -> np.ndarray:
    file = IMPRESSIONV2_DIR / f"{split}_wav2vec2_audio.npy"
    if not file.exists():
        audio_dir = Path("/impressionv2_faces/audio/")
        audio_paths = [audio_dir / f"{video}_wav2vec2.npy" for video in videos]
        audio_np = _create_wav2vec2_audio(audio_paths)
        np.save(file, audio_np)
    else:
        audio_np = np.load(file)
    return audio_np


def _create_wav2vec2_audio(video_paths: List[Path]) -> np.ndarray:
    datas = [np.load(f) for f in video_paths]
    datas = [d.reshape([-1, 768]) for d in datas]
    datas = [np.pad(d, [(0, 764 - d.shape[0]), (0, 0)]) for d in datas]
    return np.stack(datas)


# endregion

# region bert text
def _get_bert_text(split: str, videos: List[str]) -> np.ndarray:
    file = IMPRESSIONV2_DIR / f"{split}_text.npy"
    if not file.exists():
        text_np = _create_bert_text(split, videos)
        np.save(file, text_np)
    else:
        text_np = np.load(file)
    return text_np


def _create_bert_text(split: str, videos: List[str]) -> np.ndarray:
    split_dir = EMBEDDING_DIR / "text" / split
    text_list = [np.load(split_dir / f"{video}_bertemd.npy") for video in videos]
    text_np = np.concatenate(text_list)
    return text_np


# endregion

# region resnet18 face
def _get_resnet18_face(split: str, videos: List[str]) -> np.ndarray:
    video_dirs = videos2videodirs(split, videos)
    file = IMPRESSIONV2_DIR / f"{split}_face.npy"
    if not file.exists():
        face_np = _creat_resnet18_face(video_dirs)
        np.save(file, face_np)
    else:
        face_np = np.load(file)
    return face_np


def _creat_resnet18_face(video_dirs: List[Path]) -> np.ndarray:
    face_list = [
        np.load(video / "fi_face_resnet18" / "features.npy") for video in video_dirs
    ]
    face_list_pad = [np.pad(a, [(0, 459 - a.shape[0]), (0, 0)]) for a in face_list]
    face_np = np.stack(face_list_pad)
    return face_np


# endregion

# region ig65m face
def _get_ig65m_face(split: str, videos: List[str]) -> np.ndarray:
    file = IMPRESSIONV2_DIR / f"{split}_ig65m_face.npy"
    if not file.exists():
        faces_dir = Path("/impressionv2_faces/openface/")
        video_paths = [faces_dir / f"{video}_ig65m.npy" for video in videos]
        face_np = _create_ig65m_face(video_paths)
        np.save(file, face_np)
    else:
        face_np = np.load(file)
    return face_np


def _create_ig65m_face(video_paths: List[Path]) -> np.ndarray:
    video_list = [np.load(video) for video in video_paths]
    video_list_pad = [np.pad(a, [(0, 14 - a.shape[0]), (0, 0)]) for a in video_list]
    return np.stack(video_list_pad)


# endregion

# region rasampled dataset
class NumpyDataset(th.utils.data.Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays
        assert len(np.unique([a.shape[0] for a in self.arrays])) == 1

    def __getitem__(self, index):
        return tuple(th.tensor(a[index]) for a in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]


class FlattenZarrArray:
    def __init__(self, arr):
        self.arr = arr
        self.shape = arr.shape

    def __getitem__(self, idx):
        a = self.arr[idx]
        return np.reshape(a, (a.shape[0], -1))


def load_resampled_impressionv2_dataset_all():
    _, target_names = _get_gt("valid")
    return (
        [
            load_resampled_impressionv2_dataset_split(split)
            for split in ["train", "valid", "test"]
        ],
        target_names,
    )


def load_resampled_impressionv2_dataset_split(split):
    f = zarr.open(str(IMPRESSIONV2_DIR / f"{split}.zarr"))
    return NumpyDataset(f.wav2vec2, FlattenZarrArray(f.r2plus1d), f.bert, f.y)


# endregion
