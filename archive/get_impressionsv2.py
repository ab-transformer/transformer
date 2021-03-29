# wasn't used

import os
import wget
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from zipfile import ZipFile

IMPRESSIONSV2_DIR = Path("data/impressionv2")
os.makedirs(IMPRESSIONSV2_DIR)
URL = "http://158.109.8.102/FirstImpressionsV2/"


def _unzip_encrypted(file_dir, file_name, pwd):
    with ZipFile(file_dir / file_name) as zf:
        zf.extractall(file_dir, pwd=pwd)


def _download(file_name, local_dir):
    url = URL + file_name
    local = local_dir / file_name
    wget.download(url, str(local))
    return local


def _download_extract(file_name, local_dir):
    local = _download(file_name, local_dir)
    if local.suffix == '.zip':
        shutil.unpack_archive(local, local_dir)


def _download_extract_encrypted(file_name, local_dir, pwd):
    _download(file_name, local_dir)
    _unzip_encrypted(local_dir, file_name, pwd)


def get_impressionsv2():
    _download_extract("eth_gender_annotations_dev.csv", IMPRESSIONSV2_DIR)
    _download_extract("eth_gender_annotations_test.csv", IMPRESSIONSV2_DIR)
    _download_extract("predicted_attributes_caucasian.zip", IMPRESSIONSV2_DIR)

    for split in ["train", "val,", "test"]:
        _get_split(split)


def _get_split(split):
    n_video_zips = 6 if split == "train" else 2
    n_internal_video_zips = 75 if split == "train" else 25
    split_long = {"train": "training",
                  "val": "validation",
                  "test": "test"}[split]
    local_dir = IMPRESSIONSV2_DIR / split
    os.makedirs(local_dir, exist_ok=True)
    download_extract_split = partial(_download_extract, local_dir=local_dir)

    if split == "train":
        files = (f"{split}-transcription.zip", f"{split}-annotation.zip")
    elif split == "val":
        files = (f"{split}-transcription.zip", f"{split}-annotation-e.zip")
    elif split == "test":
        files = (f"{split}-transcription-e.zip", f"{split}-annotation-e.zip")
    else:
        raise ValueError("Split muss be either 'train', 'val' or 'test'!")

    download_extract_test_pkl = partial(_download_extract_encrypted, local_dir=local_dir, pwd=b'zeAzLQN7DnSIexQukc9W')
    f = download_extract_test_pkl if split == "test" else download_extract_split
    with ThreadPoolExecutor(6) as executor:
        executor.map(f, files)

    if split == "val":
        download_extract_test_pkl(files[1])

    def download_videos(i):
        download_extract_split(f'{split}-{i}{"e" if split == "test" else ""}.zip')

    with ThreadPoolExecutor(6) as executor:
        executor.map(download_videos, list(range(1, n_video_zips + 1)))

    def extract_dev(i):
        local_zip = local_dir / f'{split}-{i}{"e" if split == "test" else ""}.zip'
        shutil.unpack_archive(local_zip, local_dir)

    def extract_test(i):
        file_name = f'test-{i}e.zip'
        _unzip_encrypted(local_dir, file_name, b'zeAzLQN7DnSIexQukc9W')
        d = local_dir / f'test-{i}'
        for f in os.listdir(d):
            shutil.move(d / f, local_dir / f"{f}_ext")

    f = extract_test if split == "test" else extract_dev
    with ThreadPoolExecutor(6) as executor:
        executor.map(f, list(range(1, n_video_zips + 1)))

    def extract_internal(i):
        local_zip = local_dir / f'{split_long}80_{i:02d}.zip'
        shutil.unpack_archive(local_zip, local_dir)

    def extract_internal_test(i):
        _unzip_encrypted(local_dir, f'test80_{i:02d}.zip_ext',
                         b'.chalearnLAPFirstImpressionsSECONDRoundICPRWorkshop2016.')
        extract_internal(i)

    f = extract_internal_test if split == "test" else extract_internal
    with ThreadPoolExecutor(6) as executor:
        executor.map(f, list(range(1, n_internal_video_zips + 1)))

    # clean up
    v_ext = ".mp4"
    p_ext = '.pkl'
    zip_ext = '.zip'
    zipe_ext = '.zip_ext'
    os.makedirs(IMPRESSIONSV2_DIR / split / 'video/', exist_ok=True)
    os.makedirs(IMPRESSIONSV2_DIR / split / 'meta/', exist_ok=True)
    for filename in os.listdir(IMPRESSIONSV2_DIR / split):
        if filename.endswith(v_ext):
            shutil.move(IMPRESSIONSV2_DIR / split / filename, IMPRESSIONSV2_DIR / split / 'video' / filename)
        elif filename.endswith(p_ext):
            shutil.move(IMPRESSIONSV2_DIR / split / filename, IMPRESSIONSV2_DIR / split / 'meta' / filename)
        elif filename.endswith(zip_ext) or filename.endswith(zipe_ext):
            os.remove(IMPRESSIONSV2_DIR / split / filename)
    if split == 'test':
        p = IMPRESSIONSV2_DIR / split / 'test-1'
        if p.exists():
            p.rmdir()
        p = IMPRESSIONSV2_DIR / split / 'test-2'
        if p.exists():
            p.rmdir()


if __name__ == "__main__":
    get_impressionsv2()
