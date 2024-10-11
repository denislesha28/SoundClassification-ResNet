import random

import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import os
import sys
from functools import partial
import numpy as np
import librosa

import config
from dataset import transforms


def set_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seeds(42)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_extract_zip(url: str, file_path: str):
    import zipfile
    root = os.path.dirname(file_path)
    download_file(url=url, fname=file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)


# create this bar_progress method which is invoked automatically from wget
def download_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


class ESC50(data.Dataset):

    def __init__(self, root, test_folds=frozenset((1,)), subset="train", download=False):
        audio = 'ESC-50-master/audio'
        root = os.path.normpath(root)
        audio = os.path.join(root, audio)
        if subset in {"train", "test"}:
            self.subset = subset
        else:
            raise ValueError
        if not os.path.exists(audio) and download:
            os.makedirs(root, exist_ok=True)
            file_name = 'master.zip'
            file_path = os.path.join(root, file_name)
            url = f'https://github.com/karoldvl/ESC-50/archive/{file_name}'
            download_extract_zip(url, file_path)

        self.root = audio
        temp = sorted(os.listdir(self.root))
        folds = {int(v.split('-')[0]) for v in temp}
        self.test_folds = set(test_folds)
        self.train_folds = folds - test_folds
        train_files = [f for f in temp if int(f.split('-')[0]) in self.train_folds]
        test_files = [f for f in temp if int(f.split('-')[0]) in test_folds]
        assert set(temp) == (set(train_files) | set(test_files))
        if subset == "test":
            self.file_names = test_files
        else:
            self.file_names = train_files

        train = self.subset == "train"
        if train:
            self.wave_transforms = transforms.Compose(
                torch.Tensor,

                transforms.RandomNoise(min_noise=0.002, max_noise=0.01),
                transforms.RandomScale(max_scale=1.25),
                transforms.RandomCrop(out_len=44100, train=True),
                transforms.RandomPadding(out_len=88200, train=True),
                transforms.TimeMask(max_width=25, numbers=2),
                transforms.PitchShift(sr=44100, max_steps=2)

                # transforms.PitchShift(sr=44100, max_steps=2)
            )
            '''
            transforms.RandomScale(),
            transforms.RandomPadding(out_len=220500),
            transforms.RandomCrop(),
            transforms.TimeMask(),
            '''

            self.spec_transforms = transforms.Compose(
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
            )

        else:
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                transforms.RandomPadding(out_len=220500, train=False),
                transforms.RandomCrop(out_len=220500, train=False)
            )

            self.spec_transforms = transforms.Compose(
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
            )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        path = os.path.join(self.root, file_name)
        wave, rate = librosa.load(path, sr=config.sr)

        temp = file_name.split('.')[0]
        class_id = int(temp.split('-')[-1])

        if wave.ndim == 1:
            wave = wave[:, np.newaxis]

        if np.abs(wave.max()) > 1.0:
            wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
        wave = wave.T * 32768.0

        start = wave.nonzero()[1].min()
        end = wave.nonzero()[1].max()
        wave = wave[:, start: end + 1]

        wave_copy = np.copy(wave)
        wave_copy = self.wave_transforms(wave_copy)
        wave_copy.squeeze_(0)

        s = librosa.feature.melspectrogram(y=wave_copy.numpy(),
                                           sr=config.sr,
                                           n_mels=224,
                                           n_fft=4096,
                                           hop_length=308)
        log_s = librosa.power_to_db(s, ref=np.max)
        log_s = self.spec_transforms(log_s)
        spec = log_s

        return file_name, spec, class_id
