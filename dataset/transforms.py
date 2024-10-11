import numpy as np
import torch
import librosa
import random


def set_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seeds(42)


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    return new_value


class RandomNoise:
    def __init__(self, min_noise=0.0, max_noise=0.05):  # 0.002, 0.01
        super(RandomNoise, self).__init__()
        self.min_noise = min_noise
        self.max_noise = max_noise

    def addNoise(self, wave):
        noise_val = random.uniform(self.min_noise, self.max_noise)
        noise = torch.from_numpy(np.random.normal(0, noise_val, wave.shape[0]))
        noisy_wave = wave + noise
        return noisy_wave

    def __call__(self, x):
        return self.addNoise(x)


class RandomScale:
    def __init__(self, max_scale: float = 1.25):
        super(RandomScale, self).__init__()
        self.max_scale = max_scale

    @staticmethod
    def random_scale(max_scale: float, signal: torch.Tensor) -> torch.Tensor:
        scaling = np.power(max_scale, np.random.uniform(-1, 1))  # between 1.25**(-1) and 1.25**(1)
        output_size = int(signal.shape[-1] * scaling)
        ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)
        ref1 = ref.clone().type(torch.int64)
        ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
        r = ref - ref1.type(ref.type())
        scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r
        return scaled_signal

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_scale(self.max_scale, x)


class RandomCrop:
    def __init__(self, out_len: int = 44100, train: bool = True):
        super(RandomCrop, self).__init__()
        self.out_len = out_len
        self.train = train

    def random_crop(self, signal: torch.Tensor) -> torch.Tensor:
        if self.train:
            left = np.random.randint(0, signal.shape[-1] - self.out_len)
        else:
            left = int(round(0.5 * (signal.shape[-1] - self.out_len)))
        orig_std = signal.float().std() * 0.5
        output = signal[..., left:left + self.out_len]
        out_std = output.float().std()
        if out_std < orig_std:
            output = signal[..., :self.out_len]
        new_out_std = output.float().std()
        if orig_std > new_out_std > out_std:
            output = signal[..., -self.out_len:]
        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_crop(x) if x.shape[-1] > self.out_len else x


class RandomPadding:
    def __init__(self, out_len: int = 88200, train: bool = True):
        super(RandomPadding, self).__init__()
        self.out_len = out_len
        self.train = train

    def random_pad(self, signal: torch.Tensor) -> torch.Tensor:
        if self.train:
            left = np.random.randint(0, self.out_len - signal.shape[-1])
        else:
            left = int(round(0.5 * (self.out_len - signal.shape[-1])))
        right = self.out_len - (left + signal.shape[-1])
        pad_value_left = signal[..., 0].float().mean().to(signal.dtype)
        pad_value_right = signal[..., -1].float().mean().to(signal.dtype)
        output = torch.cat((
            torch.zeros(signal.shape[:-1] + (left,), dtype=signal.dtype, device=signal.device).fill_(pad_value_left),
            signal,
            torch.zeros(signal.shape[:-1] + (right,), dtype=signal.dtype, device=signal.device).fill_(pad_value_right)
        ), dim=-1)
        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_pad(x) if x.shape[-1] < self.out_len else x


class FrequencyMask:
    def __init__(self, max_width=24, numbers=2):
        super(FrequencyMask, self).__init__()
        self.max_width = max_width
        self.numbers = numbers

    def add_freq_mask(self, wave):
        if len(wave.shape) == 2:  # Spectrogram
            for _ in range(self.numbers):
                max_mask_len = min(self.max_width, wave.shape[0])
                if max_mask_len == 0:
                    continue
                mask_len = random.randint(0, max_mask_len)
                if wave.shape[0] - mask_len > 0:
                    start = random.randint(0, wave.shape[0] - mask_len)
                    wave[start:start + mask_len, :] = 0
        elif len(wave.shape) == 1:  # Raw waveform
            spectrogram = torch.stft(wave, n_fft=1024)
            for _ in range(self.numbers):
                max_mask_len = min(self.max_width, spectrogram.shape[0])
                if max_mask_len == 0:
                    continue
                mask_len = random.randint(0, max_mask_len)
                if spectrogram.shape[0] - mask_len > 0:
                    start = random.randint(0, spectrogram.shape[0] - mask_len)
                    spectrogram[start:start + mask_len, :] = 0
            wave = torch.istft(spectrogram, n_fft=1024)
        return wave

    def __call__(self, wave):
        return self.add_freq_mask(wave)


class TimeMask:
    def __init__(self, max_width=25, numbers=2):
        super(TimeMask, self).__init__()
        self.max_width = max_width
        self.numbers = numbers

    def add_time_mask(self, wave):
        if len(wave.shape) == 2:  # Spectrogram
            for _ in range(self.numbers):
                max_mask_len = min(self.max_width, wave.shape[1])
                if max_mask_len == 0:
                    continue
                mask_len = random.randint(0, max_mask_len)
                if wave.shape[1] - mask_len > 0:
                    start = random.randint(0, wave.shape[1] - mask_len)
                    wave[:, start:start + mask_len] = 0
        elif len(wave.shape) == 1:  # Raw waveform
            spectrogram = torch.stft(wave, n_fft=1024)
            for _ in range(self.numbers):
                max_mask_len = min(self.max_width, spectrogram.shape[1])
                if max_mask_len == 0:
                    continue
                mask_len = random.randint(0, max_mask_len)
                if spectrogram.shape[1] - mask_len > 0:
                    start = random.randint(0, spectrogram.shape[1] - mask_len)
                    spectrogram[:, start:start + mask_len] = 0
            wave = torch.istft(spectrogram, n_fft=1024)
        return wave

    def __call__(self, wave):
        return self.add_time_mask(wave)


class PitchShift:
    def __init__(self, sr=44100, max_steps=2):
        self.sr = sr
        self.max_steps = max_steps

    def __call__(self, x):
        steps = np.random.randint(-self.max_steps, self.max_steps)
        # Convert tensor to numpy array, apply pitch shift, and convert back to tensor
        y = x.numpy()
        y_shifted = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)
        return torch.tensor(y_shifted)
