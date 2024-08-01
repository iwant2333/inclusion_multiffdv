import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable, Any, Union, Tuple

import einops
import numpy as np
import scipy as sp
import torch
import torchaudio
from einops import rearrange

from torch import Tensor
from torch.nn import functional as F, Identity
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from utils import read_json, read_video, padding_video, padding_audio, resize_video, iou_with_anchors, ioa_with_anchors

@dataclass
class Metadata:
    file: str

class AVDF1M_test(Dataset):
    def __init__(self, subset: str = "test", root: str = "data", frame_padding: int = 512,
                 max_duration: int = 40, fps: int = 25, return_file_name: bool = False,):
        self.subset = subset
        self.root = root
        self.video_padding = frame_padding
        self.audio_padding = int(frame_padding / fps * 16000)
        self.max_duration = max_duration
        self.return_file_name = return_file_name
        self.txt_path = os.path.join(self.root, "valset_label.txt")
        self.filename = []
        with open(self.txt_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                self.filename.append(line.strip())

    def __getitem__(self, index: int) -> List[Tensor]:
        filename = self.filename[index]
        # filepath = os.path.join(self.root, 'unzip', self.subset, 'test_part4', filename)
        filepath = os.path.join(self.root, 'valset', filename)
        video, audio, _ = read_video(filepath)
        video = padding_video(video, target=self.video_padding)
        audio = padding_audio(audio, target=self.audio_padding)
        video = rearrange(resize_video(video, (96, 96)), "t c h w -> c t h w")
        audio = self._get_log_mel_spectrogram(audio)
        outputs = [video, audio]
        if self.return_file_name:
            outputs.append(filename)

        return outputs

    @staticmethod
    def _get_log_mel_spectrogram(audio: Tensor) -> Tensor:
        ms = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64)
        spec = torch.log(ms(audio[:, 0]) + 0.01)
        assert spec.shape == (64, 2048), "Wrong log mel-spectrogram setup in Dataset"
        return spec

    def __len__(self) -> int:
        return len(self.filename)

# if __name__ == '__main__':
#     test_dataset = AVDF1M_test("test", "/root/zhanyi/data/AVdp1m/")
#     test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=8, sampler=SequentialSampler(test_dataset))
#     print(len(test_dataloader))
