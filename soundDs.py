import os
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset

from audioUtil import AudioUtil


# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
        self.duration = 4000

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.files)

        # ----------------------------

    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        file = Path(self.path) / self.files[idx]
        aud = torchaudio.load(file)
        class_id = 1 if Path(file).stem.startswith('fire') else 0
        dur_aud = AudioUtil.pad_trunc(aud, self.duration)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)

        return sgram, class_id
