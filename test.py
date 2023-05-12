import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import training
from audioUtil import AudioUtil
from model import Model
from soundDs import SoundDS

model = Model()
model.load_state_dict(torch.load(Path('models') / 'model_esp32_youtube.pt'))
model.eval()


ds = SoundDS('..\\sound\\test')
dl = DataLoader(ds, batch_size=1, shuffle=False)
model = Model.load_from_file(Path.cwd() / 'models' / 'model_esp32_youtube.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# # Check that it is on Cuda
# next(model.parameters()).device

training.Training.inference(model, dl, device)


