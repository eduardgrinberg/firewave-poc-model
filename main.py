from pathlib import Path

import torch
from torch.utils.data import DataLoader

import training
from model import Model
from soundDs import SoundDS
from training import Training

model = Training.create_and_train_model(3)

torch.save(model.state_dict(), Path.cwd() / 'models' / 'model.pt')

ds = SoundDS('sound\\test')
dl = DataLoader(ds, batch_size=1, shuffle=False)
model = Model.load_from_file(Path.cwd() / 'models' / 'model.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# # Check that it is on Cuda
# next(model.parameters()).device

training.Training.inference(model, dl, device)
