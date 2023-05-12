from pathlib import Path

import torch

from training import Training

model = Training.create_and_train_model(3)

torch.save(model.state_dict(), Path.cwd() / 'models' / 'model_tmp.pt')