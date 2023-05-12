import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

import training
from audioUtil import AudioUtil
from model import Model
from soundDs import SoundDS

ds = SoundDS('sound\\ready_esp32_44100_32-16')
dl = DataLoader(ds, batch_size=1, shuffle=False)
model = Model.load_from_file(Path.cwd() / 'models' / 'model_tmp.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

predictions, correct_predictions = training.Training.inference(model, dl, device)

fpr, tpr, thresholds = roc_curve(correct_predictions, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


