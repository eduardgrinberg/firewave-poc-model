import torch
from torch import nn
from torch.utils.data import random_split

import training
from model import Model
from soundDs import SoundDS


class Training:
    @staticmethod
    def load_data_set():
        ds = SoundDS('sound\\ready')
        # Random split of 80:20 between training and validation
        num_items = len(ds)
        num_train = round(num_items * 0.99)
        num_val = num_items - num_train
        train_ds, val_ds = random_split(ds, [num_train, num_val])
        # Create training and validation data loaders
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

        return train_dl, val_dl

    @staticmethod
    def create_and_train_model(num_epochs):
        (train_dl, val_dl) = Training.load_data_set()
        (model, device) = Training.training(train_dl, num_epochs)
        model.eval()
        acc = Training.inference(model, val_dl, device)
        print(acc)
        return model

    @staticmethod
    def training(train_dl, num_epochs):
        # Create the model and put it on the GPU if available
        model = Model()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # Check that it is on Cuda
        next(model.parameters()).device

        # Loss Function, Optimizer and Scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                        steps_per_epoch=int(len(train_dl)),
                                                        epochs=num_epochs,
                                                        anneal_strategy='linear')

        # Repeat for each epoch
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0

            # Repeat for each batch in the training set
            for i, data in enumerate(train_dl):
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Zero the parameter gradients
                optimizer.zero_grad()

                # inputs = inputs[np.newaxis, ...]

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Keep stats for Loss and Accuracy
                running_loss += loss.item()

                # Get the predicted class with the highest score
                # _, prediction = torch.max(outputs, 1)
                prediction = training.Training.binary_prediction(outputs)
                # Count of predictions that matched the target label
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]

                # if i % 10 == 0:    # print every 10 mini-batches
                #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            # Print stats at the end of the epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        print('Finished Training')

        return model, device

    @staticmethod
    def binary_prediction(output):
        fire_predictions = output[:,1]
        predictions = torch.zeros(fire_predictions.shape[0], dtype=int)
        predictions[fire_predictions > 0] = 1

        return predictions

    @staticmethod
    def inference(model, val_dl, device):
        correct_prediction = 0
        total_prediction = 0
        correct_prediction_sig = 0
        total_prediction_sig = 0
        correct_prediction_sig2 = 0
        total_prediction_sig2 = 0
        correct_prediction_bg = 0
        total_prediction_bg = 0

        # Disable gradient updates
        with torch.no_grad():
            for data in val_dl:
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(device), data[1].to(device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Get predictions
                outputs = model(inputs)

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)
                prediction2 = training.Training.binary_prediction(outputs)

                correct_prediction_sig += ((prediction == labels) & (labels == torch.ones(labels.shape[0]))).sum().item()
                total_prediction_sig += (labels == torch.ones(labels.shape[0])).sum().item()

                correct_prediction_sig2 += ((prediction2 == labels) & (labels == torch.ones(labels.shape[0]))).sum().item()
                total_prediction_sig2 += (labels == torch.ones(labels.shape[0])).sum().item()

                correct_prediction_bg += ((prediction == labels) & (labels == torch.zeros(labels.shape[0]))).sum().item()
                total_prediction_bg += (labels == torch.zeros(labels.shape[0])).sum().item()

                # Count of predictions that matched the target label
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]

        acc_sig = correct_prediction_sig / total_prediction_sig
        print(f'Signal Accuracy: {acc_sig:.2f}, Total items: {total_prediction_sig}')

        acc_sig2 = correct_prediction_sig2 / total_prediction_sig2
        print(f'Binary Signal Accuracy: {acc_sig2:.2f}, Total items: {total_prediction_sig2}')

        acc_bg = correct_prediction_bg / total_prediction_bg
        print(f'Background Accuracy: {acc_bg:.2f}, Total items: {total_prediction_bg}')

        acc = correct_prediction / total_prediction
        print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

        return acc
