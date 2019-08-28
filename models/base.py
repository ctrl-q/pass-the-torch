"""
Contains Skeletons for Pytorch models and for Sklearn models.
This allows to easily try different learning algorithms.
"""

import time
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PyTorchModel:
    def __init__(self):
        """Base class for all PyTorch models"""
        # Your model should change this criterion
        self.criterion = nn.modules.loss._Loss()

    def fit(self, dataloaders, writer, num_epochs, optimizer,
            scheduler, patience, curr_epoch, best_score):
        """Trains the model

        Args:
            dataloaders (dict of torch.utils.data.DataLoader): dict of dataloaders returning a tuple (input, label)
            for each input
            writer (tensorboardX.SummaryWriter): writer for logging progress
            num_epochs (int): maximum number of epochs to train for
            optimizer (torch.optim.Optimizer): optimizer for the loss
            scheduler (torch.optim.lr_scheduler.{_LRscheduler, ReduceLROnPlateau}): learning rate scheduler
            patience (int): maximum number of consecutive epochs without improvement. Must be <= `num_epochs` (optional)
            curr_epoch (int): current training epoch if loading checkpoint
            best_score (float): current best score if loading checkpoint

        Returns:
            checkpoint (dict): current training state
        """
        if type(self.criterion).__name__ == "_Loss":
            raise NotImplementedError("criterion has to be changed")

        bad_epochs = 0  # consecutive epochs with no improvement
        checkpoint = {}
        try:
            while curr_epoch < num_epochs and bad_epochs < patience:
                # always loop phases in the same order
                for phase in sorted(dataloaders):
                    # we only need to store the validation score
                    score = self.run_epoch(
                        dataloaders, writer, optimizer, scheduler, phase, curr_epoch)
                curr_epoch += 1

                if score > best_score:
                    bad_epochs = 0
                    best_score = score
                    checkpoint["best"] = self.get_checkpoint(
                        optimizer, curr_epoch, score)
                else:
                    bad_epochs += 1
        finally:
            checkpoint["last"] = self.get_checkpoint(
                optimizer, curr_epoch, score)
            checkpoint.setdefault("best", checkpoint["last"])
            return checkpoint

    def get_checkpoint(self, optimizer, curr_epoch, score):
        """Returns the current state

        Args:
            optimizer (torch.optim.Optimizer): optimizer for the loss
            curr_epoch (int): current training epoch
            score (float): current score

        Returns:
            checkpoint (dict): training state
        """
        return {
            "model_state_dict": deepcopy(self.state_dict()),
            "optimizer_state_dict": deepcopy(optimizer.state_dict()),
            "epoch": curr_epoch,
            "score": score
        }

    def predict(self, dataloaders, writer, optimizer, phase):
        return self.run_epoch(dataloaders, writer, optimizer,
                              None, phase, None, out="preds")

    def run_epoch(self, dataloaders, writer, optimizer,
                  scheduler, phase, curr_epoch, out="score"):
        """Trains the model for one epoch

        Args:
            dataloaders (dict of torch.utils.data.DataLoader): dict of dataloaders returning a tuple (input, label) for each input
            writer (tensorboardX.SummaryWriter): writer for logging progress
            optimizer(torch.optim.Optimizer): optimizer for the loss
            scheduler (torch.optim.lr_scheduler.{_LRscheduler, ReduceLROnPlateau}): learning rate scheduler (optional)
            phase (str): either 'training' or 'validation'
            curr_epoch (int): current epoch
            out (str): what to return. either 'score' or 'preds'

        Returns:
            score: Return the score achieved by the model at the end of this epoch
        """
        phases = ("training", "validation")
        assert phase in phases, f"phase must be one of {phases}"

        self.train(phase == "training")
        is_autoencoder = None
        # Save labels and predictions for scoring
        n = len(dataloaders[phase].dataset)
        all_labels = torch.tensor([], device=DEVICE)
        all_preds = torch.tensor([], device=DEVICE)

        running_loss = 0.0
        start = time.time()

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            if is_autoencoder is None:  # i.e. only run this for first minibatch
                is_autoencoder = inputs.size() == labels.size(
                ) and torch.all(inputs == labels).item()

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "training"):
                outputs = self(inputs)
                if is_autoencoder:
                    outputs = outputs.view(inputs.size())
                else:
                    labels = labels.squeeze().to(torch.int64)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == "training":
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                batch_size = inputs.size(0)

                running_loss += loss.item() * batch_size
                if not is_autoencoder:
                    all_preds = torch.cat((all_preds, preds.float()))
                    all_labels = torch.cat((all_labels, labels.float()))

        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        if out == "preds":
            return all_preds

        duration = time.time() - start

        epoch_loss = running_loss / n

        if is_autoencoder:
            scalars = {
                "loss": epoch_loss
            }
            # 1 / epoch_loss so that higher values are better (for early
            # stopping in self.fit())
            score = 1 / epoch_loss

        else:
            epoch_acc = np.mean(all_preds == all_labels)
            score = f1_score(all_labels, all_preds, average="weighted")
            scalars = {
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "f1 score": score
            }

        if phase == "training":
            scalars["duration"] = duration

        for tag, scalar in scalars.items():
            writer.add_scalar(" ".join([phase, tag]),
                              scalar, global_step=curr_epoch)

        return score


class SKLearnModel:
    def __init__(self, model):
        """Base class for all SKLearn models

        Args:
            model: actual sklearn model to run
        """
        self._model = model

    def fit(self, dataloaders, writer, num_epochs=None, optimizer=None,
            scheduler=None, patience=None, curr_epoch=None, best_score=0, out="score"):
        """Fits the underlying sklearn model

        Args:
            dataloaders (dict of torch.utils.data.DataLoader): dict of dataloaders
            writer (tensorboardX.SummaryWriter): writer for logging progress

        All the other arguments are ignored and kept only for compatibility with ModelABC.fit()

        Returns:
            checkpoint (dict): current state
        """
        for phase in sorted(dataloaders):
            dataset = dataloaders[phase].dataset
            all_inputs = dataset.data
            all_labels = dataset.targets
            if phase == "training":
                shuffle = np.random.permutation(range(len(dataset)))
                all_inputs = all_inputs[shuffle]
                all_labels = all_labels[shuffle]

                start = time.time()
                self._model.fit(all_inputs, all_labels)
                duration = time.time() - start
                writer.add_scalar("training duration", duration)

            all_preds = self._model.predict(all_inputs)
            if out == "preds":
                return all_preds
            acc = np.mean(all_preds == all_labels)
            score = f1_score(all_labels, all_preds, average="weighted")

            for tag, scalar in zip(["accuracy", "F1 score"], [acc, score]):
                writer.add_scalar(f"{phase} {tag}", scalar)

            checkpoint = {
                "model": self._model,
                "score": score
            }

        return checkpoint

    def predict(self, dataloaders, writer):
        return self.fit(dataloaders, writer, out="preds")

    def __str__(self):
        return str(self._model)
