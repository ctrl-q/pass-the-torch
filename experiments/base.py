import os
import pickle

import numpy as np
import torch
import tensorboardX

from models.base import SKLearnModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SummaryWriter(tensorboardX.SummaryWriter):
    def add_text(self, tag, text_string, global_step=None, walltime=None):
        """Prints to console before running tensorboardX.SummaryWriter.add_text()"""
        message = (tag, ":", text_string)
        if global_step is not None:
            message = ("Global step", global_step, ",") + message
        print(*message)
        return super().add_text(tag, text_string, global_step, walltime)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        """Prints to console before running tensorboardX.SummaryWriter.add_scalar()"""
        message = (tag, ":", scalar_value)
        if global_step:
            message = ("Global step", global_step, ",") + message
        print(*message)
        return super().add_scalar(tag, scalar_value, global_step, walltime)


class ExperimentABC:
    def __init__(self, path, model, optimizer, scheduler=None,
                 file_suffix=None, state="best", save_to_disk=True):
        """Base Experiment class

        Args:
            path (str, path-like): path the model will be loaded from (if it exists) and saved to
            model (Model): model to run
            optimizer (torch.optim.Optimizer): optimizer for the loss
            scheduler (torch.optim.lr_scheduler.{_LRscheduler, ReduceLROnPlateau}): learning rate scheduler (optional)
            file_suffix (str): suffix to add to the saved file (optional)
            state (str): which state to load, if the file exists. either 'best' or 'last' (optional)
            save_to_disk (bool) : Whether the experiment should be saved to disk on completion (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.curr_epoch = 0
        self.score = 0.0
        self.save_to_disk = save_to_disk

        if file_suffix:
            path, ext = os.path.splitext(path)
            path += file_suffix
            path = "".join([path, ext])
        self.path = path
        self.writer = SummaryWriter(os.path.join(
            "runs", os.path.splitext(path)[0]))

        try:
            self.load(state)
        except FileNotFoundError:  # experiment not run before
            pass

        # Set random seed
        np.random.seed(4242)
        torch.manual_seed(4242)
        torch.cuda.manual_seed_all(4242)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def load(self, state):
        """Loads the experiment's state

        Args:
            state (str): which state to load, if the file exists. either 'best' or 'last'
        """
        checkpoint = torch.load(self.path, map_location=DEVICE)
        checkpoint = checkpoint[state]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.score = checkpoint["score"]

    def predict(self, dataloaders, phase):
        return self.model.predict(
            dataloaders, self.writer, self.optimizer, phase)

    def run(self, dataloaders, num_epochs, patience=None):
        """Runs the experiment

        Args:
            dataloaders (dict of torch.utils.data.DataLoader): dict of dataloaders returning a tuple (input, label) for each input
            num_epochs (int): maximum number of epochs to train for
            patience (int): maximum number of consecutive epochs without improvement. Must be <= `num_epochs` (optional)
        """
        if patience is None:
            patience = num_epochs
        assert patience <= num_epochs, "patience cannot be more than num_epochs"
        # Log configuration
        for tag, value in vars(self).items():
            self.writer.add_text(tag, str(value))

        try:
            checkpoint = self.model.fit(dataloaders, self.writer, num_epochs,
                                        self.optimizer, self.scheduler, patience, self.curr_epoch, self.score)
        finally:
            if self.save_to_disk:
                self.save(checkpoint)

    def save(self, checkpoint):
        """Saves the experiment's state

        Args:
            checkpoint (dict): current training state
        """
        return torch.save(checkpoint, self.path)


class PyTorchExperiment(ExperimentABC):
    def __init__(self, path, model, optimizer, scheduler=None,
                 file_suffix=None, state="best"):
        """Experiment class for PyTorch models"""
        super().__init__(path, model, optimizer, scheduler, file_suffix)
        self.model = self.model.to(DEVICE)


class SKLearnExperiment(ExperimentABC):
    def __init__(self, path, model, file_suffix=None):
        """Experiment class for sklearn models

        Args:
            path (str, path-like): path the model will be loaded from (if it exists) and saved to
            model (ModelABC): model to run
            file_suffix (str): suffix to add to the saved file (optional)
        """
        super().__init__(path, model, None, None, file_suffix)
        self.model = SKLearnModel(model)

    def load(self, state=None):
        """Loads the experiment's state

        Args:
            state (NoneType): ignored (kept for compatibility with ExperimentABC)
        """
        with open(self.path, "rb") as f:
            checkpoint = pickle.load(f)
        self.model = checkpoint["model"]
        self.score = checkpoint["score"]

    def save(self, checkpoint):
        """Saves the experiment's state

        Args:
            checkpoint (dict): current training state
        """
        with open(self.path, "wb") as f:
            return pickle.dump(checkpoint, f)
