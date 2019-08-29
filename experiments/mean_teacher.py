"""Mean Teach Experiment
Requires `trainset` , `train_labeled_set` and `validset` variables, of `type torch.utils.data.Dataset`
"""
from .base import PyTorchExperiment
from torch.optim.adam import Adam
from models.resnet import ResNet

from utils.mean_teacher import mt_dataloaders
from models.mean_teacher import MeanTeacherModel
from skopt.optimizer.optimizer import Optimizer

from . import parser, parse_args, mark_best
import os
from ast import literal_eval as le

NO_LABEL = -1
FOLDER = "mean-teacher"
os.makedirs(FOLDER, exist_ok=True)

if __name__ == "__main__":

    parser.add_argument("--clas", type=le, default=0.68,
                        help="Classification correction")
    parser.add_argument("--cons", type=le, default=141,
                        help="consistency correction")
    parser.add_argument("--eps", type=le, default=9,
                        help="epsilon for VAT Loss")
    parser.add_argument("--lbs", type=le, default=1, help="Labeled batch size")
    parser.add_argument("--wd", type=le, default=0.00864,
                        help="Weight decay for Adam")

    [
        (
            datapath,
            max_epoch,
            lr,
            max_patience,
            batch_size,
            trials
        ),
        args
    ] = parse_args(parser)

    [
        classification_correction,
        consistency_correction,
        vat_epsilon,
        labeled_batch_size,
        weight_decay
    ] = [value for _, value in sorted(args.items())]

    opt = Optimizer((
        max_epoch,
        lr,
        max_patience,
        batch_size,
        classification_correction,
        consistency_correction,
        vat_epsilon,
        labeled_batch_size,
        weight_decay
    ))

    results = []

    try:
        for _ in range(trials):
            # Get hyperparameters
            params = opt.ask(strategy='cl_max')
            [
                max_epoch,
                lr,
                max_patience,
                batch_size,
                classification_correction,
                consistency_correction,
                vat_epsilon,
                labeled_batch_size,
                weight_decay
            ] = params
            # Setting up model
            model = MeanTeacherModel(
                ResNet(18, 17), classification_correction, consistency_correction, vat_epsilon)

            # Setting up datasets and dataloaders
            dataloaders = mt_dataloaders(trainset, train_labeled_set, validset, int(
                batch_size), int(labeled_batch_size))

            optimizer = Adam(model._model.parameters(),
                             lr=lr, weight_decay=weight_decay)

            file_name = "-".join([
                str(param if not isinstance(param, float) else round(param, 2))
                for param in params
            ])
            path = os.path.join(FOLDER, file_name + ".tar")
            exp = PyTorchExperiment(path, model, optimizer)

            exp.run(dataloaders, max_epoch, patience=max_patience)
            exp.load("best")
            score = exp.score

            opt.tell(params, score)
            results.append((params, score))

    finally:
        mark_best(results, FOLDER)
