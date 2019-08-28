import numpy as np
import time
import contextlib
import torch
import torch.nn.functional as F
from torch import nn
from copy import deepcopy

from sklearn.metrics.classification import f1_score
from .base import PyTorchModel

NO_LABEL = -1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MeanTeacherModel(PyTorchModel):
    def __init__(self, model, classification_correction=1.0,
                 consistency_correction=1.0, eps=1, ema_decay=0.99):
        """Mean teacher implementation

        Args:
            model (nn.Module): actual model to be trained
            classification_correction: Classification correction factor (default: 1.0),
            consistency_correction: Consistency correction factor (default: 1.0),
            eps: Epsilon of VAT Loss (default: 1),
            ema_decay: Weight decay of teacher model  (default: 0.99),
        """
        self._model = model.to(DEVICE)
        self.summon_teacher()
        self.classification_correction = classification_correction
        self.consistency_correction = consistency_correction
        self.eps = eps
        self.ema_decay = ema_decay
        self.criterion = None  # calculated inside self.fit()

    def __str__(self):
        return f"\n(student): {self._model}\n\n(teacher): {self.teacher}"

    def to(self, device=None):
        """Dummy function for compatibility with PyTorchExperiment"""
        return self

    def softmax_mse_loss(self, input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss

        Args:
            input_logits: predicted logits
            target_logits: true values of the logits

        Returns:
            the sum over all examples. Divide by the batch size afterwards
            if you want the mean.
        """
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1).detach()
        num_classes = input_logits.size(1)
        return F.mse_loss(input_softmax, target_softmax) / num_classes

    def run_epoch(self, dataloaders, writer, optimizer,
                  scheduler, phase, curr_epoch, out="score"):
        phases = ("training", "validation")
        assert phase in phases, f"phase must be one of {phases}"

        if phase == "training":
            return self.train_epoch(
                dataloaders[phase], writer, optimizer, scheduler, curr_epoch)
        elif phase == "validation":
            return self.validate_epoch(
                dataloaders[phase], writer, curr_epoch, out)

    def train_epoch(self, train_loader, writer,
                    optimizer, scheduler, curr_epoch):

        all_preds = torch.tensor([], device=DEVICE)
        all_labels = torch.tensor([], device=DEVICE)
        n = len(train_loader.dataset)

        class_criterion = torch.nn.CrossEntropyLoss(ignore_index=NO_LABEL)
        vat_criterion = VATLoss(eps=self.eps)
        consistency_criterion = self.softmax_mse_loss

        # Switch to train mode
        self._model.train()
        self.teacher.train()

        running_loss = 0.0
        start = time.time()
        for (inputs, teacher_outputs), labels in train_loader:

            inputs = inputs.to(DEVICE)
            teacher_outputs = teacher_outputs.to(DEVICE)
            labels = labels.to(DEVICE).to(torch.int64).squeeze()

            optimizer.zero_grad()
            # Forward propagation for both Student and Teacher models
            with torch.set_grad_enabled(True):
                outputs = self._model(inputs)

                teacher_outputs = self.teacher(teacher_outputs)

                # Loss measurement for both Student and Teacher models
                class_loss = class_criterion(outputs, labels)

                # VAT Loss
                vat_loss = vat_criterion(self._model, inputs)

                # Consistency loss
                consistency_loss = self.consistency_correction * \
                    consistency_criterion(outputs, teacher_outputs)

                # Total loss measurement
                loss = self.classification_correction * class_loss + consistency_loss + vat_loss
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size

                _, preds = torch.max(teacher_outputs, 1)
                # Only store predictions for labeled data
                label_idx = labels != NO_LABEL
                all_preds = torch.cat((all_preds, preds[label_idx].float()))
                all_labels = torch.cat((all_labels, labels[label_idx].float()))

                assert not (torch.isnan(loss) or loss > 1e5), 'Loss explosion: {}'.format(loss.data)

                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                self.update_teacher(self.ema_decay, curr_epoch)

        duration = time.time() - start
        epoch_loss = running_loss / n

        all_preds = all_preds.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        epoch_acc = np.mean(all_preds == all_labels)
        score = f1_score(all_preds, all_labels, average='weighted')

        scalars = {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "f1 score": score,
            "duration": duration
        }

        for tag, scalar in scalars.items():
            writer.add_scalar(f"training {tag}",
                              scalar, global_step=curr_epoch)

        return score

    def validate_epoch(self, eval_loader, writer, curr_epoch, out):
        all_preds = torch.tensor([], device=DEVICE)
        all_labels = torch.tensor([], device=DEVICE)
        self.teacher.eval()

        for (inputs, _), labels in eval_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).to(torch.int64).squeeze()

            with torch.set_grad_enabled(False):
                # compute output
                output = self.teacher(inputs)

                _, preds = torch.max(output, 1)
                # Only store predictions for labeled data
                label_idx = labels != NO_LABEL
                all_preds = torch.cat((all_preds, preds[label_idx].float()))
                all_labels = torch.cat((all_labels, labels[label_idx].float()))

        all_preds = all_preds.cpu().numpy()
        if out == "preds":
            return all_preds

        all_labels = all_labels.cpu().numpy()
        epoch_acc = np.mean(all_preds == all_labels)
        score = f1_score(all_preds, all_labels, average='weighted')

        scalars = {
            "accuracy": epoch_acc,
            "f1 score": score,
        }

        for tag, scalar in scalars.items():
            writer.add_scalar(
                f"validation {tag}", scalar, global_step=curr_epoch)

        return score

    def update_teacher(self, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(
                self.teacher.parameters(), self._model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "student": self._model.state_dict(destination, prefix, keep_vars),
            "teacher": self.teacher.state_dict(destination, prefix, keep_vars)
        }

    def load_state_dict(self, state_dict, strict=True):
        self._model.load_state_dict(state_dict["student"], strict)
        self.teacher.load_state_dict(state_dict["teacher"], strict)

# Mean teacher utility functions
# Code taken from
# https://github.com/CuriousAI/mean-teacher/tree/master/pytorch/mean_teacher

    def summon_teacher(self):
        """
        Exponentially moving average model creation

        """
        self.teacher = deepcopy(self._model).to(DEVICE)
        for param in self.teacher.parameters():
            param.detach()

# VAT utility functions
# Code taken from https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
