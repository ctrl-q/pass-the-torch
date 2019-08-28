"""
This class provides autoencoders that are used to learn representations of the data.
Three main classes are provided:
- AutoEncoderABC (A base class for auto encoders)
- VanillaAutoEncoder (An implementation of a regular auto encoder)
- ConvolutionalAutoEncoder (An implementation of a convolutional auto encoder)
"""
import torch
from torch import nn

from .base import PyTorchModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AutoEncoderABC(nn.Module, PyTorchModel):

    def __init__(self, classifier):
        """Abstract base class for autoencoders

        Args:
            classifier (SKLearnModel): model to use if you want to perform classification (optional)
        """
        super().__init__()
        self.criterion = nn.MSELoss()
        self.classifier = classifier

    def run_epoch(self, dataloaders, writer, optimizer,
                  scheduler, phase, curr_epoch, out="score"):
        if self.classifier is None:
            score = super().run_epoch(dataloaders, writer, optimizer,
                                      scheduler, "training", curr_epoch, out)
            return score  # same behavior as always
        elif phase == "validation":  # only run the model once per epoch, there's no separate training and validation phases
            super().run_epoch(dataloaders, writer, optimizer,
                              scheduler, "training", curr_epoch, out)
            dataset = dataloaders["training"].dataset
            labelled_dataset = dataloaders["training_labeled"].dataset

            validset = dataloaders["validation"].dataset
            k = len(validset.map_labels)

            self.eval()
            self.classifier._model.__init__(
                self, dataset, labelled_dataset, k, 4242)  # reset classifier
            original_data, validset.data = validset.data, self.transform(
                validset.data)  # transform validation data

            score = self.classifier.fit(
                {"validation": dataloaders["validation"]}, writer)["score"]

            validset.data = original_data  # restore original data
            self.to(DEVICE)

            return score

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def transform(self, x):
        # Take a numpy array of images and return the embedding
        x = torch.Tensor(x).permute((0, 3, 1, 2)).contiguous()
        return self.encoder(x).detach().numpy()


class Vanilla(AutoEncoderABC):

    def __init__(self, patch_size=32, compression_ratio=2,
                 nb_layers=3, classifier=None):
        """Vanilla AutoEncoder from b2phot baseline

        Args:
            patch_size: the size of the input images,
            compression_ratio: the ratio by which to reduce the input in between layers,
            nb_layers: The number of layers of the encoder and of the decoder,
            classifier: The classifier to use to assign labels to the data,
        """
        super().__init__(classifier)

        en_layers = []
        de_layers = []
        for l in range(nb_layers):
            en_in = int(patch_size / (compression_ratio**(l)))
            en_out = int(patch_size / (compression_ratio**(l + 1)))
            en_layers += [nn.Linear(en_in, en_out), nn.ReLU(True)]
            de_layers += [nn.ReLU(True), nn.Linear(en_out, en_in)]
        de_layers.reverse()
        de_layers[-1] = nn.Sigmoid()

        self.encoder = nn.Sequential(*en_layers)
        self.decoder = nn.Sequential(*de_layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return super().forward(x)

    def transform(self, x):
        x = torch.Tensor(x).permute((0, 3, 1, 2))
        x = x.contiguous().view(x.size(0), -1)
        return self.encoder(x).detach().numpy()


class Convolutional(AutoEncoderABC):
    def __init__(self, encoder_channels=[16, 32, 64, 16], kernel_sizes=[
                 3, 3, 3, 3], batch_norm=True, dropout=0, classifier=None):
        """AutoEncoder with convolutions

        Args:
            encoder_channels (sequence): number of output channels for each convolution layer of the encoder
            kernel_sizes (sequence): kernel size for each convolution layer
            batch_norm (bool): Whether to use Batch Norm
            dropout (float): percentage of the hidden units to dropout at each layer
        """
        super().__init__(classifier)
        assert len(encoder_channels) == len(
            kernel_sizes), "encoder_features and kernel_sizes should have the same length"

        in_channels = 3
        en_layers = []
        de_layers = []
        for i, (out_channels, kernel_size) in enumerate(
                zip(encoder_channels, kernel_sizes)):
            # encoder
            en_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size),
                nn.ReLU(True)
            ])
            if batch_norm:
                en_layers.append(nn.BatchNorm2d(out_channels))
            if dropout:
                en_layers.append(nn.Dropout(dropout, True))

            # decoder
            if dropout:
                de_layers.append(nn.Dropout(dropout, True))
            if batch_norm:
                de_layers.append(nn.BatchNorm2d(in_channels))

            de_layers.extend([
                nn.ReLU(True),
                nn.ConvTranspose2d(out_channels, in_channels, kernel_size)
            ])

            in_channels = out_channels

        de_layers.reverse()

        self.encoder = nn.Sequential(*en_layers)
        self.decoder = nn.Sequential(*de_layers)


class Duplicate(object):
    """Transform used  where input is also the label
    """

    def __call__(self, x):
        """
        Args:
            x: input

        Returns:
            duplicate (tuple): tuple containing x twice
        """
        return x, x
