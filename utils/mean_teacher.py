"""
Utility functions/classes for the mean teacher model
"""
import itertools
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

NO_LABEL = -1


def mt_dataloaders(unlabeled_dataset, train_dataset,
                   valid_dataset, batch_size=100, labeled_batch_size=10):
    # Need to lace training dataset with unlabeled examples
    """
    Data Loader for the mean teacher

    Args:
        unlabeled_dataset: The unlabeled dataset to be loaded
        train_dataset: the train dataset
        valid_dataset: the valid dataset
        batch_size: the batch size to use for the unlabeled dataset
        labeled_batch_size: the batch size to use for the labeled dataset

    Returns:
        dataloaders: a dictionary of data loaders for the train and validation dataset
    """
    concat_train_dataset = train_dataset + unlabeled_dataset

    n_train_labeled = len(train_dataset)
    labeled_idx = list(range(n_train_labeled))
    unlabeled_idx = list(range(n_train_labeled, len(concat_train_dataset)))

    batch_sampler = TwoStreamBatchSampler(
        unlabeled_idx, labeled_idx, batch_size, labeled_batch_size)

    train_loader = DataLoader(concat_train_dataset,
                              batch_sampler=batch_sampler,
                              num_workers=0,
                              pin_memory=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False)

    dataloaders = {
        "training": train_loader,
        "validation": valid_loader
    }
    return dataloaders


class TwoStreamBatchSampler(Sampler):
    """Iterate over two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices,
                 batch_size, secondary_batch_size):
        """

        Args.:
            primary_indices: the indices of the unlabeled data points,
            secondary_indices: the indices of the labeled data points,
            batch_size: batch size for an iteration over the unlabeled data points,
            secondary_batch_size: batch size for an iteration over the labeled data points,
        """
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = self.iterate_once(self.primary_indices)
        secondary_iter = self.iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(self.grouper(primary_iter, self.primary_batch_size),
                   self.grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

    def iterate_once(self, iterable):
        return np.random.permutation(iterable)

    def iterate_eternally(self, indices):
        def infinite_shuffles():
            while True:
                yield np.random.permutation(indices)
        return itertools.chain.from_iterable(infinite_shuffles())

    def grouper(self, iterable, n):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3) --> ABC DEF"
        args = [iter(iterable)] * n
        return zip(*args)
