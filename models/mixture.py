from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import time


class GMMClassifier(object):

    def __init__(self, encoder, dataset, labelled_dataset, k, seed):
        super(GMMClassifier, self).__init__()
        self._encoder = encoder.to("cpu")
        encoded_data = encoder.transform(dataset.data)
        encoded_labelled_data = encoder.transform(labelled_dataset.data)
        self._gmm = self._train_gmm(encoded_data, k, seed)
        self._cluster_label = self._assign_labels_to_clusters(
            encoded_labelled_data, labelled_dataset.targets)
        self.n_clusters = None

    def predict(self, x):
        """
        predicts label(tree specie) for data

        Args:
            x: data to be predicted

        Returns:
            predicted labels (tree specie) for each data point
        """
        return self._cluster_label[self._gmm.predict(x)]

    def _assign_labels_to_clusters(self, data, labels_true):
        """
        Assign class label to each GMM cluster using labeled data.
        The class label is based on the class of majority samples within a cluster.
        Unassigned clusters are labeled as -1.

        Args:
            data: data features (labelled data)
            labels_true: data labels

        Returns:
            labels for each cluster
        """
        labels_pred = self._gmm.predict(data)
        labelled_clusters = []
        for i in range(self._gmm.n_components):
            idx = np.where(labels_pred == i)[0]
            if len(idx) != 0:
                labels_freq = np.bincount(labels_true[idx])
                labelled_clusters.append(np.argmax(labels_freq))
            else:
                labelled_clusters.append(-1)

        return np.asarray(labelled_clusters)

    def _train_gmm(self, trainset, n_clusters, seed):
        """
        Train Gaussian Mixture model.

        Args:
            trainset: The dataset to train the model on
            n_clusters: The number of clusters we ideally want to split the data into
            seed: Seed used for reproducibility of experiments

        Returns:
            An sklearn GaussianMixture object
        """
        print("\tFitting Gaussian Mixture Model ...", end=' ')
        start_time = time.time()
        self.n_clusters = n_clusters
        gmm = GaussianMixture(n_components=n_clusters,
                              n_init=5, max_iter=1000, random_state=seed)
        gmm.fit(trainset)
        print("DONE in {:.2f} sec".format(time.time() - start_time))
        return gmm


class BGMClassifier(object):

    def __init__(self, encoder, dataset, labelled_dataset, k, seed):
        super(BGMClassifier, self).__init__()
        self._encoder = encoder.to("cpu")
        encoded_data = encoder.transform(dataset.data)
        encoded_labelled_data = encoder.transform(labelled_dataset.data)
        self._bgm = self._train_bgm(encoded_data, k, seed)
        self._cluster_label = self._assign_labels_to_clusters(
            encoded_labelled_data, labelled_dataset.targets)
        self.n_clusters = None

    def predict(self, x):
        """
        predicts label(tree specie) for data

        Args:
            x: data to be predicted

        Returns:
            predicted labels (tree specie) for each data point
        """
        return self._cluster_label[self._bgm.predict(x)]

    def _assign_labels_to_clusters(self, data, labels_true):
        """
        Assign class label to each Bayesian Gaussian Mixture cluster using labeled data.
        The class label is based on the class of majority samples within a cluster.
        Unassigned clusters are labeled as -1.

        Args:
            data: data features (labelled data)
            labels_true: data labels

        Returns:
            labels for each cluster
        """
        labels_pred = self._bgm.predict(data)
        labelled_clusters = []
        for i in range(self._bgm.n_components):
            idx = np.where(labels_pred == i)[0]
            if len(idx) != 0:
                labels_freq = np.bincount(labels_true[idx])
                labelled_clusters.append(np.argmax(labels_freq))
            else:
                labelled_clusters.append(-1)

        return np.asarray(labelled_clusters)

    def _train_bgm(self, trainset, n_clusters, seed):
        """
        Train Bayesian Gaussian Mixture model.

        Args:
            trainset: The dataset to train the model on
            n_clusters: The number of clusters we ideally want to split the data into
            seed: Seed used for reproducibility of experiments

        Returns:
            An sklearn BayesianGaussianMixture object
        """
        print("\tFitting Bayesian Gaussian Mixture ...", end=' ')
        start_time = time.time()
        self.n_clusters = n_clusters
        bgm = BayesianGaussianMixture(
            n_components=n_clusters, n_init=5, max_iter=1000, random_state=seed)
        bgm.fit(trainset)
        print("DONE in {:.2f} sec".format(time.time() - start_time))
        return bgm
