import time
import numpy as np
from sklearn.cluster import KMeans


class KmeansClassifier(object):

    def __init__(self, encoder, dataset, labelled_dataset, k, seed):
        super(KmeansClassifier, self).__init__()
        self._encoder = encoder.to("cpu")
        encoded_data = encoder.transform(dataset.data)
        encoded_labelled_data = encoder.transform(labelled_dataset.data)
        self._kmeans = self._train_kmeans(encoded_data, k, seed)
        self._cluster_label = self._assign_labels_to_clusters(
            encoded_labelled_data, labelled_dataset.targets)

    def predict(self, x):
        """
        predicts label(tree specie) for data

        Args:
            x: data to be predicted

        Returns:
            predicted labels (tree specie) for each data point
        """
        return self._cluster_label[self._kmeans.predict(x)]

    def _assign_labels_to_clusters(self, data, labels_true):
        """
        Assign class label to each K-means cluster using labeled data.
        The class label is based on the class of majority samples within a cluster.
        Unassigned clusters are labeled as -1.

        Args:
            data: data features (labelled data)
            labels_true: data labels

        Returns:
            labels for each cluster
        """
        labels_pred = self._kmeans.predict(data)
        labelled_clusters = []
        for i in range(self._kmeans.n_clusters):
            idx = np.where(labels_pred == i)[0]
            if len(idx) != 0:
                labels_freq = np.bincount(labels_true[idx])
                labelled_clusters.append(np.argmax(labels_freq))
            else:
                labelled_clusters.append(-1)

        return np.asarray(labelled_clusters)

    def _train_kmeans(self, trainset, n_clusters, seed):
        """
        Train K-means model.

        Args:
            trainset: The dataset to train the model on
            n_clusters: The number of clusters we ideally want to split the data into
            seed: Seed used for reproducibility of experiments

        Returns:
            An sklearn KMeans object
        """
        print("\tFitting k-means ...", end=' ')
        start_time = time.time()
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters,
                        n_init=5, max_iter=1000, random_state=seed, n_jobs=-1)
        kmeans.fit(trainset)
        print("DONE in {:.2f} sec".format(time.time() - start_time))
        return kmeans
