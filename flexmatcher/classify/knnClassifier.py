from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from sklearn.model_selection import StratifiedKFold
from flexmatcher.classify import Classifier
import numpy as np
import Levenshtein as lev


class KNNClassifier(Classifier):

    """Classify data-points (in string format) using their 3 nearest neighbors
    using the levenshtein distance metric.

    Attributes:
        labels (ndarray): Vector storing the labels of each data-point.
        strings (list): List of strings for which the labels are provided.
        num_classes (int): Number of classes/columns to match to.
        column_index (dict): Dictionary mapping each column to its index.
    """

    def __init__(self):
        """Initializes the classifier."""
        pass

    def fit(self, data):
        """Store the strings and their corresponding labels.

        Args:
            data (dataframe): Training data (values and their correct column).
        """
        self.labels = np.array(data['class'])
        self.strings = np.array(data['value'])
        self.num_classes = len(data['class'].unique())
        self.column_index = dict(zip(sorted(list(data['class'].unique())),
                                     range(self.num_classes)))

    def predict_training(self, folds=5):
        """Do cross-validation and return probabilities for each data-point.

        Args:
            folds (int): Number of folds used for prediction on training data.
        """
        prediction = np.zeros((len(self.strings), self.num_classes))
        skf = StratifiedKFold(n_splits=folds)
        for train_index, test_index in skf.split(self.strings, self.labels):
            # prepare the training and test data
            training_strings = self.strings[train_index]
            test_strings = self.strings[test_index]
            training_labels = self.labels[train_index]
            # predicting the results
            part_prediction = self.find_knn(training_strings, training_labels,
                                            test_strings)
            prediction[test_index] = part_prediction
        return prediction

    def find_knn(self, train_strings, train_labels, test_strings):
        """Find 3 nearest neighbors of each item in test_strings in
        train_strings and report their labels as the prediction.

        Args:
            train_strings (ndarray): Numpy array with strings in training set
            train_labels (ndarray): Numpy array with labels of train_strings
            test_strings (ndarray): Numpy array with string to be predict for
        """
        prediction = np.zeros((len(test_strings), self.num_classes))
        for i in range(len(test_strings)):
            a_str = test_strings[i]
            dists = np.array([0] * len(train_strings))
            for j in range(len(train_strings)):
                b_str = train_strings[j]
                dists[j] = lev.distance(a_str, b_str)
            # finding the top 3
            top3 = dists.argsort()[:3]
            for ind in top3:
                prediction[i][self.column_index[train_labels[ind]]] += 1.0 / 3
        return prediction

    def predict(self, data):
        """Predict the class for a new given data.

        Args:
            data (dataframe): Dataframe of values to predict the column for.
        """
        input_strings = np.array(data['value'])
        return self.find_knn(self.strings, self.labels, input_strings)
