"""
Define class FeatureBoxWithCore.

FeatureBoxWithCore implements a specific type of FeatureBox which accepts a
sklearn transformer as input, extracts the features using the transformer using
the data or the header, and if necessary fits a logistic regression to get
likelihoods instead of the actual features.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from flexmatcher.featurebox import FeatureBox


class FeatureBoxWithCore(FeatureBox):

    """Implement a feature-extractor that accepts a sklearn transformer as its
    core feature extractor engine, and uses it to extract features from the
    data or the column header.

    Attributes:
        uses_data (boolean): specifies if the features are extracted using the
        data under each column (if True) or using the column names (if False).
        core (sk.TransformerMixin): a core estimator from sklearn that will
            be used to extract feature from the column header and column data.
        data_type (str): specifies the type of data that is expected by the
            classifier. If uses_data is False, then the data_type must be a
            string since column headers are strings.
        return_probs (boolean): specifies if the actual features should be
            extracted for each data point or the likelihood of the data point
            being associated with each possible class.
        clf (sk.logisticRegression): the logistic regression classifier that
        will be used if return_probs is True. This attribute will be None
        if return_probs is False.
    """

    def __init__(self, core=CountVectorizer(), uses_data=True,
                 data_type='str', return_probs=False):
        """Initialize the class object.

        Args:
            uses_data (boolean): specifies if the features are extracted using
            the data under each column (if True) or using the column names (if
            False).
            core (sk.TransformerMixin): a core estimator from sklearn that will
            be used to extract feature from the column header and column data.
            The default value is a CountVectorizer module which can be applied
            on strings.
            data_type (str): specifies the type of data that is expected by the
            classifier. If uses_data is False, then the data_type must be a
            string since column headers are strings.
            return_probs (boolean): specifies if the actual features should be
            extracted for each data point or the likelihood of the data point
            being associated with each possible class.
        """
        self.uses_data = uses_data
        self.core = core
        self.data_type = data_type
        self.return_probs = return_probs
        # enforcing type 'str' when uses_data is False
        if not uses_data and data_type != 'str':
            print('Data type {} is not valid for a FeatureBox that \
                  extracts features from column headers. Switching back \
                  to data type to "str"'.format(data_type))
            self.data_type = 'str'
        # creating a classifier if needed
        if return_probs:
            self.clf = LogisticRegression(class_weight='balanced')
        else:
            self.clf = None

    def fit(self, X, y=None):
        """Fit the parameters that are required for feature extraction based
        on the input data.

        Args:
            X (np.array): Numpy array of shape [n_samples] storing the input
            data.
            y (np.array): Numpy array of shape [n_samples] storing the labels
            associated with each data point.

        Returns:
            (FeatureBoxWithCore): Returns the self object as required by sklearn
            guidelines.
        """
        if self.use_data:
            self._fit_data(X, y)
        else:
            self._fit_header(X, y)
        return self

    def _fit_data(self, X, y):
        """Fit the parameters for feature extraction when features are extracted
        from the data under each column.

        Args:
            X (np.array): Numpy array of shape [n_samples] storing the input
            data.
            y (np.array): Numpy array of shape [n_samples] storing the labels
            associated with each data point.
        """
        # Creating the new X and y matrices based on the data listed
        data_X = np.concatenate(X)
        repeated_y = [np.array([y[i]] * len(X[i])) for i in range(len(X))]
        data_y = np.concatenate(repeated_y)
        # Checking if the classifier needs to be trained or not
        if self.return_probs:
            local_features = self.core.fit_transform(data_X, data_y)
            self.clf.fit(local_features, data_y)
        else:
            self.core.fit(data_X, data_y)

    def _fit_header(self, X, y):
        """Fit the parameters for feature extraction when features are extracted
        from the header of each column.

        Args:
            X (np.array): Numpy array of shape [n_samples] storing the input
            data.
            y (np.array): Numpy array of shape [n_samples] storing the labels
            associated with each data point.
        """
        # Checking if the classifier needs to be trained or not
        if self.return_probs:
            local_features = self.core.fit_transform(X, y)
            self.clf.fit(local_features, y)
        else:
            self.core.fit(X, y)

    def transform(self, X):
        """Transform the data based on the fitted_parameters (if any) to obtain
        the resulting features.

        Args:
            X (np.array): Numpy array of shape [n_samples] storing the input
            data that should be transformed.

        Returns:
            (np.array): Numpy array of shape [n_samples, num_features] storing
            the resulting features computed for the input data.
        """
        if self.use_data:
            return self._transform_data(X)
        else:
            return self._transform_header(X)

    def _transform_data(self, X):
        """Transform the data listed under each column to obtain the resulting
        features and then combine the features to get a unique set of features
        for the entire column.

        Args:
            X (np.array): Numpy array of shape [n_samples] storing the input
            data that should be transformed.

        Returns:
            (np.array): Numpy array of shape [n_samples, num_features] storing
            the resulting features computed for the input data.
        """
        # Checking if the classifier needs to used or not
        data_X = np.concatenate(X)
        if self.return_probs:
            # need to run the LR classifier and return the probabilities.
            local_features = self.core.transform(data_X)
            data_features = self.clf.predict_proba(local_features)
        else:
            data_features = self.core.transform(data_X)
        # Combining features back together
        features = np.zeros((len(X), data_features.shape[1]))
        index = 0
        for i in range(len(data_X)):
            num_points = len(data_X[i])
            features[i] = np.mean(data_features[index:(index + num_points)],
                                  axis=0)
        return features

    def _transform_header(self, X):
        """Transform the header of each column to obtain the resulting features.

        Args:
            X (np.array): Numpy array of shape [n_samples] storing the input
            data that should be transformed.

        Returns:
            (np.array): Numpy array of shape [n_samples, num_features] storing
            the resulting features computed for the input data.
        """
        # Checking if the classifier needs to used or not
        if self.return_probs:
            # need to run the LR classifier and return the probabilities.
            local_features = self.core.transform(X)
            return self.clf.predict_proba(local_features)
        else:
            return self.core.transform(X)
