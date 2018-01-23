"""
Define abstract class FeatureBox.

FeatureBox is the generic feature-extraction component of FlexMatcher. This
module defines the interface of FeatureBox and specifies the abstract methods
and attributes that FlexMatcher uses to obtain features.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from abc import ABCMeta, abstractmethod, abstractproperty
from sklearn.base import BaseEstimator, ClassifierMixin


class FeatureBox(BaseEstimator, ClassifierMixin):

    """Define the interface implemented by all feature-extractor modules in
    FlexMatcher.

    Attributes:
        uses_data(boolean): specifies if the features are extracted using the
        data under each column (if True) or using the column names (if False).
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the parameters that are required for feature extraction based
        on the input data.

        Args:
            X (np.array): Numpy array of shape [n_samples] storing the input
            data.
            y (np.array): Numpy array of shape [n_samples] storing the labels
            associated with each data point.
        """
        pass

    @abstractmethod
    def transform(self, X):
        """Transform the data (based on the fitted_parameters) if any to obtain
        the resulting features.

        Args:
            X (np.array): Numpy array of shape [n_samples] storing the input
            data that should be transformed.
        """
        pass

    @abstractproperty
    def uses_data(self):
        pass
