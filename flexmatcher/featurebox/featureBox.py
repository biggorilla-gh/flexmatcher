"""
Define abstract class FeatureBox.

FeatureBox is the generic feature-extraction component of FlexMatcher. This
module defines the interface of FeatureBox and specifies the abstract methods
and attributes that FlexMatcher uses to obtain features.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureBox(BaseEstimator, TransformerMixin):

    """Define the interface implemented by all feature-extractor modules in
    FlexMatcher.

    Attributes:
        uses_data(boolean): specifies if the features are extracted using the
        data under each column (if True) or using the column names (if False).
        data_type(str): a string that specifies the type of the column. There
        are 5 possible values for this: str, int, float, bool, cat
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
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        """Transform the data (based on the fitted_parameters) if any to obtain
        the resulting features.

        Args:
            X (np.array): Numpy array of shape [n_samples] storing the input
            data that should be transformed.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def uses_data(self):
        raise NotImplementedError

    @uses_data.setter
    @abstractmethod
    def uses_data(self, val):
        raise NotImplementedError

    @property
    @abstractmethod
    def data_type(self):
        raise NotImplementedError

    @data_type.setter
    @abstractmethod
    def data_type(self, val):
        raise NotImplementedError
