"""
Define class NumericDistFeatureBox.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from flexmatcher.featurebox import FeatureBox
from flexmatcher.utils import TypeDetector


class NumericDistFeatureBox(FeatureBox):

    def __init__(self):
        self.uses_data = True
        self.data_type = 'float'  # note that float capture int too.

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # transforming to the correct type
        X = np.array([TypeDetector.transform_to_type(x, self.data_type)
                      for x in X])
        return np.array([self._extract_features(x) for x in X])

    def _extract_features(self, number_list):
        return [np.min(number_list), np.max(number_list),
                np.mean(number_list), np.median(number_list)]

    @property
    def uses_data(self):
        return self._uses_data

    @uses_data.setter
    def uses_data(self, val):
        self._uses_data = val

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, val):
        self._data_type = val
