from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
