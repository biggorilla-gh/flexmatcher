from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from sklearn.base import BaseEstimator, TransformerMixin
import Levenshtein as lev
import numpy as np


class LevenshteinCore(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self._classes = sorted(list(set(y)))
        return self

    def transform(self, X):
        return np.array([self._extract_features(x) for x in X])

    def _extract_features(self, str_):
        features = []
        for class_ in self._classes:
            features.append(lev.distance(str_, class_))
        return features
