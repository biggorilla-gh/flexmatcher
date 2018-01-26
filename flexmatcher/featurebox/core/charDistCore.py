from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np


class CharDistCore(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract_features(x) for x in X])

    def _extract_features(self, str_):
        len_ = len(str_)
        digit_frac = 0 if len_ == 0 else sum(c.isdigit() for c in str_) / len_
        digit_cnt = sum(c.isdigit() for c in str_)
        alpha_frac = 0 if len_ == 0 else sum(c.isalpha() for c in str_) / len_
        alpha_cnt = sum(c.isalpha() for c in str_)
        space_frac = 0 if len_ == 0 else sum(c.isspace() for c in str_) / len_
        space_cnt = sum(c.isspace() for c in str_)
        return [len_,
                digit_frac, digit_cnt,
                alpha_frac, alpha_cnt,
                space_frac, space_cnt]
