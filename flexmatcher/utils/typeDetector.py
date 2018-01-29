from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np


class TypeDetector(BaseEstimator, TransformerMixin):

    thresh = 0.8
    category_min_support = 20

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract_types(x) for x in X])

    def _extract_types(self, value_list):
        is_str = 1  # we can make an string out of everything
        is_int, is_float, is_bool, is_cat = 0, 0, 0, 0
        values = pd.Series(value_list)
        # testing the obvious (if it has a type, then it has a type!)
        if str(values.dtype).startswith('int'):
            is_int = True
        elif str(values.dtype).startswith('float'):
            is_float = True
        elif str(values.dtype).startswith('bool'):
            is_bool = True
        else:  # the type is object and values have a mixture of types
            is_int, _ = self.str_to_int(values)
            is_float, _ = self.str_to_float(values)
            is_bool, _ = self.str_to_bool(values)
            is_cat, _ = self.str_to_cat(values)
        return [is_str, is_int, is_float, is_bool, is_cat]

    @classmethod
    def str_to_int(cls, values):
        # try converting the values
        int_values = []
        success_count = 0
        for v in values:
            try:
                int_values.append(int(v))
                success_count += 1
            except ValueError:
                int_values.append(np.nan)
        if success_count / len(values) > cls.thresh:
            return 1, int_values
        else:
            return 0, None

    @classmethod
    def str_to_float(cls, values):
        # try converting the values
        float_values = []
        success_count = 0
        for v in values:
            try:
                float_values.append(float(v))
                success_count += 1
            except ValueError:
                float_values.append(np.nan)
        if success_count / len(values) > cls.thresh:
            return 1, float_values
        else:
            return 0, None

    @classmethod
    def str_to_bool(cls, values):
        # try converting the values
        bool_values = []
        success_count = 0
        for v in values:
            # if in float/integer format:
            try:
                v_int = float(v)
                if v_int == 0 or v_int == 1:
                    bool_values.append(v_int)
                    success_count += 1
                else:
                    bool_values.append(np.nan)
                continue
            except ValueError:
                pass
            # if in string format
            v_str = str(v)
            if v_str.lower() in ['y', 'yes']:
                bool_values.append(1)
                success_count += 1
            elif v_str.lower() in ['n', 'no']:
                bool_values.append(0)
                success_count += 1
            else:
                bool_values.append(np.nan)
        if success_count / len(values) > cls.thresh:
            return 1, bool_values
        else:
            return 0, None

    @classmethod
    def str_to_cat(cls, values):
        # just counting the number of unique values
        num_categories = len(set(values))
        if len(values) / num_categories > cls.category_min_support:
            return 1, values
        else:
            return 0, None

    @classmethod
    def all_to_str(cls, values):
        # just counting the number of unique values
        return [str(x) for x in values]

    @classmethod
    def has_type(cls, type_, one_hot_types):
        if type_ == 'str':
            return one_hot_types[0] == 1
        elif type_ == 'int':
            return one_hot_types[1] == 1
        elif type_ == 'float':
            return one_hot_types[2] == 1
        elif type_ == 'bool':
            return one_hot_types[3] == 1
        elif type_ == 'cat':
            return one_hot_types[4] == 1
