import pandas as pd
import unittest
import random

import flexmatcher


class NameOnlyTest(unittest.TestCase):

    def setUp(self):
        target_columns = ['California', 'Hawaii', 'New Hampshire',
                          'Colorado', 'Maryland', 'Oklahoma',
                          'Oregon', 'Tennessee', 'Washington',
                          'Nebraska']
        # creating 10 different dataframes
        self.schema_list = []
        self.mappings = []
        for i in range(10):
            vals = {}
            mapping = {}
            for col in target_columns:
                # replace two characters with 'x'
                col_list = list(col)
                col_list[random.randint(0, len(col) - 1)] = 'x'
                col_list[random.randint(0, len(col) - 1)] = 'x'
                new_col = ''.join(col_list)
                vals[new_col] = ['hello there'] * 5
                mapping[new_col] = col
            self.schema_list.append(pd.DataFrame(vals))
            self.mappings.append(mapping)

    def test_prediction(self):
        # Using Flexmatcher
        for ind in range(10):     # ind is the index to predict
            schemas = [x for i, x in enumerate(self.schema_list) if i != ind]
            mappings = [x for i, x in enumerate(self.mappings) if i != ind]
            fm = flexmatcher.FlexMatcher(schemas,
                                         mappings)
            fm.train()
            pred_mapping = fm.make_prediction(self.schema_list[ind])
            for col in self.mappings[ind]:
                self.assertEqual(pred_mapping[col], self.mappings[ind][col])
