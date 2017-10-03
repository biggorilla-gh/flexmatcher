import pandas as pd
import unittest
import random

import flexmatcher


class KNNTest(unittest.TestCase):

    def setUp(self):
        self.target_class = ['California', 'Hawaii', 'New Hampshire',
                             'Colorado', 'Maryland', 'Oklahoma',
                             'Oregon', 'Tennessee', 'Washington', 'Nebraska']
        self.dat_train = self.create_data()
        self.dat_test = self.create_data()

    def create_data(self):
        col_value = [''] * 100
        col_class = [''] * 100
        for i in range(100):
            ind = random.randint(0, len(self.target_class) - 1)
            sample_class = self.target_class[ind]
            value = list(sample_class)
            value[random.randint(0, len(sample_class) - 1)] = 'x'
            value[random.randint(0, len(sample_class) - 1)] = 'x'
            value = ''.join(value)
            col_class[i] = sample_class
            col_value[i] = value
        return(pd.DataFrame(list(zip(col_value, col_value, col_class)),
                            columns=['name', 'value', 'class']))

    def test_prediction(self):
        # Using Flexmatcher
        clf = flexmatcher.classify.KNNClassifier()
        clf.fit(self.dat_train)
        res = clf.predict(self.dat_test)
        for i in range(len(self.dat_test)):
            correct_ind = clf.column_index[self.dat_test['class'][i]]
            self.assertTrue(res[i, correct_ind] > 0.5)
