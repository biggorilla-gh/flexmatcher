import unittest

import flexmatcher.utils as ut


class TypeDetectorTest(unittest.TestCase):

    def test_type_detection(self):
        td = ut.TypeDetector()
        # parsing different column names
        some_integers = ['0', '1', '2']
        self.assertEqual(td._extract_features(some_integers),
                         [1, 1, 1, 0, 0])
        # ------------------------------
        integers_with_erros = ['0', '1', '2'] * 10 + ['a']
        self.assertEqual(td._extract_features(integers_with_erros),
                         [1, 1, 1, 0, 0])
        # ------------------------------
        integers_with_many_erros = ['a', '1', 'b'] * 10
        self.assertEqual(td._extract_features(integers_with_many_erros),
                         [1, 0, 0, 0, 0])
        # ------------------------------
        some_floats = ['0.0', '0.1', '0.2']
        self.assertEqual(td._extract_features(some_floats),
                         [1, 0, 1, 0, 0])
        # ------------------------------
        floats_with_errors = ['0.0', '0.1', '0.2'] * 10 + ['a']
        self.assertEqual(td._extract_features(floats_with_errors),
                         [1, 0, 1, 0, 0])
        # ------------------------------
        floats_with_many_errors = ['a', '0.1', 'b'] * 10
        self.assertEqual(td._extract_features(floats_with_many_errors),
                         [1, 0, 0, 0, 0])
        # ------------------------------
        some_booleans_v1 = ['0', '1', '0'] * 10
        self.assertEqual(td._extract_features(some_booleans_v1),
                         [1, 1, 1, 1, 0])
        # ------------------------------
        some_booleans_v2 = ['y', 'n', 'y'] * 10
        self.assertEqual(td._extract_features(some_booleans_v2),
                         [1, 0, 0, 1, 0])
        # ------------------------------
        some_booleans_v3 = ['yes', 'no', 'yes'] * 10
        self.assertEqual(td._extract_features(some_booleans_v3),
                         [1, 0, 0, 1, 0])
        # ------------------------------
        booleans_with_errors = ['yes', 'no', 'yes'] * 10 + ['a']
        self.assertEqual(td._extract_features(booleans_with_errors),
                         [1, 0, 0, 1, 0])
        # ------------------------------
        booleans_with_many_errors = ['yes', 'no', 'a'] * 10
        self.assertEqual(td._extract_features(booleans_with_many_errors),
                         [1, 0, 0, 0, 0])
        # ------------------------------
        some_categories = ['a', 'b', 'c'] * 100
        self.assertEqual(td._extract_features(some_categories),
                         [1, 0, 0, 0, 1])
        # ------------------------------
        some_categories = ['0', '1'] * 100
        self.assertEqual(td._extract_features(some_categories),
                         [1, 1, 1, 1, 1])
