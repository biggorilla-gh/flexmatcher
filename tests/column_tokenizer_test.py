import unittest

import flexmatcher.utils as ut


class ColumnAnalyzer(unittest.TestCase):

    def test_prediction(self):
        # parsing different column names
        self.assertTrue(ut.columnAnalyzer('RIT_17') == ['rit', '17'])
        self.assertTrue(ut.columnAnalyzer('RIT-17') == ['rit', '17'])
        self.assertTrue(ut.columnAnalyzer('RIT 17') == ['rit', '17'])
        self.assertTrue(ut.columnAnalyzer('RIT+17') == ['rit', '17'])
        self.assertTrue(ut.columnAnalyzer('rit+17') == ['rit', '17'])
        self.assertTrue(ut.columnAnalyzer('TestingThisThing') ==
                        ['testing', 'this', 'thing'])
        self.assertTrue(ut.columnAnalyzer('TestingThisTHING') ==
                        ['testing', 'this', 'thing'])
        self.assertTrue(ut.columnAnalyzer('TestingThisTHING') ==
                        ['testing', 'this', 'thing'])
