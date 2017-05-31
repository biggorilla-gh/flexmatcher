"""
Implement classifier for FlexMatcher.

This module defines an interface for classifiers.

Todo:
    * Implement more relevant classifiers.
    * Implement simple rules (e.g., does data match a phone number?).
    * Shuffle data before k-fold cutting in predict_training.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from abc import ABCMeta, abstractmethod

class Classifier(object):

    """Define classifier interface for FlexMatcher."""
    __metaclass__ = ABCMeta
    @abstractmethod
    def __init__(self, data):
        """Train based on the input training data."""
        pass

    @abstractmethod
    def predict_training(self, folds):
        """Predict the training data (using k-fold cross validation)."""
        pass

    @abstractmethod
    def predict(self, data):
        """Predict for unseen data."""
        pass



