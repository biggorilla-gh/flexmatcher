from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from flexmatcher.classify import Classifier
import numpy as np


class NGramClassifier(Classifier):

    """Classify data-points using counts of n-gram sequence of words or chars.

    The NGramClassifier uses n-grams of words or characters (based on user
    preference) and extracts count features or binary features (based on user
    preference) to train a classifier. It uses a LogisticRegression
    classifier as its training model.

    Attributes:
        labels (ndarray): Vector storing the labels of each data-point.
        features (ndarray): Matrix storing the extracting features.
        vectorizer (object): Vectorizer for transforming text to features. It
        will be either of type CountVectorizer or HashingVectorizer.
        clf (LogisticRegression): The classifier instance.
        num_classes (int): Number of classes/columns to match to
        all_classes (ndarray): Sorted array of all possible classes
    """

    def __init__(self, ngram_range=(1, 1), analyzer='word', count=True,
                 n_features=200):
        """Initializes the classifier.

        Args:
            ngram_range (tuple): Pair of ints specifying the range of ngrams.
            analyzer (string): Determines what type of analyzer to be used.
            Setting it to 'word' will consider each word as a unit of language
            and 'char' will consider each character as a unit of language.
            count (boolean): Determines if features are counts of n-grams
            versus a binary value encoding if the n-gram is present or not.
            n_features (int): Maximum number of features used.
        """
        # checking what type of vectorizer to create
        if count:
            self.vectorizer = CountVectorizer(analyzer=analyzer,
                                              ngram_range=ngram_range,
                                              max_features=n_features)
        else:
            self.vectorizer = HashingVectorizer(analyzer=analyzer,
                                                ngram_range=ngram_range,
                                                n_features=n_features)

    def fit(self, data):
        """
        Args:
            data (dataframe): Training data (values and their correct column).
        """
        self.labels = np.array(data['class'])
        self.num_classes = len(data['class'].unique())
        self.all_classes = np.sort(np.unique(self.labels))
        values = list(data['value'])
        self.features = self.vectorizer.fit_transform(values).toarray()
        # training the classifier
        self.lrm = linear_model.LogisticRegression(class_weight='balanced')
        self.lrm.fit(self.features, self.labels)

    def predict_training(self, folds=5):
        """Do cross-validation and return probabilities for each data-point.

        Args:
            folds (int): Number of folds used for prediction on training data.
        """
        partial_clf = linear_model.LogisticRegression(class_weight='balanced')
        prediction = np.zeros((len(self.features), self.num_classes))
        skf = StratifiedKFold(n_splits=folds)
        for train_index, test_index in skf.split(self.features, self.labels):
            # prepare the training and test data
            training_features = self.features[train_index]
            test_features = self.features[test_index]
            training_labels = self.labels[train_index]
            # fitting the model and predicting
            partial_clf.fit(training_features, training_labels)
            curr_pred = partial_clf.predict_proba(test_features)
            prediction[test_index] = \
                self.predict_proba_ordered(curr_pred, partial_clf.classes_)
        return prediction

    def predict_proba_ordered(self, probs, classes):
        """Fills out the probability matrix with classes that were missing.

        Args:
            probs (list): list of probabilities, output of predict_proba
            classes_ (ndarray): list of classes from clf.classes_
            all_classes (ndarray): list of all possible classes
        """
        proba_ordered = np.zeros((probs.shape[0], self.all_classes.size),
                                 dtype=np.float)
        sorter = np.argsort(self.all_classes)
        idx = sorter[np.searchsorted(self.all_classes, classes, sorter=sorter)]
        proba_ordered[:, idx] = probs
        return proba_ordered

    def predict(self, data):
        """Predict the class for a new given data.

        Args:
            data (dataframe): Dataframe of values to predict the column for.
        """
        values = list(data['value'])
        features = self.vectorizer.transform(values).toarray()
        return self.lrm.predict_proba(features)
