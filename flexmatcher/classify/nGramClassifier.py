from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from flexmatcher.classify import Classifier
import numpy as np

class NGramClassifier(Classifier):

    """Classify data-points using counts of n-gram sequence of words or chars.

    The NGramClassifier uses n-grams of words or characters (based on user
    preference) and extracts count features or binary features (based on user
    preference) to train a classifier. It uses the Gaussian naive Bayes
    classifier as its training model.

    Attributes:
        labels (ndarray): Vector storing the labels of each data-point.
        features (ndarray): Matrix storing the extracting features.
        vectorizer (object): Vectorizer for transforming text to features. It
        will be either of type CountVectorizer or HashingVectorizer.
        clf (DecisionTreeClassifier): The classifier instance.
        num_classes (int): Number of classes/columns to match to
    """

    def __init__(self, data, ngram_range=(1,1), analyzer='word', count=True):
        """Extracts features and labels from the data and fits a model.

        Args:
            data (dataframe): Training data (values and their correct column).
            ngram_range (tuple): Pair of ints specifying the range of ngrams.
            analyzer (string): Determines what type of analyzer to be used.
            Setting it to 'word' will consider each word as a unit of language
            and 'char' will consider each character as a unit of language.
            count (boolean): Determines if features are counts of n-grams
            versus a binary value encoding if the n-gram is present or not.
        """
        self.labels = np.array(data['class'])
        self.num_classes = len(data['class'].unique())
        values = list(data['value'])
        # checking what type of vectorize to create
        if count:
            self.vectorizer = CountVectorizer(analyzer = analyzer,
                                              ngram_range = ngram_range,
                                              max_features = 200)
        else:
            self.vectorizer = HashingVectorizer(analyzer = analyzer,
                                                ngram_range = ngram_range,
                                                n_features = 200)
        self.features = self.vectorizer.fit_transform(values).toarray()
        # training the classifier
        self.gnb = GaussianNB()
        self.gnb.fit(self.features, self.labels)

    def predict_training(self, folds=5):
        """Do cross-validation and return probabilities for each data-point.

        Args:
            folds (int): Number of folds used for prediction on training data.
        """
        partial_clf = GaussianNB()
        prediction = np.zeros((len(self.features), self.num_classes))
        skf = StratifiedKFold(n_splits=folds)
        for train_index, test_index in skf.split(self.features, self.labels):
            # prepare the training and test data
            training_features = self.features[train_index]
            test_features = self.features[test_index]
            training_labels = self.labels[train_index]
            # fitting the model and predicting
            partial_clf.fit(training_features, training_labels)
            prediction[test_index] = partial_clf.predict_proba(test_features)
        return prediction

    def predict(self, data):
        """Predict the class for a new given data.

        Args:
            data (dataframe): Dataframe of values to predict the column for.
        """
        values = list(data['value'])
        features = self.vectorizer.transform(values).toarray()
        return self.gnb.predict_proba(features)


