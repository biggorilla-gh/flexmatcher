"""
Implement FlexMatcher.

This module is the main module of the FlexMatcher package and implements the
FlexMatcher class.

Todo:
    * Extend the module to work with and without data or column names.
    * Allow users to add/remove classifiers.
    * Combine modules (i.e., create_training_data and training functions).
    * Resolve the warning related to loading linear regression.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import flexmatcher.classify as clf
import flexmatcher.utils as utils
from sklearn import linear_model
import numpy as np
import pandas as pd
import warnings

class FlexMatcher:

    """Match a given schema to the mediated schema.

    The FlexMatcher learns to match an input schema to a mediated schema.
    The class considers panda dataframes as databases and their column names as
    the schema. FlexMatcher learn to do schema matching by training on
    instances of dataframes and how their columns are matched against the
    mediated schema.

    Attributes:
        training_data (dataframe): Dataframe with 3 columns. The name of
            the column in the schema, the value under that column and the name
            of the column in the mediated schema it was mapped to.
        column_training_data (dataframe): Dataframe  with 2 columns. The name
            the column in the schema and the name of the column in the mediated
            schema it was mapped to.
        data_src_num (int): Store the number of available data sources.
        classifier_list (list): List of classifiers used in the training.
        classifier_type (string): List containing the type of each classifier.
            Possible values are 'column' and 'value' classifiers.
        prediction_list (list): List of predictions on the training data
            produced by each classifier.
        weights (ndarray): A matrix where cell (i,j) captures how good the j-th
            classifier is at predicting if a column should match the i-th
            column (where columns are sorted by name) in the mediated schema.
        columns ()
    """

    def __init__(self):
        pass

    def create_training_data(self, dataframes, mappings, sample_size=200):
        """Transform dataframes and mappings into training data.

        The method uses the names of columns as well as the data under each
        column as its training data. It also replaces missing values with 'NA'.

        Args:
            dataframes (list): List of dataframes to train on.
            mapping (list): List of dictionaries mapping columns of dataframes
                to columns in the mediated schema.
        """
        training_data_list = []
        for (datafr, mapping) in zip(dataframes, mappings):
            sampled_rows = datafr.sample(min(sample_size, datafr.shape[0]))
            sampled_data = pd.melt(sampled_rows)
            sampled_data.columns = ['name', 'value']
            sampled_data['class'] = \
                sampled_data.apply(lambda row: mapping[row['name']], axis=1)
            training_data_list.append(sampled_data)
        training_data = pd.concat(training_data_list, ignore_index=True)
        self.training_data = training_data.fillna('NA')
        self.data_src_num = len(dataframes)
        self.column_training_data = training_data.copy()
        self.column_training_data['value'] = self.column_training_data['name']
        self.column_training_data = self.column_training_data.drop('name', 1)
        self.columns = \
            sorted(list(set.union(*[set(x.values()) for x in mappings])))
        # removing columns that are not present in the dataframe
        # TODO: this should change (It's not ideal to change problem definition
        # without notifying the user)
        available_columns = []
        for (datafr, mapping) in zip(dataframes, mappings):
                for c in datafr.columns:
                    available_columns.append(mapping[c])
        self.columns = sorted(list(set(available_columns)))

    def train(self):
        """Train each classifier and the meta-classifier."""
        word_count_clf = clf.NGramClassifier(self.training_data)
        biword_count_clf = clf.NGramClassifier(self.training_data,
                                               ngram_range=(2,2))
        char_count_clf = clf.NGramClassifier(self.training_data,
                                             analyzer='char_wb',
                                             ngram_range=(1,6))
        char_dist_clf = clf.CharDistClassifier(self.training_data)
        self.classifier_list = [word_count_clf, biword_count_clf,
                                char_count_clf, char_dist_clf]
        self.classifier_type = ['value', 'value',
                                'value', 'value']
        if self.data_src_num > 5:
            col_char_dist_clf = clf.CharDistClassifier(self.column_training_data)
            col_word_count_clf = clf.NGramClassifier(self.column_training_data,
                                                    analyzer=utils.columnAnalyzer)
            col_char_count_clf = clf.NGramClassifier(self.column_training_data,
                                                    analyzer='char_wb',
                                                    ngram_range=(4,6))
            self.classifier_list = self.classifier_list + [col_char_dist_clf,
                                                           col_char_count_clf,
                                                           col_word_count_clf]
            self.classifier_type = self.classifier_type + (['column'] * 3)
        self.prediction_list = \
            [x.predict_training() for x in self.classifier_list]
        self.train_meta_learner()

    def train_meta_learner(self):
        """Train the meta-classifier.

        The data used for training the meta-classifier is the probability of
        assigning each point to each column (or class) by each classifier. The
        learned weights suggest how good each classifier is at predicting a
        particular class."""
        # suppressing a warning from scipy that gelsd is broken and gless is
        # being used instead.
        warnings.filterwarnings(action="ignore", module="scipy",
                                message="^internal gelsd")
        coeff_list = []
        for class_ind, class_name in enumerate(self.columns):
            # preparing the dataset for linear regression
            regression_data = self.training_data[['class']].copy()
            regression_data['is_class'] = \
                np.where(self.training_data['class'] == class_name, True, False)
            # adding the prediction probability from classifiers
            for classifier_ind, prediction in enumerate(self.prediction_list):
                regression_data['classifer' + str(classifier_ind)] = \
                    prediction[:,class_ind]
            # setting up the linear regression
            stacker = linear_model.LogisticRegression(fit_intercept=False,
                                                      class_weight='balanced')
            stacker.fit(regression_data.iloc[:,2:], regression_data['is_class'])
            coeff_list.append(stacker.coef_.reshape(1,-1))
        self.weights = np.concatenate(tuple(coeff_list))

    def make_prediction(self, data):
        """Map the schema of a given dataframe to the column of mediated schema.

        The procedure runs each classifier and then uses the weights (learned
        by the meta-trainer) to combine the prediction of each classifier.
        """
        data = data.fillna('NA')
        if data.shape[0] > 100:
            data = data.sample(100)
        # predicting each column
        predicted_mapping = {}
        for column in list(data):
            column_dat = data[[column]]
            column_dat.columns = ['value']
            column_name = pd.DataFrame({'value': [column]*column_dat.shape[0]})
            scores = np.zeros((len(column_dat), len(self.columns)))
            for clf_ind, clf_inst in enumerate(self.classifier_list):
                if self.classifier_type[clf_ind] == 'value':
                    raw_prediction = clf_inst.predict(column_dat)
                elif self.classifier_type[clf_ind] == 'column':
                    raw_prediction = clf_inst.predict(column_name)
                # applying the weights to each class in the raw prediction
                for class_ind in range(len(self.columns)):
                    raw_prediction[:,class_ind] = \
                        (raw_prediction[:,class_ind] *
                         self.weights[class_ind, clf_ind])
                scores = scores + raw_prediction
            flat_scores = scores.sum(axis=0) / len(column_dat)
            max_ind = flat_scores.argmax()
            predicted_mapping[column] = self.columns[max_ind]
        return predicted_mapping
