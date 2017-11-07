"""
Implement FlexMatcher.

This module is the main module of the FlexMatcher package and implements the
FlexMatcher class.

Todo:
    * Extend the module to work with and without data or column names.
    * Allow users to add/remove classifiers.
    * Combine modules (i.e., create_training_data and training functions).
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import flexmatcher.classify as clf
import flexmatcher.utils as utils
from sklearn import linear_model
import numpy as np
import pandas as pd
import pickle
import time
import os


class FlexMatcher:

    """Match a given schema to the mediated schema.

    The FlexMatcher learns to match an input schema to a mediated schema.
    The class considers panda dataframes as databases and their column names as
    the schema. FlexMatcher learn to do schema matching by training on
    instances of dataframes and how their columns are matched against the
    mediated schema.

    Attributes:
        train_data (dataframe): Dataframe with 3 columns. The name of
            the column in the schema, the value under that column and the name
            of the column in the mediated schema it was mapped to.
        col_train_data (dataframe): Dataframe  with 2 columns. The name
            the column in the schema and the name of the column in the mediated
            schema it was mapped to.
        data_src_num (int): Store the number of available data sources.
        classifier_list (list): List of classifiers used in the training.
        classifier_type (string): List containing the type of each classifier.
            Possible values are 'column' and 'value' classifiers.
        prediction_list (list): List of predictions on the training data
            produced by each classifier.
        meta_model (LogisticRegression): The meta-learner model which outputs
            the best prediction based on the predictions of the base
            classifiers.
        columns (list): The sorted list of column names in the mediated schema.
    """

    def __init__(self, dataframes, mappings, sample_size=300):
        """Prepares the list of classifiers that are being used for matching
        the schemas and creates the training data from the input datafames
        and their mappings.

        Args:
            dataframes (list): List of dataframes to train on.
            mapping (list): List of dictionaries mapping columns of dataframes
                to columns in the mediated schema.
            sample_size (int): The number of rows sampled from each dataframe
                for training.
        """
        print('Create training data ...')
        self.create_training_data(dataframes, mappings, sample_size)
        print('Training data done ...')
        unigram_count_clf = clf.NGramClassifier(ngram_range=(1, 1))
        bigram_count_clf = clf.NGramClassifier(ngram_range=(2, 2))
        unichar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                ngram_range=(1, 1))
        bichar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                               ngram_range=(2, 2))
        trichar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                ngram_range=(3, 3))
        quadchar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                 ngram_range=(4, 4))
        char_dist_clf = clf.CharDistClassifier()
        self.classifier_list = [unigram_count_clf, bigram_count_clf,
                                unichar_count_clf, bichar_count_clf,
                                trichar_count_clf, quadchar_count_clf,
                                char_dist_clf]
        self.classifier_type = ['value', 'value', 'value', 'value',
                                'value', 'value', 'value']
        if self.data_src_num > 5:
            col_char_dist_clf = clf.CharDistClassifier()
            col_trichar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                        ngram_range=(3, 3))
            col_quadchar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                         ngram_range=(4, 4))
            col_quintchar_count_clf = clf.NGramClassifier(analyzer='char_wb',
                                                          ngram_range=(5, 5))
            col_word_count_clf = \
                clf.NGramClassifier(analyzer=utils.columnAnalyzer)
            knn_clf = \
                clf.KNNClassifier()
            self.classifier_list = self.classifier_list + \
                [col_char_dist_clf, col_trichar_count_clf,
                 col_quadchar_count_clf, col_quintchar_count_clf,
                 col_word_count_clf, knn_clf]
            self.classifier_type = self.classifier_type + (['column'] * 6)

    def create_training_data(self, dataframes, mappings, sample_size):
        """Transform dataframes and mappings into training data.

        The method uses the names of columns as well as the data under each
        column as its training data. It also replaces missing values with 'NA'.

        Args:
            dataframes (list): List of dataframes to train on.
            mapping (list): List of dictionaries mapping columns of dataframes
                to columns in the mediated schema.
            sample_size (int): The number of rows sampled from each dataframe
                for training.
        """
        train_data_list = []
        col_train_data_list = []
        for (datafr, mapping) in zip(dataframes, mappings):
            sampled_rows = datafr.sample(min(sample_size, datafr.shape[0]))
            sampled_data = pd.melt(sampled_rows)
            sampled_data.columns = ['name', 'value']
            sampled_data['class'] = \
                sampled_data.apply(lambda row: mapping[row['name']], axis=1)
            train_data_list.append(sampled_data)
            col_data = pd.DataFrame(datafr.columns)
            col_data.columns = ['name']
            col_data['value'] = col_data['name']
            col_data['class'] = \
                col_data.apply(lambda row: mapping[row['name']], axis=1)
            col_train_data_list.append(col_data)
        train_data = pd.concat(train_data_list, ignore_index=True)
        self.train_data = train_data.fillna('NA')
        self.col_train_data = pd.concat(col_train_data_list, ignore_index=True)
        self.col_train_data = \
            self.col_train_data.drop_duplicates().reset_index(drop=True)
        self.data_src_num = len(dataframes)
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
        self.prediction_list = []
        for (clf_inst, clf_type) in zip(self.classifier_list,
                                        self.classifier_type):
            start = time.time()
            # fitting the models and predict for training data
            if clf_type == 'value':
                clf_inst.fit(self.train_data)
                # predicting the training data
                self.prediction_list.append(clf_inst.predict_training())
            elif clf_type == 'column':
                clf_inst.fit(self.col_train_data)
                # predicting the training data
                col_data_prediction = \
                    pd.concat([pd.DataFrame(clf_inst.predict_training()),
                               self.col_train_data], axis=1)
                data_prediction = self.train_data.merge(col_data_prediction,
                                                        on=['name', 'class'],
                                                        how='left')
                data_prediction = np.asarray(data_prediction)
                data_prediction = \
                    data_prediction[:, range(3, 3 + len(self.columns))]
                self.prediction_list.append(data_prediction)
            print(time.time() - start)
        start = time.time()
        self.train_meta_learner()
        print('Meta: ' + str(time.time() - start))

    def train_meta_learner(self):
        """Train the meta-classifier.

        The data used for training the meta-classifier is the probability of
        assigning each point to each column (or class) by each classifier. The
        meta-classifier is a logistic regression model that predicts the best
        label based on the output of base-classifiers."""
        self.meta_model = linear_model.LogisticRegression()
        # preparing the dataset for logistic regression
        regression_data = self.train_data[['class']].copy()
        # adding the prediction probability from classifiers
        for classifier_ind, prediction in enumerate(self.prediction_list):
            classifier_name = 'classifer' + str(classifier_ind)
            column_names = [classifier_name + x for x in self.columns]
            for c in column_names:
                regression_data[c] = None
            regression_data[column_names] = prediction
        # setting up the logistic regression
        self.meta_model.fit(regression_data.iloc[:, 1:],
                            regression_data['class'])

    def make_prediction(self, data):
        """Map the schema of a given dataframe to the column of mediated schema.

        The procedure runs each classifier and then uses the weights (learned
        by the meta-trainer) to combine the prediction of each classifier.

        Args:
            data (dataframe): the input dataframe for which predictions should
                 be made.
        Returns:
            dict: a dictionary that maps the columns of the input dataframe to
            the columns in the mediated schema.
        """
        data = data.fillna('NA').copy(deep=True)
        if data.shape[0] > 100:
            data = data.sample(100)
        # predicting each column
        predicted_mapping = {}
        for column in list(data):
            column_dat = data[[column]]
            column_dat.columns = ['value']
            column_name = pd.DataFrame({'value': [column]*column_dat.shape[0]})
            regression_data = pd.DataFrame()
            for clf_ind, clf_inst in enumerate(self.classifier_list):
                if self.classifier_type[clf_ind] == 'value':
                    raw_prediction = clf_inst.predict(column_dat)
                elif self.classifier_type[clf_ind] == 'column':
                    raw_prediction = clf_inst.predict(column_name)
                # applying the weights to each class in the raw prediction
                classifier_name = 'classifier' + str(clf_ind)
                column_names = [classifier_name + x for x in self.columns]
                curr_dat = pd.DataFrame(raw_prediction, columns=column_names)
                regression_data = pd.concat([regression_data, curr_dat],
                                            axis=1)
            result = self.meta_model.predict_proba(regression_data)
            result = np.sum(result, axis=0) / len(column_dat)
            max_ind = result.argmax()
            predicted_mapping[column] = self.meta_model.classes_[max_ind]
        return predicted_mapping

    def save_model(self, output_file):
        """Serializes the FlexMatcher object into a model file using python's
        pickel library.

        Args:
            output_file (str): the path of the output file.
        """
        data_tuple = (self.classifier_list, self.classifier_type,
                      self.meta_model, self.columns)
        with open(output_file, 'wb') as f:
            pickle.dump(data_tuple, f)

    @classmethod
    def load_model(cls, input_file):
        """Deserialize the FlexMatcher object from a model file using python's
        pickel library.

        Args:
            input_file (str): the path to the model file.

        Returns:
            FlexMatcher: the loaded instance of FlexMatcher
        """
        if not os.path.exists(input_file):
            print('The model file (' + input_file + ') does not exists!')
            return None
        obj = cls.__new__(cls)
        with open(input_file, 'rb') as f:
            data_tuple = pickle.load(f)
            (classifier_list, classifier_type, meta_model, columns) = data_tuple
            obj.columns = columns
            obj.meta_model = meta_model
            obj.classifier_list = classifier_list
            obj.classifier_type = classifier_type
        return obj
