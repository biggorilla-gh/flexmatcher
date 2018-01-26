"""
Implement FlexMatcher.

This module is the main module of the FlexMatcher package and implements the
FlexMatcher class.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import flexmatcher.featurebox as fbox
import flexmatcher.utils as utils
import flexmatcher.core as core

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from munkres import Munkres
import pandas as pd


class FlexMatcher(object):

    def __init__(self):
        print('Setting up FlexMatcher ...')
        # setting up the set of default FeatureBoxes
        self.feature_boxes = {}
        self._init_data_featureboxes()
        self._init_header_featureboxes()
        self._build_pipeline()

    def _init_data_featureboxes(self):
        # features based on words
        for n_gram in range(2):
            self.feature_boxes[str(n_gram) + 'w'] = \
                fbox.FeatureBoxWithCore(
                    core=CountVectorizer(ngram_range=(n_gram, n_gram)),
                    return_probs=False
                )
        # features based on characters
        for n_gram in range(4):
            self.feature_boxes[str(n_gram) + 'c'] = \
                fbox.FeatureBoxWithCore(
                    core=CountVectorizer(analyzer='char_wb',
                                         ngram_range=(n_gram, n_gram)),
                    return_probs=False
                )
        # features based on type of characters
        self.feature_boxes['c_dist'] = fbox.FeatureBoxWithCore(
            core=core.CharDistCore(),
            return_probs=False
        )

    def _init_header_featureboxes(self):
        # features based on words
        self.feature_boxes['col_1w'] = \
            fbox.FeatureBoxWithCore(
                core=CountVectorizer(analyzer=utils.columnAnalyzer),
                uses_data=False,
                return_probs=False
            )
        # features based on characters
        for n_gram in range(3, 6):
            self.feature_boxes['col_' + str(n_gram) + 'c'] = \
                fbox.FeatureBoxWithCore(
                    core=CountVectorizer(analyzer='char_wb',
                                         ngram_range=(n_gram, n_gram)),
                    uses_data=False,
                    return_probs=False
                )
        self.feature_boxes['col_c_dist'] = fbox.FeatureBoxWithCore(
            core=core.CharDistCore(),
            uses_data=False,
            return_probs=False
        )
        # TODO: features to be implemented
        # knn_clf = clf.KNNClassifier()

    def _build_pipeline(self):
        # building a feature union
        transformer_list = []
        for box_name, box in self.feature_boxes.items():
            selector_key = 'data' if box.uses_data else 'name'
            selector_pl = Pipeline([
                ('selector', utils.ItemSelector(key=selector_key)),
                ('estimator', box)
            ])
            transformer_list.append((box_name, selector_pl))
        self.union = FeatureUnion(transformer_list=transformer_list)
        # building the entire pipeline
        self.pipeline = Pipeline([
            ('union', self.union),
            ('clf', LogisticRegression(class_weight='balanced'))
        ])

    def train(self, dataframes, mappings, sample_size=100):
        print('Creating Tranining Data ...')
        self._transform_train_data(dataframes, mappings, sample_size)
        self.pipeline.fit(self.training_data[['name', 'data']],
                          self.training_data['label'])

    def _transform_train_data(self, dataframes, mappings, sample_size):
        all_column_names = []
        all_column_data = []
        all_column_lables = []
        for (full_df, map_) in zip(dataframes, mappings):
            if sample_size > full_df.shape[0]:
                # TODO: issue a warning
                df = full_df
            else:
                df = full_df.sample(sample_size)
            df_column_names = list(df.columns)
            df_column_data = [list(df.iloc[:, i]) for i in range(len(list(df)))]
            df_column_labels = []
            for col in df_column_names:
                try:
                    df_column_labels.append(map_[col])
                except KeyError:
                    # TODO: issue a warning
                    df_column_labels.append('no_mapping')
            # appending the extracted results
            all_column_names += df_column_names
            all_column_data += df_column_data
            all_column_lables += df_column_labels
        # creating final training data based on these
        self.training_data = pd.DataFrame({'name': all_column_names,
                                           'data': all_column_data,
                                           'label': all_column_lables})
        self.num_sources = len(dataframes)
        self.labels = list(set(all_column_lables))

    def predict(self, full_df, sample_size=100, predict_all=True):
        predict_data = self._transform_predict_data(full_df, sample_size)
        if predict_all:
            # convert the results into dictionary
            result = dict(zip(predict_data['name'],
                              self.pipeline.predict(predict_data)))
            return result
        else:
            munk = Munkres()
            likelihoods = self.pipeline.predict_proba(predict_data)
            indexes = munk.compute(likelihoods)
            predicted_mapping = {}
            for (row, col) in indexes:
                predicted_mapping[list(full_df)[row]] = \
                    self.pipeline.classes_[col]
        return predicted_mapping

    def _transform_predict_data(self, full_df, sample_size):
        if sample_size > full_df.shape[0]:
            # TODO: issue a warning
            df = full_df
        else:
            df = full_df.sample(sample_size)
        df_column_names = list(df.columns)
        df_column_data = [list(df.iloc[:, i]) for i in range(len(list(df)))]
        # creating final training data based on these
        predict_data = pd.DataFrame({'name': df_column_names,
                                     'data': df_column_data})
        return predict_data
