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
import flexmatcher.featurebox.core as core

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from munkres import Munkres
import pandas as pd
import pickle
import os


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
                    core=CountVectorizer(ngram_range=(n_gram, n_gram))
                )
        # features based on characters
        for n_gram in range(4):
            self.feature_boxes[str(n_gram) + 'c'] = \
                fbox.FeatureBoxWithCore(
                    core=CountVectorizer(analyzer='char_wb',
                                         ngram_range=(n_gram, n_gram))
                )
        # features based on type of characters
        self.feature_boxes['c_dist'] = fbox.FeatureBoxWithCore(
            core=core.CharDistCore()
        )
        # features based on the distribution of the numbers
        self.feature_boxes['num_dist'] = fbox.NumericDistFeatureBox()

    def _init_header_featureboxes(self):
        # features based on words
        self.feature_boxes['col_1w'] = \
            fbox.FeatureBoxWithCore(
                core=CountVectorizer(analyzer=utils.columnAnalyzer),
                uses_data=False
            )
        # features based on characters
        for n_gram in range(3, 6):
            self.feature_boxes['col_' + str(n_gram) + 'c'] = \
                fbox.FeatureBoxWithCore(
                    core=CountVectorizer(analyzer='char_wb',
                                         ngram_range=(n_gram, n_gram)),
                    uses_data=False
                )
        self.feature_boxes['col_c_dist'] = fbox.FeatureBoxWithCore(
            core=core.CharDistCore(),
            uses_data=False
        )
        self.feature_boxes['lev_clf'] = fbox.FeatureBoxWithCore(
            core=core.LevenshteinCore(),
            uses_data=False,
            return_probs=False
        )

    def _build_pipeline(self):
        # building a feature union
        transformer_list = []
        # adding the types of values as features
        type_detector_pl = Pipeline([
            ('selector', utils.ItemSelector(key='data')),
            ('estimator', utils.TypeDetector())
        ])
        transformer_list.append(('type_detector', type_detector_pl))
        # adding the feature boxes
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
        all_column_types = []
        all_column_data = []
        all_column_lables = []
        td = utils.TypeDetector()
        for (full_df, map_) in zip(dataframes, mappings):
            if sample_size > full_df.shape[0]:
                # TODO: issue a warning
                df = full_df
            else:
                df = full_df.sample(sample_size)
            df_column_names = list(df.columns)
            df_column_data = [list(df.iloc[:, i]) for i in range(len(list(df)))]
            df_column_types = [td._extract_types(x) for x in df_column_data]
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
            all_column_types += df_column_types
            all_column_lables += df_column_labels
        # creating final training data based on these
        self.training_data = pd.DataFrame({'name': all_column_names,
                                           'data': all_column_data,
                                           'types': all_column_types,
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

    def save_model(self, output_file):
        """Serializes the FlexMatcher object into a model file using python's
        pickel library.
        Args:
            output_file (str): the path of the output file.
        """
        with open(output_file, 'wb') as f:
            pickle.dump(self, f)

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
        with open(input_file, 'rb') as f:
            loaded_matcher = pickle.load(f)
        return loaded_matcher
