from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import re

def columnAnalyzer(text):
    features = []
    tokens = text.split('-')
    for token in tokens:
        words = re.findall('[a-zA-Z][^A-Z]*', token)
        for word in words:
            features.append(word)
    return list(set(features))

