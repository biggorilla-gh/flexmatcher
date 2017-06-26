from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import re

def columnAnalyzer(text):
    features = []
    words = re.findall('([a-z][a-z1-9]*|[1-9]+|[A-Z](?:[a-z1-9]+|[A-Z1-9]+))', text)
    for word in words:
        features.append(word.lower())
    return list(features)

