import csv
import sys
import os
import statistics
import time
import argparse
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd     #(version 1.0.0)
#import plotly           #(version 4.5.0)
#import plotly.express as px
#import plotly.io as pio
import math
from model.Sentiment_model import analyzefile
In_text = str(input('Input the string:'))
valence, arousal,emotion_name = analyzefile(In_text, 'mean')

print(valence)
print(arousal)
print(emotion_name)
