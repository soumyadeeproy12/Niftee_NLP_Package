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
path = './lib/'
lmtzr = WordNetLemmatizer()
stops = set(stopwords.words("english"))
#anew = "../lib/vad-nrc.csv"
anew = path + "EnglishShortened.csv"
avg_V = 5.06    # average V from ANEW dict
avg_A = 4.21
avg_D = 5.18

# performs sentiment analysis on inputFile using the ANEW database, outputting results to a new CSV file in outputDir
def analyzefile(line, mode):
    s = tokenize.word_tokenize(line.lower())
  
    all_words = []
    found_words = []
    total_words = 0
    v_list = []  # holds valence scores
    a_list = []  # holds arousal scores
    d_list = []  # holds dominance scores

    words = nltk.pos_tag(s)
    for index, p in enumerate(words):
        w = p[0]
        pos = p[1]
        if w in stops or not w.isalpha():
            continue
        j = index-1
        neg = False
        while j >= 0 and j >= index-3:
            
            if words[j][0] == 'not' or words[j][0] == 'no' or words[j][0] == 'n\'t':
                neg = True
                break
            j -= 1


        if pos[0] == 'N' or pos[0] == 'V':
            lemma = lmtzr.lemmatize(w, pos=pos[0].lower())
        else:
            lemma = w

        all_words.append(lemma)

        with open(anew) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Word'].casefold() == lemma.casefold():
                    if neg:
                        found_words.append("neg-"+lemma)
                    else:
                        found_words.append(lemma)
                    v = float(row['valence'])
                    a = float(row['arousal'])
                    d = float(row['dominance'])

                    if neg:
                        
                        v = 5 - (v - 5)
                        a = 5 - (a - 5)
                        d = 5 - (d - 5)

                    v_list.append(v)
                    a_list.append(a)
                    d_list.append(d)

    if len(found_words) == 0:
        Valence = 0
        Arousal = 0
        Dominance = 0
    
    else:
        if mode == 'median':
            sentiment = statistics.median(v_list)
            arousal = statistics.median(a_list)
            dominance = statistics.median(d_list)
        elif mode == 'mean':
            sentiment = statistics.mean(v_list)
            arousal = statistics.mean(a_list)
            dominance = statistics.mean(d_list)
        elif mode == 'mika':
            # calculate valence
            if statistics.mean(v_list) < avg_V:
                sentiment = max(v_list) - avg_V
            elif max(v_list) < avg_V:
                sentiment = avg_V - min(v_list)
            else:
                sentiment = max(v_list) - min(v_list)
            # calculate arousal
            if statistics.mean(a_list) < avg_A:
                arousal = max(a_list) - avg_A
            elif max(a_list) < avg_A:
                arousal = avg_A - min(a_list)
            else:
                arousal = max(a_list) - min(a_list)
            # calculate dominance
            if statistics.mean(d_list) < avg_D:
                dominance = max(d_list) - avg_D
            elif max(d_list) < avg_D:
                dominance = avg_D - min(a_list)
            else:
                dominance = max(d_list) - min(d_list)
        else:
            raise Exception('Unknown mode')

          # set sentiment label
        label = 'neutral'
        if sentiment > 6:
            label = 'positive'
        elif sentiment < 4:
            label = 'negative'

        Valence = sentiment
        Arousal = arousal
        Dominance = dominance
        Valence = 0.25*(Valence - 1) - 1
        Arousal = 0.25*(Arousal - 1) - 1
        Dominance = 0.25*(Dominance - 1) - 1

    ls_expr_intensity = [
      "Slightly", "Moderately", "Very", "Extremely"
      ]
    ls_expr_name = [
      "pleased", "happy", "delighted", "excited", "astonished", 
      "aroused", # first quarter

      "tensed", "alarmed", "afraid", "annoyed", "distressed", 
      "frustrated", "miserable", # second quarter

      "sad", "gloomy", "depressed", "bored", "droopy", "tired", 
      "sleepy", # third quarter

      "calm", "serene", "content", "satisfied"  # fourth quarter
  ]

  # analyzing intensity
    if Dominance < 0.05 and Valence < 0.01 and Valence > -0.01 and Arousal >-0.01 and Arousal <0.01:
        expression_name = "Neutral"
        expression_intensity = ""
    else: 
        if Dominance < 0.225:
            expression_intensity = ls_expr_intensity[0]
        elif Dominance < 0.45:
            expression_intensity = ls_expr_intensity[1]
        elif Dominance < 0.705:
            expression_intensity = ls_expr_intensity[2]
        else:
            expression_intensity = ls_expr_intensity[3]
        if Valence == 0:
            if Arousal >= 0:
                theta = 90
            else:
                theta = 270
        else:
            theta = math.atan(Arousal / Valence)
            theta = theta * (180 / math.pi)

            if Valence < 0:
                theta = 180 + theta
            elif Arousal < 0:
                theta = 360 + theta


        if theta < 15 or theta > 354:
            expression_name = ls_expr_name[0]
        elif theta < 30:
            expression_name = ls_expr_name[1]
        elif theta < 45.5:
            expression_name = ls_expr_name[2]
        elif theta < 60:
            expression_name = ls_expr_name[3]
        elif theta < 75:
            expression_name = ls_expr_name[4]
        elif theta < 90:
            expression_name = ls_expr_name[5]
        elif theta < 105:
            expression_name = ls_expr_name[6]
        elif theta < 120:
            expression_name = ls_expr_name[7]
        elif theta < 135:
            expression_name = ls_expr_name[8]
        elif theta < 150:
            expression_name = ls_expr_name[9]
        elif theta < 165:
            expression_name = ls_expr_name[10]
        elif theta < 180:
            expression_name = ls_expr_name[11]
        elif theta < 195:
            expression_name = ls_expr_name[12]
        elif theta < 210:
            expression_name = ls_expr_name[13]
        elif theta < 225:
            expression_name = ls_expr_name[14]
        elif theta < 240:
            expression_name = ls_expr_name[15]
        elif theta < 255:
            expression_name = ls_expr_name[16]
        elif theta < 260:
            expression_name = ls_expr_name[17]
        elif theta < 275:
            expression_name = ls_expr_name[18]
        elif theta < 290:
            expression_name = ls_expr_name[19]
        elif theta < 305:
            expression_name = ls_expr_name[20]
        elif theta < 320:
            expression_name = ls_expr_name[21]
        elif theta < 335:
            expression_name = ls_expr_name[22]
        elif theta < 354:
            expression_name = ls_expr_name[23]
        else:
            expression_name = "Unknown"
            expression_intensity = ""

  # TODO: return also variable output and not only string


   # i += 1
    return Valence, Arousal, expression_intensity + " " + expression_name