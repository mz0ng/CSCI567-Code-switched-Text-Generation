import math
import random
import csv
import time
from collections import Counter, defaultdict
from nltk import word_tokenize,trigrams
from nltk import FreqDist, KneserNeyProbDist


### create and train trigram model with KneserNey smoothing
grams = []

with open('CS.csv', newline='', encoding="utf8") as f:
    reader = csv.reader(f)
    data = list(reader)
data.pop(0)


for line in data:
    word_tok = word_tokenize(line[1])
    grams = grams + [i for i in trigrams(word_tok)]


freq_dist = FreqDist(grams)
kneser_ney = KneserNeyProbDist(freq_dist)
for i in range(10):
    print(kneser_ney.generate())