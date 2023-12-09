from nltk.tokenize import word_tokenize
import csv
import re
import jieba
import llm
import pickle
import sys
import datetime
import time


embeds = []
pairs = []
tokens = []

### load raw training data
with open('CS_train.csv', newline='', encoding="utf8") as f:
    reader = csv.reader(f)
    data = list(reader)

### create vocabulary and tokenizing EN and ZH sentences
for index, line in enumerate(data):
    en_l = word_tokenize(line[1])
    zh_l = list(jieba.cut_for_search(line[2]))
    pairs.append([en_l, zh_l])
    for i in en_l:
        tokens.append(i)
    for j in zh_l:
        tokens.append(j)

### keep only distinct words and save them to file
tokens = list(set(tokens))
print('Num tokens:', len(tokens))
with open ('Tokens.txt', 'w', encoding="utf8") as f:
    for t in tokens:
        f.write(t+'\n')

### create embeddings for each token
### augmenting dataset with (x,x) (y,x) (y,y) pairs
model = llm.get_embedding_model('ada-002')
model.key = '' # replace value with own OpenAI key
for index, pair in enumerate(pairs):
    x_embed = list(model.embed_multi(pair[0]))
    y_embed = list(model.embed_multi(pair[1]))
    embeds.append([x_embed, x_embed])
    embeds.append([x_embed, y_embed])
    embeds.append([y_embed, x_embed])
    embeds.append([y_embed, y_embed])
    ### saving for backup
    if (index+1) % 50 == 0:
        with open('Embeddings', 'wb') as f:
            pickle.dump(pairs, f)

with open('Embeddings', 'wb') as f:
    pickle.dump(pairs, f)