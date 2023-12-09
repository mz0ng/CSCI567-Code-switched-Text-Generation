import os
import pandas as pd
import csv
from googletrans import Translator
import random

translator = Translator()


### getting disctinct transcript from SEAME
### original data came in as individual transcripts save in 'conversation' and 'interview' folder
### data can be found here: https://github.com/peter-yh-wu/code_mix/tree/master/tasks/SEAME/data
# all_files = os.listdir("conversation/")
# df = pd.DataFrame(columns=['Category', 'Transcript'])
# d = {}
# for fname in all_files:
#     with open("conversation/"+fname, 'r', encoding="utf8") as f:
#         lines = [line.rstrip() for line in f]
#         for line in lines:
#             items = line.split(' ')
#             trans = ' '.join(items[4:])
#             if trans not in d:
#                 d[trans] = items[3]

# all_files = os.listdir("interview/")

# for fname in all_files:
#     with open("interview/"+fname, 'r', encoding="utf8") as f:
#         lines = [line.rstrip() for line in f]
#         for line in lines:
#             items = line.split(' ')
#             trans = ' '.join(items[4:])
#             if trans not in d:
#                 d[trans] = items[3]

# for key, value in d.items():
#     df.loc[len(df)] = [value, key]

# df.to_csv('SEAME.csv', index=False)


### Separeting Mono from CS
with open('SEAME.csv', newline='', encoding="utf8") as f:
    reader = csv.reader(f)
    data = list(reader)

mono = [['Language', 'Content']]
cs = [['Language', 'Content']]

for i in data:
    if i[0] == 'CS':
        cs.append(i)
        pass
    else:
        mono.append(i)

with open('CS.csv', 'w', encoding="utf8", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in cs:
            writer.writerow(i)

with open('MONO.csv', 'w', encoding="utf8", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in mono:
            writer.writerow(i)


### Translate CS into EN and ZH
cs_trans = [['CS', 'EN', 'ZH']]
cs_trans = []

with open('CS.csv', newline='', encoding="utf8") as f:
    reader = csv.reader(f)
    data = list(reader)

for i, val in enumerate(data):
    text = val[1]
    en = translator.translate(text, dest='en').text
    zh = translator.translate(en, dest='zh-CN').text
    cs_trans.append([text, en, zh])
    
with open('CS_translated.csv', 'w', encoding="utf8", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for l in cs_trans:
        writer.writerow(l)


### Splitting Training (50K according to the paper), Testing set (rest of available entries)
with open('CS_translated.csv', newline='', encoding="utf8") as f:
    reader = csv.reader(f)
    data = list(reader)

data.pop(0)

ids = random.sample(range(0, 56928), 6928)
train = []
test = []

for index, line in enumerate(data):
    if index in ids:
        test.append(line)
    else:
        train.append(line)

with open('CS_train.csv', 'w', encoding="utf8", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for l in train:
        writer.writerow(l)
with open('CS_test.csv', 'w', encoding="utf8", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for l in test:
        writer.writerow(l)

