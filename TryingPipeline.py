import numpy as np
import re
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier


#working with data

input_data = open("/Users/uliakaliberda/PycharmProjects/untitled/news_data/news_train.txt", encoding = "utf-8")
test_data = open("/Users/uliakaliberda/PycharmProjects/untitled/news_data/news_test.txt", encoding = "utf-8")

pred_test = open("pred_test.txt", 'w')

test_opened_data = []
line = test_data.readline()
while line:
   test_opened_data.append(line)
   line = test_data.readline()

parsed_data = []
labels = []


line = input_data.readline()
while line:
    splited=list(re.split("\t",line))
    parsed_data.append(splited[2])
    labels.append(splited[0])
    line = input_data.readline()
n = len(parsed_data)

indexes = []
for i in range(n):
    indexes.append(i)

shuffle(indexes)

parsed_add = []
labels_add = []

for i in indexes:
    parsed_add.append(parsed_data[i])
    labels_add.append(labels[i])

for i in range(n):
    parsed_data.append(parsed_add[i])
    labels.append(labels_add[i])

#making classifier

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                     ('clf-svm', SGDClassifier(loss="log", penalty="l2", epsilon=0.0001, alpha = 0.0000001))])

text_clf.fit(parsed_data, labels)

parsed_data_predicted = parsed_add[:10000]
labels_predicted = labels_add[:10000]

#predicted = text_clf.predict(parsed_data_predicted)
for word in predicted:
   pred_test.write(word+'\n')


#print(np.mean(predicted == labels_predicted))

