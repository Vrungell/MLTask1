# -*- coding: utf-8 -*-
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import naive_bayes
input_data = open("/Users/uliakaliberda/PycharmProjects/untitled/news_data/news_train.txt", encoding = "utf-8")
test_data = open("/Users/uliakaliberda/PycharmProjects/untitled/news_data/news_test.txt", encoding = "utf-8")
test_opened_data = []
line = test_data.readline()
while line:
   test_opened_data.append(line)
   line = test_data.readline()

'''i = 0;
for line in test_data:
    test_opened_data.append(' '.join(line.split(' ')[0:]))
print(test_opened_data)'''

'''words = set()
labels_set = set()
words_dict = {}
labels_dict = {}
parsed_test_data = []
for line in test_data:
    parsed_test_data.append(line)
parsed_test_data = np.asarray(parsed_test_data)'''#fordict
parsed_data = []
labels = []
i = 0;
line = input_data.readline()
while line:
    splited=list(re.split("\t",line))

    parsed_data.append(splited[2])
    labels.append(splited[0])
    line = input_data.readline()

parsed_data=np.array(parsed_data)

'''while line:
   parsed_data.append(re.findall(r"[\w']+",line))
  # parsed_data[i] = np.array(parsed_data[i])
   labels.append(parsed_data[i][0])
   line = input_data.readline()
   i+=1'''
#parsed_data = np.array(parsed_data)

'''i = 0;
for line in input_data:
    parsed_data.append(' '.join(line.split(' ')[1:]))
    #for k in parsed_data[i].split():
    #    words.add(k)
    labels.append(line.split()[0])
    #parsed_data[i] = parsed_data[i]
    #labels_set.add(labels[i])
    i+=1'''
#labels = np.array(labels)
'''i = 0;
for word in words:
    words_dict[word] = i;
    i+=1;
i = 0;
for label in labels_set:
    labels_dict[label] = i;
    i+=1;
print(words_dict, labels_dict)'''#fordict
count_vect = CountVectorizer()
training_file_sparse = count_vect.fit_transform(parsed_data)
tf_transformer = TfidfTransformer(use_idf=False).fit(training_file_sparse)
X_train_tf = tf_transformer.transform(training_file_sparse)
tfidf_transformer = TfidfTransformer()
training_file_tf = tfidf_transformer.fit_transform(training_file_sparse)


NB = naive_bayes.MultinomialNB().fit(training_file_tf, np.array(labels))
for i in range(len(parsed_data)):
    NB.predict(np.array(parsed_data[i]))#bayes'''


'''i = 0;
element = ''
parsed_data = []
while (element in input_data)!='/n':
    word = ''
    parsed_data.append([])
    while (element in input_data!=' '):
        word+=element;
    parsed_data[i].append(word)
    i+=1
print(element, word)
line = input_data.readline()#.decode('utf8')

while (line!=''):
    parsed_data.append(line)
    line = input_data.readline()#.decode('utf8')

print(parsed_data)'''#decoding
#print(input_data)