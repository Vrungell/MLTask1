import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import naive_bayes

def fitting_data(x):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts = count_vect.fit_transform(x)
    X_train_tf = tfidf_transformer.fit_transform(X_train_counts)
    return X_train_tf

input_data = open("/Users/uliakaliberda/PycharmProjects/untitled/news_data/news_train.txt", encoding = "utf-8")
test_data = open("/Users/uliakaliberda/PycharmProjects/untitled/news_data/news_test.txt", encoding = "utf-8")
test_opened_data = []
line = test_data.readline()
while line:
   test_opened_data.append(line)
   line = test_data.readline()
parsed_data = []
labels = []
i = 0;
line = input_data.readline()
while line:
    splited=list(re.split("\t",line))
    parsed_data.append(splited[2])
    labels.append(splited[0])
    line = input_data.readline()

parsed_data_predicted = parsed_data[:100]
labels_predicted = labels[:100]

X_train_tf = fitting_data(parsed_data)
#X_predict_tf = fitting_data(parsed_data_predicted)

clf = MultinomialNB().fit(X_train_tf, labels)
predicted = clf.predict(parsed_data_predicted)
np.mean(predicted == labels_predicted)
print(predicted)