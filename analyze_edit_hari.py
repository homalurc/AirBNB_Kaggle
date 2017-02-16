import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.datasets import load_iris
from sklearn import preprocessing
import csv
from sklearn.naive_bayes import GaussianNB

# train_users = pd.read_csv('train_users_2.csv')
# test_users = pd.read_csv('test_users.csv')
#
# print("We have", train_users.shape[0], "users in the training set and",
#       test_users.shape[0], "in the test set.")
# print("In total we have", train_users.shape[0] + test_users.shape[0], "users.")



# Normalize the data attributes for the Iris dataset.

# load the iris dataset
# iris = load_iris()
# print(train_users.shape)
# print(test_users.shape)


dataset = {}

dataset1 = {}
dataset2  ={}
i = 0

columns = ['id', 'date_account_created', 'timestamp_first_active', 'date_first_booking', 'gender', 'age', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'country_destination']
with open('Data/train_users_2.csv', 'rb') as f:
    reader = csv.reader(f)
    row_index = 0
    for row in reader:
        for col_index in range(0, len(columns)-1):
            col_name = columns[col_index]
            col_val = row[col_index]
            if row_index == 0:
                dataset[col_name] = []
            else:
                dataset[col_name].append(col_val)
        col_index = len(columns)-1
        col_name = columns[col_index]
        col_val = row[col_index]
        if row_index == 0:
            dataset2[col_name] = []
        else:
            dataset2[col_name].append(col_val)
        row_index += 1
        # if row_index == 100:
        #     break

l = preprocessing.LabelEncoder()

non_string = ['date_first_booking', 'timestamp_first_active', 'date_account_created']
for col in dataset:
    #print col
    #date_first_booking,  timestamp_first_active
    if not col in non_string:
        X_mod = l.fit_transform(dataset[col])
        dataset[col] = X_mod

for col in dataset1:
    #print col
    #date_first_booking,  timestamp_first_active
    if not col in non_string:
        X_mod = l.fit_transform(dataset1[col])
        dataset1[col] = X_mod
for col in dataset2:
    #print col
    #date_first_booking,  timestamp_first_active
    if not col in non_string:
        X_mod = l.fit_transform(dataset2[col])
        dataset2[col] = X_mod

with open('Data/test_users.csv', 'rb') as f:
    reader = csv.reader(f)
    row_index = 0
    for row in reader:
        for col_index in range(0, len(columns)-1):
            col_name = columns[col_index]
            col_val = row[col_index]
            if row_index == 0:
                dataset1[col_name] = []
            else:
                dataset1[col_name].append(col_val)
        row_index += 1
        # if row_index == 100:
        #     break

l = preprocessing.LabelEncoder()

non_string = ['date_first_booking', 'timestamp_first_active', 'date_account_created']
for col in dataset:
    #print col
    #date_first_booking,  timestamp_first_active
    if not col in non_string:
        X_mod = l.fit_transform(dataset[col])
        dataset[col] = X_mod
for col in dataset1:
    #print col
    #date_first_booking,  timestamp_first_active
    if not col in non_string:
        X_mod = l.fit_transform(dataset1[col])
        dataset1[col] = X_mod
for col_index in range(0, len(columns)-1):
    col_name = columns[col_index]
    print dataset[col_name][0]

x=[[]]
for col_index in range(0, len(columns)-1):
    col_name = columns[col_index]

    print dataset1[col_name][0]

for i in range(1,213451):
    list1= []
    for col_index in range(0, len(columns)-1):
        col_name = columns[col_index]
        
        list1.append(dataset[col_name][i])
    x.append(list1)
y=[[]]
for i in range(1,62096):
    list1= []
    for col_index in range(0, len(columns)-1):
        col_name = columns[col_index]
        
        list1.append(dataset1[col_name][i])
    y.append(list1)
print x
X=np.array(x)
X1=np.array(dataset2.values())
X1=np.reshape(X1,(213451,))
Y=np.array(y)

print X.shape
print X1.shape
print Y.shape
gnb = GaussianNB()
y_pred = gnb.fit(X,X1).predict(Y)

print y_pred[0]

#y_pred = gnb.fit(dataset, iris.target).predict(iris.data)

#
# normalized_X = preprocessing.normalize(X_mod, axis=0)
# print normalized_X
