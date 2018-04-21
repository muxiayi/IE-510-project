import numpy as np
import pandas as pd

#read data
df = pd.read_csv('originaldata.csv')
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

#Complete data set:
X = df[df.columns[2:32]]
Y = df['diagnosis']
Y = Y.values.reshape(Y.shape[0],1)

# normalizing original data:
original_X = X.loc[0:570,X.columns[0:]]
mean = original_X.mean()
std_error = original_X.std()
original_X = (original_X - mean)/std_error
original_X.to_csv('normal_original_data.csv')


#train set (80%):
train_X = X.loc[0:456,X.columns[0:]]
train_Y = Y[0:457]

#test set (20%):
test_X = X.loc[0:114,X.columns[0:]]
test_Y = Y[0:115]

#train set normalizing:
mean = train_X.mean()
std_error = train_X.std()
train_X = (train_X - mean)/std_error
train_X.to_csv('normaltrainXdata.csv')

#test set normalizing:
mean = test_X.mean()
std_error = test_X.std()
test_X = (test_X - mean)/std_error
test_X.to_csv('normaltestXdata.csv')
