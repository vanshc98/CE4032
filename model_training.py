###
"""
# Random Forest Regressor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('datasets/train_output.csv')
train_y = pd.DataFrame()
train_y['DURATION'] = train['DURATION']
train = train.drop(columns = ['DURATION','DATE','ORIGIN_CALL','ORIGIN_STAND','TRIP_ID'])

test = pd.read_csv('datasets/test_output.csv')
test = test.drop(columns = ['DATE','ORIGIN_CALL','ORIGIN_STAND','TRIP_ID'])
submission = pd.DataFrame()
submission['TRIP_ID'] = test['TRIP_ID']

# Xtrain = train
# Ytrain = train_y
# Xtest = test

clf = RandomForestRegressor(n_estimators=50, n_jobs=4, random_state=21)
clf.fit(train, train_y)

y_pred = clf.predict(test)
y_final = []
for i in range(len(y_pred)):
    temp = y_pred[i] * 15
    y_final.append(temp)

submission['TRAVEL_TIME'] = y_final

submission.to_csv('datasets/rf_submission.csv', index=False)

#################
# Score: 0.84100
#################
"""

# Feedforward Neural Network

import csv
import numpy as np
import tensorflow as tf
import math
import pylab as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

NUM_FEATURES = 24
no_labels = 1
batch_size = 128
learning_rate = 0.01
epochs = 40
num_neurons = 50
decay = math.pow(10,-6)
#decay = 0
seed = 10
np.random.seed(seed)

def ffn(x, hidden_units):
  # Hidden
  with tf.name_scope('hidden'):
    weights = tf.Variable(tf.truncated_normal([NUM_FEATURES, hidden_units],
                            stddev=1.0 / np.sqrt(float(NUM_FEATURES)), dtype=tf.float32),name='weights')
    biases = tf.Variable(tf.zeros([hidden_units]),name='biases')
    hidden = tf.nn.relu(tf.matmul(x, weights) + biases)
  # Linear
  with tf.name_scope('linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden_units, no_labels],stddev=1.0 / np.sqrt(float(hidden_units)), dtype=tf.float32),name='weights')
    biases = tf.Variable(tf.zeros([no_labels]),name='biases')
    logits = tf.matmul(hidden, weights) + biases
  return logits, weights

def ffn4(x, hidden_units1, hidden_units2):
  # Hidden
  with tf.name_scope('hidden1'):
    weights1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, hidden_units1],
                            stddev=1.0 / np.sqrt(float(NUM_FEATURES)), dtype=tf.float32),name='weights1')
    biases1 = tf.Variable(tf.zeros([hidden_units1]),name='biases1')
    hidden1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)
  with tf.name_scope('hidden2'):
    weights2 = tf.Variable(tf.truncated_normal([hidden_units1, hidden_units2],
                            stddev=1.0 / np.sqrt(float(hidden_units1)), dtype=tf.float32),name='weights2')
    biases2 = tf.Variable(tf.zeros([hidden_units2]),name='biases2')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)
  # Linear
  with tf.name_scope('linear'):
    weights3 = tf.Variable(
        tf.truncated_normal([hidden_units2, no_labels],stddev=1.0 / np.sqrt(float(hidden_units2)), dtype=tf.float32),name='weights3')
    biases3 = tf.Variable(tf.zeros([no_labels]),name='biases3')
    logits = tf.matmul(hidden2, weights3) + biases3
  return logits, weights3

#initialization of datasets
train = pd.read_csv('datasets/modified_train.csv')
train_y = pd.DataFrame()
train_y['TRAVEL_TIME'] = train['DURATION']
train = train.drop(columns = ['DURATION','DATE','END_TIME','ORIGIN_CALL','ORIGIN_STAND','TRIP_ID'])

test = pd.read_csv('datasets/modified_test.csv')
submission = pd.DataFrame()
submission['TRIP_ID'] = test['TRIP_ID']
test = test.drop(columns = ['DATE','END_TIME','ORIGIN_CALL','ORIGIN_STAND','TRIP_ID'])

x_train = train.to_numpy()
y_train = train_y.to_numpy()
x_test = test.to_numpy()

# print(x_train[0:5])
# print(y_train[0:5])

scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
y_train_ = y_train.reshape(len(y_train), no_labels)
#y_test = y_test_.reshape(len(y_test_), no_labels)

# Create the model
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
y_ = tf.placeholder(tf.float32, [None, 1])


# Build the graph for the deep net
#y, weights = ffn(x, num_neurons)
y, weights = ffn4(x, num_neurons, num_neurons)


optimizer = tf.train.AdamOptimizer(learning_rate)
loss = tf.reduce_mean(tf.square(y_ - y) + (decay * tf.nn.l2_loss(weights)))
train_op = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

N = len(x_train)
idx = np.arange(N)
train_err = []
#test_acc = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        np.random.shuffle(idx)
        X_train, Y_train = x_train[idx], y_train_[idx]
        for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
            train_op.run(feed_dict={x: X_train[start:end], y_: Y_train[start:end]})
            #loss_ = loss.eval(feed_dict={x: X_train[start:end], y_: Y_train[start:end]})
        train_error = loss.eval(feed_dict={x: x_train, y_: y_train_})
        train_err.append(train_error)
        #train_acc.append(accuracy.eval(feed_dict={x: x_train, y_: y_train}))
        if i % 1 == 0:
            print('iter %d: train_err: %g'%(i, train_err[i]))
    y_pred = sess.run(y, {x:x_test})

submission['TRAVEL_TIME'] = y_pred
submission.to_csv('datasets/nn_submission.csv', index=False)

#plot learning curves
plt.figure(1)
plt.plot(range(epochs), train_err, label = 'train error')
#plt.plot(range(epochs), test_acc,  label = 'test accuracy')
plt.xlabel(str(epochs) + ' iterations')
plt.ylabel('Accuracy')
plt.show()
