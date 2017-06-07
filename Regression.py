import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

def load_data(filename):
    #load data
    reader = csv.reader(open(filename, "rb"), delimiter=",")
    d = list(reader)

    data = np.array(d)
    X = data[:,0:23]
    y = data[:,23]
    y = np.delete(y, 0, 0)
    X = np.delete(X, 0, 0)
    y.shape = (len(y), 1)
    y = y.astype("float")

    biasone = np.ones((np.shape(X)[0], 1))
    X = np.concatenate((biasone, X), axis=1)
    return X, y

def replace_NA(X, col):
    rownum = np.shape(X)[0]
    a = []
    for i in range(rownum):
        if X[i][col] != 'NA':
            a.append(float(X[i][col]))

    a = np.asarray(a)
    # print a
    return np.mean(a)

def preprocess_data(X):
    # Race: white, hispanic, asian, black, american indian/alaska, other pacific islanders, more than one race
    # Race: 0, 1, 2, 3, 4, 5, 6
    rownum = np.shape(X)[0]
    colnum = np.shape(X)[1]
    mean18 = replace_NA(X, 18)
    mean19 = replace_NA(X, 19)
    mean23 = replace_NA(X, 23)
    for i in range(rownum):
        for j in range(colnum):
            if j == 3:
                if X[i][j] == 'NA':
                    X[i][j] = str(float(randint(0, 6)))
                elif X[i][j] == 'White':
                    X[i][j] = '0.0'
                elif X[i][j] == 'Hispanic':
                    X[i][j] = '1.0'
                elif X[i][j] == 'Asian':
                    X[i][j] = '2.0'
                elif X[i][j] == 'Black':
                    X[i][j] = '3.0'
                elif X[i][j] == 'American Indian/Alaska Native':
                    X[i][j] = '4.0'
                elif X[i][j] == 'Native Hawaiian/Other Pacific Islander':
                    X[i][j] = '5.0'
                elif X[i][j] == 'More than one race':
                    X[i][j] = '6.0'
            elif j == 18:
                if X[i][j] == 'NA':
                    X[i][j] = str(mean18)
            elif j == 19:
                if X[i][j] == 'NA':
                    X[i][j] = str(mean19)
            elif j == 23:
                if X[i][j] == 'NA':
                    X[i][j] = str(mean23)
            else:
                if X[i][j] == 'NA':
                    X[i][j] = str(float(randint(0, 1)))
                    # X[i][j] = 0.0
                else:
                    X[i][j] = str(float(X[i][j]))

    X = X.astype("float")

    return X

def calculate_w(X, y):
    print "Calculating w"
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    print "done"
    return w

def test_error(X, y, w):
    Y_test = np.dot(X, w)
    rownum = np.shape(y)[0]
    MSE = []
    error_rate = []
    for i in range(rownum):
        error_rate.append(abs(Y_test[i][0] - y[i][0]) / y[i][0])
        MSE.append((Y_test[i][0] - y[i][0])**2)

    MSE = np.asarray(MSE)
    error_rate = np.asarray(error_rate)
    # print error_rate
    MSE = np.mean(MSE)
    avg_error_rate = np.mean(error_rate)
    # print MSE
    return MSE, avg_error_rate


X, y = load_data("pisa2009train.csv")
X = preprocess_data(X)
w = calculate_w(X, y)


print "Start Testing: "
print "Loading Data: "
X, y = load_data("pisa2009test.csv")
print "Preprocess Data: "
X = preprocess_data(X)
mean_squared_error, avg_err_rate = test_error(X, y, w)
print "Mean Squared Error for the testing data is: " + str(mean_squared_error)
print "Average Error Rate: " + str(avg_err_rate)
