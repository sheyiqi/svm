#!/home/yiqi.she/venv/common/bin/python3

import sys, os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import LabelKFold

def my_kernel(X, Y):
    """
    We create a custom kernel:

                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    """
    return np.dot(X, Y.T)

data = np.loadtxt("dataset", dtype = np.float)
X = data[:,:-1]
y = data[:,-1:]
y = np.reshape(y,[81,])
x_train = X[:63,:]
x_test = X[63:,:]
y_train = y[:63]
y_test = y[63:]
clf = SVC(kernel = my_kernel)
clf.fit(x_train, y_train)
fitdata = clf.predict(x_test)
#print (clf.score(x_test,y_test))
#print (clf.predict_proba(X))
#print ("clf.support_")
#print (clf.support_)
#print ("clf.support_vectors_")
#print (clf.support_vectors_)
#print ("clf.n_support_")
#print (clf.n_support_)
#print ("clf.dual_coef_")
#print (clf.dual_coef_)
#print ("clf.intercept_")
#print (clf.intercept_)
os.system("rm result")
os.system("touch result")
f = open('result','a+')
f.write("real__value"+str(y_test)+'\n')
f.write("predict_val"+str(fitdata)+'\n')
f.close()
os.system("cat result")
