#!/home/yiqi.she/venv/common/bin/python3

import sys, os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import LabelKFold

data = np.loadtxt("dataset", dtype = np.float)
X = data[:,:-1]
y = data[:,-1:]
y = np.reshape(y,[81,])
x_train = X[:63,:]
x_test = X[63:,:]
y_train = y[:63]
y_test = y[63:]
for a in range(-3,3,1):
    c = 10**a
    for b in range(-3,3,1):
        g = 10**b
        clf = SVC(C=c,gamma=g)
        clf.fit(x_train, y_train)
        fitdata = clf.predict(x_test)
        print (clf.score(x_test,y_test))
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
        #os.system("cat result")
