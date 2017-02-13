#!/home/yiqi.she/venv/common/bin/python3

import sys, os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import LabelKFold

data = np.loadtxt("dataset_sample", dtype = np.float)
X = data[:,:-9]
y = data[:,-9:]
x_train = X[:72,:]
x_test = X[72:,:]
y_train = y[:72,:]
y_test = y[72:,:]
clf = SVC()
print (x_train)
print (y_train)
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
os.system("cat result")
