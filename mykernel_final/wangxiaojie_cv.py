#!/home/yiqi.she/venv/common/bin/python3
import sys, os
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

def my_kernel(X1, Y1):
    """
    We create a custom kernel:

                 (6  0  0  0  0  0  0  0 )
                 (0  4  0  0  0  0  0  0 )
                 (0  0  2  0  0  0  0  0 )
                 (0  0  0  1  0  0  0  0 )
    k(X, Y) = X  (0  0  0  0  2  0  0  0 ) Y.T
                 (0  0  0  0  0  51 0  0 )
                 (0  0  0  0  0  0  18 0 )
                 (0  0  0  0  0  0  0  10)
    """
    return np.dot(np.dot(X1,[[6,0,0,0,0,0,0,0],[0,4,0,0,0,0,0,0],[0,0,2,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,2,0,0,0],[0,0,0,0,0,51,0,0],[0,0,0,0,0,0,18,0],[0,0,0,0,0,0,0,10]]), Y1.T)

num = -1
kfold1 = KFold(9, n_folds=9,shuffle=True)
data1 = np.loadtxt("d1", dtype = np.float)
X1 = data1[:,:num]
y1 = data1[:,num:]
data2 = np.loadtxt("d2", dtype = np.float)
X2 = data2[:,:num]
y2 = data2[:,num:]
data3 = np.loadtxt("d3", dtype = np.float)
X3 = data3[:,:num]
y3 = data3[:,num:]
data4 = np.loadtxt("d4", dtype = np.float)
X4 = data4[:,:num]
y4 = data4[:,num:]
data5 = np.loadtxt("d5", dtype = np.float)
X5 = data5[:,:num]
y5 = data5[:,num:]
data6 = np.loadtxt("d6", dtype = np.float)
X6 = data6[:,:num]
y6 = data6[:,num:]
data7 = np.loadtxt("d7", dtype = np.float)
X7 = data7[:,:num]
y7 = data7[:,num:]
data8 = np.loadtxt("d8", dtype = np.float)
X8 = data8[:,:num]
y8 = data8[:,num:]
data9 = np.loadtxt("d9", dtype = np.float)
X9 = data9[:,:num]
y9 = data9[:,num:]

os.system("rm result")
array1= np.zeros((9,9))
array2= np.zeros((9,9))
array3= np.zeros((9,9))
array4= np.zeros((9,9))
array5= np.zeros((9,9))

for a in range(-3,3,1):
    c = 10**a
    for b in range(-3,3,1):
        g = 10**b
        #print ("a=")
        #print (a)
        #print ("b=")
        #print (b)
        score1 =0
        score2 =0
        score3 =0
        score4 =0
        score5 =0
        for train, test in kfold1:
            #print("TRAIN:", train, "TEST:", test)
            X1_train, X1_test = X1[train], X1[test]
            y1_train, y1_test = y1[train], y1[test]
            X2_train, X2_test = X2[train], X2[test]
            y2_train, y2_test = y2[train], y2[test]
            X3_train, X3_test = X3[train], X3[test]
            y3_train, y3_test = y3[train], y3[test]
            X4_train, X4_test = X4[train], X4[test]
            y4_train, y4_test = y4[train], y4[test]
            X5_train, X5_test = X5[train], X5[test]
            y5_train, y5_test = y5[train], y5[test]
            X6_train, X6_test = X6[train], X6[test]
            y6_train, y6_test = y6[train], y6[test]
            X7_train, X7_test = X7[train], X7[test]
            y7_train, y7_test = y7[train], y7[test]
            X8_train, X8_test = X8[train], X8[test]
            y8_train, y8_test = y8[train], y8[test]
            X9_train, X9_test = X9[train], X9[test]
            y9_train, y9_test = y9[train], y9[test]
            X_train = np.concatenate((X1_train,X2_train,X3_train,X4_train,X5_train,X6_train,X7_train,X8_train,X9_train),axis=0)        
            y_train = np.concatenate((y1_train,y2_train,y3_train,y4_train,y5_train,y6_train,y7_train,y8_train,y9_train),axis=0)        
            X_test = np.concatenate((X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,X7_test,X8_test,X9_test),axis=0)       
            y_test = np.concatenate((y1_test,y2_test,y3_test,y4_test,y5_test,y6_test,y7_test,y8_test,y9_test),axis=0)        
            #print (X_train.shape)
            #print (y_train.shape)
            #print (X_test.shape)
            #print (y_test.shape)
            clf1 = SVC(C = c, gamma = g, kernel = my_kernel)
            clf2 = SVC(C = c, gamma = g, kernel = 'linear')
            clf3 = SVC(C = c, gamma = g, kernel = 'poly')
            clf4 = SVC(C = c, gamma = g, )
            clf5 = SVC(C = c, gamma = g, kernel = 'sigmoid')
            clf1.fit(X_train,y_train)
            clf2.fit(X_train,y_train)
            clf3.fit(X_train,y_train)
            clf4.fit(X_train,y_train)
            clf5.fit(X_train,y_train)
            score1 += clf1.score(X_test,y_test)/9
            score2 += clf2.score(X_test,y_test)/9
            score3 += clf3.score(X_test,y_test)/9
            score4 += clf4.score(X_test,y_test)/9
            score5 += clf5.score(X_test,y_test)/9
            fitdata1 =  clf1.predict(X_test)
            fitdata2 =  clf2.predict(X_test)
            fitdata3 =  clf3.predict(X_test)
            fitdata4 =  clf4.predict(X_test)
            fitdata5 =  clf5.predict(X_test)
            for i in range(len(fitdata1)):
                array1[int(y_test[i][0])-1][int(fitdata1[i])-1]+=1
            for i in range(len(fitdata2)):
                array2[int(y_test[i][0])-1][int(fitdata2[i])-1]+=1
            for i in range(len(fitdata3)):
                array3[int(y_test[i][0])-1][int(fitdata3[i])-1]+=1
            for i in range(len(fitdata4)):
                array4[int(y_test[i][0])-1][int(fitdata4[i])-1]+=1
            for i in range(len(fitdata5)):
                array5[int(y_test[i][0])-1][int(fitdata5[i])-1]+=1
print (array1)
print (array2)
print (array3)
print (array4)
print (array5)

        #print ("my_kernel")
        #print (score1)
        #print ("linear")
        #print (score2)
        #print ("poly")
        #print (score3)
        #print ("rbf")
        #print (score4)
        #print ("sigmoid")
        #print (score5)
        
#os.system("touch result")
#f = open('result','a+')
#f.write("c="+str(c)+'\n')
#f.write("g="+str(g)+'\n')
#f.write("my_kernel"+'\t'+str(score1)+'\n')
#f.write("linear"+'\t'+'\t'+str(score2)+'\n')
#f.write("poly"+'\t'+'\t'+str(score3)+'\n')
#f.write("rbf"+'\t'+'\t'+str(score4)+'\n')
#f.write("sigmoid"+'\t'+'\t'+str(score5)+'\n')
#f.write("--------------------------------------------------"+'\n')
#f.close()

#os.system("cat result")
