from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from func import imputation_uncertainty,multiple_imputation,missing_value_generator,produce_NA,importance,class_var,trainmodel,trainmodel1,testmodel,knn_imputation,getbias
import pandas as pd
from sklearn import model_selection
import heapq
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


data_df = pd.read_csv('./dataset/adult_clean.csv', delimiter=',', header=None)
data = data_df.values

#split dataset

c_train,c_test = model_selection.train_test_split(data,test_size = 0.4)
traindata=c_train
#c_train,c_test = model_selection.train_test_split(c_train,test_size = 0.5)
#traindata=c_train


new_train,new_test = model_selection.train_test_split(traindata,test_size = 0.5)

X_train = new_train[:,1:]
Y_train = new_train[:,0]






#classifier= svm.SVC(kernel='linear', C=1, gamma='auto')
#classifier = LogisticRegression(random_state=0)

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, Y_train)






X_full=new_test

#完整数据跟inputed的数据分别分类，然后看结果的差别
"""完整数据"""
Male = np.array(list(filter(lambda x: x[1] == 1, X_full)))
Female = np.array(list(filter(lambda x: x[1] == 0, X_full)))
X_testM = Male[:,1:]
Y_testM = Male[:,0]
    #Calculate the accurancy without miss data
Y_predM1 = classifier.predict(X_testM)


cnf_matrixM = confusion_matrix(Y_testM, Y_predM1)
#print(cnf_matrixM)
X_testF = Female[:,1:]
Y_testF = Female[:,0]
    #Calculate the accurancy without miss data
Y_predF1 = classifier.predict(X_testF)


cnf_matrixF = confusion_matrix(Y_testF, Y_predF1)
#print(cnf_matrixF)


accurancy=(cnf_matrixM[1][1]+cnf_matrixM[0][0]+cnf_matrixF[1][1]+cnf_matrixF[0][0])/(sum(sum(cnf_matrixF))+sum(sum(cnf_matrixM)))
#print("Accurancy full:",accurancy)
#bias=abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))
bias=abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))





fulldata=np.vstack((Male, Female))




"""缺失数据"""
X_protect = new_test[:,:2]
X_miss = new_test[:,2:]


X_miss=produce_NA(X_miss, 0.6, mecha="MCAR", opt=None, p_obs=None, q=None)
#保持protect feature完整之后对其他feature生产确实数据，然后重新整合
x=[]
for i in range(len(X_miss)):
    x.append(np.insert(X_miss[i].tolist(),0,X_protect[i]))
X_miss = np.array(x)





"""数据填充Imputation"""
#另一种imputation
m_imputations=5

X_imputed,imputation_var,imputed_list1=knn_imputation(X_miss, m_imputations)
#X_imputed,imputation_var,imputed_list1=multiple_imputation(X_miss,m_imputations, 1)



Male = np.array(list(filter(lambda x: x[1] == 1, X_imputed)))
Female = np.array(list(filter(lambda x: x[1] == 0, X_imputed)))
X_testM = Male[:,1:]
Y_testM = Male[:,0]
    #Calculate the accurancy without miss data
Y_predM2 = classifier.predict(X_testM)

cnf_matrixM = confusion_matrix(Y_testM, Y_predM2)
#print(cnf_matrixM)
X_testF = Female[:,1:]
Y_testF = Female[:,0]
    #Calculate the accurancy without miss data
Y_predF2 = classifier.predict(X_testF)

cnf_matrixF = confusion_matrix(Y_testF, Y_predF2)
#print(cnf_matrixF)

accurancy=(cnf_matrixM[1][1]+cnf_matrixM[0][0]+cnf_matrixF[1][1]+cnf_matrixF[0][0])/(sum(sum(cnf_matrixF))+sum(sum(cnf_matrixM)))
#print("Imputed accurancy:",accurancy)
#bias=abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))
bias2=abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))
#print("Imputed bias:",bias2)



"""Accurancy importance"""


#获取两个方差
global class_var



class_list=[classifier.predict(i[:,1:]) for i in imputed_list1]
class_var=np.var(class_list,axis=0,ddof=1)

imputation_var=imputation_var[:,np.newaxis].T
class_var=class_var[:,np.newaxis].T




#将两个方差放到feature里
for i in range(len(imputation_var)):
    trainmodel=np.append(X_imputed,imputation_var.T,axis=1)
    trainmodel=np.append(trainmodel,class_var.T,axis=1)
    
importance1=[]
for i in range(len(class_list[0])):
    k=0
    for j in range(len(class_list)):
        if class_list[j][i]==new_test[:,0][i]:
            k+=1
    importance1.append(1-k/m_imputations)


accmodel = LinearRegression()
accmodel.fit(trainmodel, importance1)


Male_train = np.array(list(filter(lambda x: x[1] == 1, trainmodel)))
Female_train = np.array(list(filter(lambda x: x[1] == 0, trainmodel)))



"""importance"""

importance=[]
for i in range(len(Y_predM1)):
    if Y_predM2[i]==Y_predM1[i]:
        importance.append(0)
        
    elif Y_testM[i]==1:
        if Y_predM2[i]==1:
            importance.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-(cnf_matrixM[1][1]+1)/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
        else:
            importance.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]+1))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
    else:
        if Y_predM2[i]==0:
            importance.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]+1))-bias2)
        else:
            importance.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-(cnf_matrixM[0][1]+1)/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)

for i in range(len(Y_predF1)):
    if Y_predF2[i]==Y_predF1[i]:
        importance.append(0)
        
    elif Y_testF[i]==1:
        if Y_predF2[i]==1:
            importance.append(abs((cnf_matrixF[1][1]+1)/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
        else:
            importance.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0]+1)-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
    else:
        if Y_predF2[i]==0:
            importance.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0]+1)-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
        else:
            importance.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs((cnf_matrixF[0][1]+1)/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)

aldata=np.vstack((Male_train, Female_train))



"""训练importance model"""
biasmodel = LinearRegression()
biasmodel.fit(aldata, importance)



"""Test Test Test"""
accfull,biasfull,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF = getbias(c_test,classifier)
print("Accurancy full:",accfull)
print("Bias full:",biasfull)


X_protect = c_test[:,:2]
X_miss = c_test[:,2:]
X_miss=produce_NA(X_miss, 0.6, mecha="MCAR", opt=None, p_obs=None, q=None)
x=[]
for i in range(len(X_miss)):
    x.append(np.insert(X_miss[i].tolist(),0,X_protect[i]))
X_miss = np.array(x)


"""数据填充Imputation"""
m_imputations=5
X_imputed,imputation_var,imputed_list1=knn_imputation(X_miss, m_imputations)
#X_imputed,imputation_var,imputed_list1=multiple_imputation(X_miss,m_imputations, 1)


var_imp=imputation_uncertainty(imputed_list1)


X_al=X_imputed.copy()
X_random=X_imputed.copy()
X_imputed3=X_imputed.copy()
X_imputed2=X_imputed.copy()


accmiss,biasmiss,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF = getbias(X_imputed,classifier)
print("Accurancy miss:",accmiss)
print("Bias miss:",biasmiss)




#获取两个方差

class_list=[classifier.predict(i[:,1:]) for i in imputed_list1]
class_var=np.var(class_list,axis=0,ddof=1)

imputation_var=imputation_var[:,np.newaxis].T
class_var=class_var[:,np.newaxis].T



#将两个方差放到feature里
for i in range(len(imputation_var)):
    testmodel=np.append(X_imputed,imputation_var.T,axis=1)
    testmodel=np.append(testmodel,class_var.T,axis=1)

pre_accimp=accmodel.predict(testmodel)

preimp=biasmodel.predict(testmodel)
pre_biasimp=preimp.tolist()

"""将男女分组"""
accimp_M,accimp_F=[],[]
for i in range(len(pre_accimp)):
    if testmodel[i][1]==1:
        accimp_M.append(pre_accimp[i])
    else:
        accimp_F.append(pre_accimp[i])
    
Male_imputed = np.array(list(filter(lambda x: x[1] == 1, X_imputed)))
Female_imputed = np.array(list(filter(lambda x: x[1] == 0, X_imputed)))
    
Male_full = np.array(list(filter(lambda x: x[1] == 1, c_test)))
Female_full = np.array(list(filter(lambda x: x[1] == 0, c_test)))

x_axis_data = []
y_axis_data = []
y_axis_data1 = []
y_axis_data2 = []
y_axis_data3 = []
y_axis_data4 = []
y_axis_data5 = []



dot=[]


k=0
while k<=0.4:
    X_al=X_imputed.copy()
    maximp1 = sorted(range(len(pre_biasimp)), key = lambda sub: pre_biasimp[sub])[:(int(len(pre_biasimp)*k))]
    for i in maximp1:
        X_al[i]=c_test[i]
    
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_al,classifier)
    
    
    
    x_axis_data.append(k)
    y_axis_data.append(newbias)
    k+=0.1

    
plt.plot(x_axis_data, y_axis_data, 'b.-', alpha=0.5, linewidth=2, label='FDA_fair')


k=0
while k<=0.4:
    X_al=X_imputed.copy()
    maximp1 = sorted(range(len(var_imp)), key = lambda sub: var_imp[sub])[:(int(len(var_imp)*k))]
    for i in maximp1:
        X_al[i]=c_test[i]
    
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_al,classifier)

    y_axis_data3.append(newbias)
    k+=0.1

    
plt.plot(x_axis_data, y_axis_data3, 'y.-', alpha=0.5, linewidth=2, label='AVID')



k=0
while k<=0.4:
    X_al=X_imputed.copy()
    randomchoose=random.sample(range(len(preimp)),int(len(preimp)*k))
    for i in randomchoose:
        X_al[i]=c_test[i]
    
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_al,classifier)

 
    y_axis_data4.append(newbias)
    k+=0.1

    
plt.plot(x_axis_data, y_axis_data4, 'k.-', alpha=0.5, linewidth=2, label='Random')




k=0
while k<=0.4:
    X_imputed2=X_imputed.copy()
    maximp2 = sorted(range(len(pre_accimp)), key = lambda sub: pre_accimp[sub])[(int(len(pre_accimp)*(1-k))):]
    for i in maximp2:
        X_imputed2[i]=c_test[i]
    
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_imputed2,classifier)

    y_axis_data1.append(newbias)
    k+=0.1

    
plt.plot(x_axis_data, y_axis_data1, 'r.-', alpha=0.5, linewidth=2, label='FDA_acc')




k=0
while k<=0.4:
    X_imputed3=X_imputed.copy()
    maximp3 = sorted(range(len(pre_accimp)), key = lambda sub: pre_accimp[sub])[(int(len(pre_accimp)*(1-k/2))):]
    for i in maximp3:
        X_imputed3[i]=c_test[i]
    
    for i in sorted(range(len(pre_biasimp)), key = lambda sub: pre_biasimp[sub])[:(int(len(pre_biasimp)*k*0.5))]:
        X_imputed3[i]=c_test[i]
    
    
    
    
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_imputed3,classifier)
    
    #x_axis_data.append(k)
    y_axis_data2.append(newbias)
    k+=0.1

    
plt.plot(x_axis_data, y_axis_data2, 'g.-', alpha=0.5, linewidth=2, label='Comb1:1')

k=0
while k<=0.4:
    for i in sorted(range(len(accimp_F)), key = lambda sub: accimp_F[sub])[(int(len(accimp_F)*(1-k))):]:
        Female_imputed[i]=Female_full[i]
    for i in sorted(range(len(accimp_M)), key = lambda sub: accimp_M[sub])[(int(len(accimp_M)*(1-k))):]:
        Male_imputed[i]=Male_full[i]
        
    fulldata=np.vstack((Male_imputed, Female_imputed))
        
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(fulldata,classifier)
    
    
    #x_axis_data.append(k)
    y_axis_data5.append(newbias)
    k+=0.1

    
plt.plot(x_axis_data, y_axis_data5, 'm.-', alpha=0.5, linewidth=2, label='B FDA_acc')


plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.ylabel("Bias",fontsize=14)
plt.xlabel("Selection ratio",fontsize=14)
plt.rcParams['font.size'] = 13
plt.legend()
plt.show()



