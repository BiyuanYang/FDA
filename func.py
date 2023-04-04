import numpy as np
from numpy.lib.function_base import select
from scipy.stats import poisson
from scipy.special import entr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import sys
import csv
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences



def missing_value_generator(X, missing_rate, seed):
    row_num = X.shape[0]
    column_num = X.shape[1]
    missing_value_average_each_row = column_num * (missing_rate/100)

    np.random.seed(seed)
    poisson_dist = poisson.rvs(mu = missing_value_average_each_row, size = row_num, random_state = seed)
    poisson_dist = np.clip(poisson_dist, 0, X.shape[1]-1)

    column_idx = np.arange(1,column_num)
    X_missing = X.copy().astype(float)
    for i in range(row_num):
        missing_idx = np.random.choice(column_idx, poisson_dist[i], replace=False)
        for j in missing_idx:
            X_missing[i,j] = np.nan

    return X_missing
    
    

def imputation_uncertainty(imputed_list):
    delta = np.var(imputed_list, axis=0, ddof=1)
    delta = np.sum(delta, axis=1)

    return delta
    
    
def class_var(imputed_list,classifier):
    class_list=[classifier.predict(i) for i in imputed_list]
    class_var=np.var(class_list,axis=0,ddof=1)
    
    return class_var



def trainmodel(X_test1,Y_test1,classifier):


    #X_test1=missing_value_generator(X_test1,30,1)
    X_test1=produce_NA(X_test1, 0.3, mecha="MCAR", opt=None, p_obs=None, q=None)
    m_imputations=5
    X_imputed, imputed_list, mice=multiple_imputation(X_test1, m_imputations, 1)

    imputation_var=imputation_uncertainty(imputed_list)
    imputation_var=imputation_var.tolist()
    #print(imputation_var)



    class_list=[classifier.predict(i) for i in imputed_list]


    global class_var
    class_var=class_var(imputed_list,classifier)
    class_var=class_var.tolist()
    #print(class_var)

    importance=[]
    for i in range(len(class_list[0])):
        k=0
        for j in range(len(class_list)):
            if class_list[j][i]==Y_test1[i]:
                k+=1
        importance.append(1-k/m_imputations)



    #print(class_var[:20])
    #input()
    #print(importance[:20])
    impute_var=imputation_uncertainty(imputed_list)

    X_imputed=X_imputed.tolist()


    with open('otherfeatures.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in X_imputed:
            writer.writerow(row)
            
    data=[list(t) for t in zip(class_var,imputation_var,importance)]
    with open('1.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    #print(data)
    f1 = pd.read_csv('otherfeatures.csv')
    f2 = pd.read_csv('1.csv')
    file = [f1,f2]
    train = pd.concat(file,axis=1)
    train.to_csv("trainmodel" + ".csv", index=0, sep=',')
    trainmodel = pd.read_csv('trainmodel.csv', delimiter=',', header=None)
    return trainmodel

def trainmodel1(X_test1,Y_test1,classifier):


    #X_test1=missing_value_generator(X_test1,30,1)
    X_test1=produce_NA(X_test1, 0.3, mecha="MCAR", opt=None, p_obs=None, q=None)
    m_imputations=5
    X_imputed, imputed_list, mice=multiple_imputation(X_test1, m_imputations, 1)

    imputation_var=imputation_uncertainty(imputed_list)
    imputation_var=imputation_var.tolist()
    #print(imputation_var)



    class_list=[classifier.predict(i) for i in imputed_list]


    global class_var
    class_var=class_var(imputed_list,classifier)
    class_var=class_var.tolist()
    #print(class_var)

    importance=[]
    for i in range(len(class_list[0])):
        k=0
        for j in range(len(class_list)):
            if class_list[j][i]==Y_test1[i]:
                k+=1
        #importance.append(1-k/m_imputations)
        
        if X_imputed[i][0]==1:
            importance.append((1-k/m_imputations)*2)
        else:
            importance.append(1-k/m_imputations)



    #print(class_var[:20])
    #input()
    #print(importance[:20])
    impute_var=imputation_uncertainty(imputed_list)

    X_imputed=X_imputed.tolist()


    with open('otherfeatures.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in X_imputed:
            writer.writerow(row)
            
    data=[list(t) for t in zip(class_var,imputation_var,importance)]
    with open('1.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    #print(data)
    f1 = pd.read_csv('otherfeatures.csv')
    f2 = pd.read_csv('1.csv')
    file = [f1,f2]
    train = pd.concat(file,axis=1)
    train.to_csv("trainmodel" + ".csv", index=0, sep=',')
    trainmodel = pd.read_csv('trainmodel.csv', delimiter=',', header=None)
    return trainmodel





def testmodel(X_test,classifier):


    #X_test=missing_value_generator(X_test,30,1)
    m_imputations=5
    X_imputed, imputed_list, mice=multiple_imputation(X_test, m_imputations, 1)

    imputation_var=imputation_uncertainty(imputed_list)
    imputation_var=imputation_var.tolist()
    #print(imputation_var)



    class_list=[classifier.predict(i) for i in imputed_list]


    global class_var
    class_var1=class_var(imputed_list,classifier)
    class_var1=class_var1.tolist()
    #print(class_var)

    

    X_imputed=X_imputed.tolist()


    with open('otherfeatures1.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in X_imputed:
            writer.writerow(row)
            
    data=[list(t) for t in zip(class_var1,imputation_var)]
    with open('2.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

    #print(data)
    f1 = pd.read_csv('otherfeatures1.csv')
    f2 = pd.read_csv('2.csv')
    file = [f1,f2]
    test = pd.concat(file,axis=1)
    test.to_csv("testmodel" + ".csv", index=0, sep=',')
    testmodel = pd.read_csv('testmodel.csv', delimiter=',', header=None)
    return testmodel

def importance(trainmodel,testmodel):
    X_train = trainmodel[:,:-1]
    Y_train1 = trainmodel[:,-1]
    Y_train=[]
    for n in Y_train1:
      Y_train.append(float(n));
    X_test = testmodel

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, Y_train)

    preimp=model.predict(X_test)


    importance=preimp.tolist()
    return importance


def produce_NA(X, p_miss, mecha, opt=None, p_obs=None, q=None):
     
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    missdata=X_nas
    return missdata

def getbias(x,classifier):
    Male = np.array(list(filter(lambda x: x[1] == 1, x)))
    Female = np.array(list(filter(lambda x: x[1] == 0, x)))
    X_testM = Male[:,1:]
    Y_testM = Male[:,0]
    """if you use CNN to do the classification, delete the comment out mark#"""
    #X_testM = pad_sequences(X_testM, padding='post', maxlen=100)
    
    
    """if you use CNN to do the classification, delete the comment out mark#"""
    Y_predM = classifier.predict(X_testM)
    #Y_predM = classifier.predict(X_testM[:,1:])
    
    """if you use CNN to do the classification, delete the comment out mark#"""
    #threshold = 0.5
    #Y_predM = np.where(Y_predM > threshold, 1, 0)

    
    cnf_matrixM = confusion_matrix(Y_testM, Y_predM)
    X_testF = Female[:,1:]
    Y_testF = Female[:,0]
    
    
    """if you use CNN to do the classification, delete the comment out mark#"""
    #X_testF = pad_sequences(X_testF, padding='post', maxlen=100)
    
    """A part of Fair preprocessing, remove the protected attribute when train the model """
    Y_predF = classifier.predict(X_testF)
    #Y_predF = classifier.predict(X_testF[:,1:])

    
    """if you use CNN to do the classification, delete the comment out mark#"""
    #Y_predF = np.where(Y_predF > threshold, 1, 0)
    
    cnf_matrixF = confusion_matrix(Y_testF, Y_predF)


    accurancy=(cnf_matrixM[1][1]+cnf_matrixM[0][0]+cnf_matrixF[1][1]+cnf_matrixF[0][0])/(sum(sum(cnf_matrixF))+sum(sum(cnf_matrixM)))
    bias=abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))
    return accurancy,bias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF


def multiple_imputation(X_missing, m_imputations, seed):
    #mice=KNNImputer(n_neighbors=5,metric='nan_euclidean')
    mice = IterativeImputer(sample_posterior=True, random_state = seed)
    mice.fit(X_missing)
    imputed_list = [mice.transform(X_missing) for i in range(m_imputations)]
        
    X_imputed = np.mean(imputed_list, axis=0)

    assert (X_imputed != imputed_list[0]).any()
    
    imputation_var=imputation_uncertainty(imputed_list)

    return X_imputed,imputation_var,imputed_list


def knn_imputation(X_missing,m_imputations):
    imputer=KNNImputer(n_neighbors=5)
    df_miss = pd.DataFrame(X_missing)

    missdata=df_miss.sample(frac=0.7)
    missdata=missdata.values
    imputer.fit(missdata)
    imputed_list = [imputer.transform(X_missing) for i in range(m_imputations)]
    X_imputed = np.mean(imputed_list, axis=0)
    #assert (X_imputed != imputed_list[0]).any()
    imputation_var=imputation_uncertainty(imputed_list)
    return X_imputed, imputation_var, imputed_list


