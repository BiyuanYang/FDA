from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
from statistics import mean



data_df = pd.read_csv('./Dataset/compas_clean.csv', delimiter=',', header=None)
data = data_df.values

#split dataset

c_train,c_test = model_selection.train_test_split(data,test_size = 0.3)
traindata=c_train



new_train,new_test = model_selection.train_test_split(traindata,test_size = 0.5)

X_train = new_train[:,1:]
Y_train = new_train[:,0]
#classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier = LogisticRegression(random_state=0)
#classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train, Y_train)





X_full=new_test


"""full dastaset, get the classlabel without missing value"""
Male = np.array(list(filter(lambda x: x[1] == 1, X_full)))
Female = np.array(list(filter(lambda x: x[1] == 0, X_full)))
X_testM = Male[:,1:]
Y_testM = Male[:,0]
Y_predM1 = classifier.predict(X_testM)
X_testF = Female[:,1:]
Y_testF = Female[:,0]
Y_predF1 = classifier.predict(X_testF)


fulldata=np.vstack((Male, Female))


onlyimpb,onlyimpa=[],[]
mostfairb,mostfaira=[],[]
mostacca,mostaccb=[],[]
randoma,randomb=[],[]
comba,combb=[],[]
comba1,combb1=[],[]
comba2,combb2=[],[]
onlyvara,onlyvarb=[],[]
protecta,protectb=[],[]


"""Run it ten times and average the results ten times"""
for p in range(10):


    """Manufacturing missing data"""
    X_protect = new_test[:,:2]
    X_miss = new_test[:,2:]


    X_miss=produce_NA(X_miss, 0.4, mecha="MCAR", opt=None, p_obs=None, q=None)
    x=[]
    for i in range(len(X_miss)):
        x.append(np.insert(X_miss[i].tolist(),0,X_protect[i]))
    X_miss = np.array(x)





    """Imputation, can change the imputation methods"""
    m_imputations=5#Times of imputation
    X_imputed,imputation_var,imputed_list1=knn_imputation(X_miss, m_imputations)
    #X_imputed,imputation_var,imputed_list1=multiple_imputation(X_miss,m_imputations, 1)





    Male = np.array(list(filter(lambda x: x[1] == 1, X_imputed)))
    Female = np.array(list(filter(lambda x: x[1] == 0, X_imputed)))
    X_testM = Male[:,1:]
    Y_testM = Male[:,0]
    Y_predM2 = classifier.predict(X_testM)
    cnf_matrixM = confusion_matrix(Y_testM, Y_predM2)
    X_testF = Female[:,1:]
    Y_testF = Female[:,0]
    Y_predF2 = classifier.predict(X_testF)
    cnf_matrixF = confusion_matrix(Y_testF, Y_predF2)




    accurancy=(cnf_matrixM[1][1]+cnf_matrixM[0][0]+cnf_matrixF[1][1]+cnf_matrixF[0][0])/(sum(sum(cnf_matrixF))+sum(sum(cnf_matrixM)))
    bias2=abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))


     


    """Accurancy importance"""


    #Get Missing variance MV and Class variance CV
    global class_var
    class_list=[classifier.predict(i[:,1:]) for i in imputed_list1]
    class_var=np.var(class_list,axis=0,ddof=1)

    imputation_var=imputation_var[:,np.newaxis].T
    class_var=class_var[:,np.newaxis].T




    #Put MV and CV as 2 new features
    for i in range(len(imputation_var)):
        trainmodel=np.append(X_imputed,imputation_var.T,axis=1)
        trainmodel=np.append(trainmodel,class_var.T,axis=1)
        
    importance_acc=[]
    for i in range(len(class_list[0])):
        k=0
        for j in range(len(class_list)):
            if class_list[j][i]==new_test[:,0][i]:
                k+=1
        importance_acc.append(1-k/m_imputations)

    """Train fair importance model"""
    accmodel = LinearRegression()
    accmodel.fit(trainmodel, importance_acc)




    Male_train = np.array(list(filter(lambda x: x[1] == 1, trainmodel)))
    Female_train = np.array(list(filter(lambda x: x[1] == 0, trainmodel)))



    """Fairness importance(Calculate the difference of bias when use imputed values and real values)"""

    importance_fair=[]
    for i in range(len(Y_predM1)):
        if Y_predM2[i]==Y_predM1[i]:
            importance_fair.append(0)
            
        elif Y_testM[i]==1:
            if Y_predM2[i]==1:
                importance_fair.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-(cnf_matrixM[1][1]-1)/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
            else:
                importance_fair.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-(cnf_matrixM[1][1]+1)/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
        else:
            if Y_predM2[i]==0:
                importance_fair.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-(cnf_matrixM[0][1]+1)/(cnf_matrixM[0][1]+cnf_matrixM[0][0]+1))-bias2)
            else:
                importance_fair.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-(cnf_matrixM[0][1]-1)/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)

    for i in range(len(Y_predF1)):
        if Y_predF2[i]==Y_predF1[i]:
            importance_fair.append(0)
            
        elif Y_testF[i]==1:
            if Y_predF2[i]==1:
                importance_fair.append(abs((cnf_matrixF[1][1]-1)/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
            else:
                importance_fair.append(abs((cnf_matrixF[1][1]+1)/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs(cnf_matrixF[0][1]/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
        else:
            if Y_predF2[i]==0:
                importance_fair.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs((cnf_matrixF[0][1]+1)/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)
            else:
                importance_fair.append(abs(cnf_matrixF[1][1]/(cnf_matrixF[1][1]+cnf_matrixF[1][0])-cnf_matrixM[1][1]/(cnf_matrixM[1][1]+cnf_matrixM[1][0]))+abs((cnf_matrixF[0][1]-1)/(cnf_matrixF[0][1]+cnf_matrixF[0][0])-cnf_matrixM[0][1]/(cnf_matrixM[0][1]+cnf_matrixM[0][0]))-bias2)

    aldata=np.vstack((Male_train, Female_train))



    """Train fair importance model"""
    biasmodel = LinearRegression()
    biasmodel.fit(aldata, importance_fair)



    """Testset"""
    accfull,biasfull,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF = getbias(c_test,classifier)
    print("Accurancy full:",accfull)
    print("Bias full:",biasfull)


    X_protect = c_test[:,:2]
    X_miss = c_test[:,2:]
    X_miss=produce_NA(X_miss, 0.4, mecha="MCAR", opt=None, p_obs=None, q=None)
    x=[]
    for i in range(len(X_miss)):
        x.append(np.insert(X_miss[i].tolist(),0,X_protect[i]))
    X_miss = np.array(x)


    """Imputation"""
    m_imputations=5
    X_imputed,imputation_var,imputed_list1=knn_imputation(X_miss, m_imputations)

    
    var_imp=imputation_uncertainty(imputed_list1)



 
    accmiss,biasmiss,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF = getbias(X_imputed,classifier)
    print("Accurancy miss:",accmiss)
    print("Bias miss:",biasmiss)




    #Get variance
    class_list=[classifier.predict(i[:,1:]) for i in imputed_list1]
    class_var=np.var(class_list,axis=0,ddof=1)

    imputation_var=imputation_var[:,np.newaxis].T
    class_var=class_var[:,np.newaxis].T



    #put MV and CV to the features
    for i in range(len(imputation_var)):
        testmodel=np.append(X_imputed,imputation_var.T,axis=1)
        testmodel=np.append(testmodel,class_var.T,axis=1)

    pre_accimp=accmodel.predict(testmodel)

    preimp=biasmodel.predict(testmodel)





    """Part of fair pre-processing (protected group and other proportional sampling)"""
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
    

    

    
    pre_biasimp=preimp.tolist()



    x_axis_data = []
    x_axis_data1 = []
    x_axis_data2 = []
    x_axis_data3 = []
    x_axis_data4 = []
    x_axis_data5 = []
    x_axis_data6 = []
    x_axis_data7 = []
    x_axis_data8 = []
    y_axis_data = []
    y_axis_data1 = []
    y_axis_data2 = []
    y_axis_data3 = []
    y_axis_data4 = []
    y_axis_data5 = []
    y_axis_data6 = []
    y_axis_data7 = []
    y_axis_data8 = []
    
    
    """Equal proportional sampling of protected groups"""

    for i in sorted(range(len(accimp_F)), key = lambda sub: accimp_F[sub])[(int(len(accimp_F)*(1-0.3))):]:
        Female_imputed[i]=Female_full[i]
    for i in sorted(range(len(accimp_M)), key = lambda sub: accimp_M[sub])[(int(len(accimp_M)*(1-0.3))):]:
        Male_imputed[i]=Male_full[i]
        
    fulldata=np.vstack((Male_imputed, Female_imputed))
        
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(fulldata,classifier)
    protecta.append(newacc)
    protectb.append(newbias)
    x_axis_data8.append(newbias)
    y_axis_data8.append(newacc)

        
    plt.scatter(x_axis_data8, y_axis_data8, c='olive',marker='P', alpha=0.1, linewidth=2)#, label='Most fair')
    
    
    
    
    """Only imputation"""
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_imputed,classifier)
        
    onlyimpa.append(newacc)
    onlyimpb.append(newbias)
        
    x_axis_data7.append(newbias)
    y_axis_data7.append(newacc)


        
    plt.scatter(x_axis_data7, y_axis_data7, c='royalblue',marker='h', alpha=0.1, linewidth=2)#, label='Most fair')


    

    """most fair"""
    X_al=X_imputed.copy()
    maximp1 = sorted(range(len(pre_biasimp)), key = lambda sub: pre_biasimp[sub])[:(int(len(pre_biasimp)*0.3))]
    for i in maximp1:
        X_al[i]=c_test[i]
        
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_al,classifier)
        
    mostfaira.append(newacc)
    mostfairb.append(newbias)
        
    x_axis_data.append(newbias)
    y_axis_data.append(newacc)


        
    plt.scatter(x_axis_data, y_axis_data, c='purple',marker='D', alpha=0.1, linewidth=2)#, label='Most fair')


    """only var"""
    X_al=X_imputed.copy()
    maximp1 = sorted(range(len(var_imp)), key = lambda sub: var_imp[sub])[:(int(len(var_imp)*0.3))]
    for i in maximp1:
        X_al[i]=c_test[i]
        
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_al,classifier)
    onlyvara.append(newacc)
    onlyvarb.append(newbias)
    x_axis_data1.append(newbias)
    y_axis_data1.append(newacc)
    plt.scatter(x_axis_data1, y_axis_data1, c='blue',marker='d', alpha=0.1, linewidth=2)#, label='Only varaince')



    """random"""
    X_al=X_imputed.copy()
    randomchoose=random.sample(range(len(preimp)),int(len(preimp)*0.3))
    for i in randomchoose:
        X_al[i]=c_test[i]
        
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_al,classifier)

    randoma.append(newacc)
    randomb.append(newbias)
    x_axis_data2.append(newbias)
    y_axis_data2.append(newacc)


        
    plt.scatter(x_axis_data2, y_axis_data2, c='black', marker='s',alpha=0.1, linewidth=2)#, label='random choose')



    """most acc"""
    X_imputed2=X_imputed.copy()
    maximp2 = sorted(range(len(pre_accimp)), key = lambda sub: pre_accimp[sub])[(int(len(pre_accimp)*(1-0.3))):]
    for i in maximp2:
        X_imputed2[i]=c_test[i]
        
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_imputed2,classifier)
    mostacca.append(newacc)
    mostaccb.append(newbias)
    x_axis_data3.append(newbias)
    y_axis_data3.append(newacc)


        
    plt.scatter(x_axis_data3, y_axis_data3,  c='green', marker='o',alpha=0.1, linewidth=2)#, label='Most acc')





    """comb"""
    X_imputed3=X_imputed.copy()
    maximp3 = sorted(range(len(pre_accimp)), key = lambda sub: pre_accimp[sub])[(int(len(pre_accimp)*(1-0.3*0.5))):]
    for i in maximp3:
        X_imputed3[i]=c_test[i]
    for i in sorted(range(len(pre_biasimp)), key = lambda sub: pre_biasimp[sub])[:(int(len(pre_biasimp)*0.3*0.5))]:
        X_imputed3[i]=c_test[i]
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_imputed3,classifier)
    comba.append(newacc)
    combb.append(newbias)
    x_axis_data4.append(newbias)
    y_axis_data4.append(newacc)

    plt.scatter(x_axis_data4, y_axis_data4,  c='red', marker='*',alpha=0.1, linewidth=2)#, label='combination')




    X_imputed4=X_imputed.copy()
    maximp3 = sorted(range(len(pre_accimp)), key = lambda sub: pre_accimp[sub])[(int(len(pre_accimp)*(1-0.3*0.3))):]
    for i in maximp3:
        X_imputed4[i]=c_test[i]
    for i in sorted(range(len(pre_biasimp)), key = lambda sub: pre_biasimp[sub])[:(int(len(pre_biasimp)*0.3*0.7))]:
        X_imputed4[i]=c_test[i]
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_imputed4,classifier)
    comba1.append(newacc)
    combb1.append(newbias)
    x_axis_data5.append(newbias)
    y_axis_data5.append(newacc)

    plt.scatter(x_axis_data5, y_axis_data5,  c='m',marker='^', alpha=0.1, linewidth=2)#, label='combination')
    
    X_imputed5=X_imputed.copy()
    maximp3 = sorted(range(len(pre_accimp)), key = lambda sub: pre_accimp[sub])[(int(len(pre_accimp)*(1-0.3*0.7))):]
    for i in maximp3:
        X_imputed5[i]=c_test[i]
    for i in sorted(range(len(pre_biasimp)), key = lambda sub: pre_biasimp[sub])[:(int(len(pre_biasimp)*0.3*0.3))]:
        X_imputed5[i]=c_test[i]
    newacc,newbias,Y_predM,Y_predF,Male,Female,Y_testM,Y_testF,cnf_matrixM,cnf_matrixF=getbias(X_imputed5,classifier)
    comba2.append(newacc)
    combb2.append(newbias)
    x_axis_data6.append(newbias)
    y_axis_data6.append(newacc)

    plt.scatter(x_axis_data6, y_axis_data6,  c='orange',marker='v', alpha=0.1, linewidth=2)#, label='combination'











plt.scatter([mean(mostfairb)],[mean(mostfaira)], c='purple', marker='D',alpha=1, linewidth=5, label='FDA_fair')
plt.scatter([mean(mostaccb)],[mean(mostacca)], c='green',  marker='o',alpha=1, linewidth=5, label='FDA_acc')
plt.scatter([mean(onlyvarb)],[mean(onlyvara)], c='blue', marker='d', alpha=1, linewidth=5, label='AVID')
plt.scatter([mean(randomb)],[mean(randoma)], c='black', marker='s', alpha=1, linewidth=5, label='Random choose')
plt.scatter([mean(combb)],[mean(comba)], c='red',  marker='*', alpha=1,linewidth=5, label='Combination1:1')
plt.scatter([mean(combb1)],[mean(comba1)], c='m',marker='^',alpha=1, linewidth=5, label='Combination3:7')
plt.scatter([mean(combb2)],[mean(comba2)], c='orange',marker='v',alpha=1, linewidth=5, label='Combination7:3')
plt.scatter([mean(onlyimpb)],[mean(onlyimpa)], c='royalblue',marker='h',alpha=1, linewidth=5, label='Only imputation')
plt.scatter([mean(protectb)],[mean(protecta)], c='olive',marker='P', alpha=1, linewidth=5, label='Balance FDA_acc')


plt.ylabel("Acc",fontsize=12)
plt.xlabel("Bias",fontsize=12)
plt.legend()
plt.show()



