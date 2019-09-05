from data import loadData
from sklearn import linear_model
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc

data = loadData()

df_x = data.loc[:,'b3356':'b4703']
df_y = data.loc[:,'GrowthRate']

Lasso_model = linear_model.Lasso(alpha = 1e-5,tol = 0.001)
Lasso_model.fit(df_x,df_y)

coef = Lasso_model.coef_
print('Features with nonzero weights are used:')
print('Number of features Used For all Models:',len(coef != 0))
print()

df_x = df_x.loc[:,coef != 0] #features that with nonzero weight
y1 = data.loc[:,'Strain']
y2 = data.loc[:,'Medium']
y3 = data.loc[:,'Stress']
y4 = data.loc[:,'GenePerturbed']

def p4PR(x,y):


    df_x = x
    df_y1 = y

    dummies_y1 = pd.get_dummies(df_y1)

    num_fold = 10
    k_fold = KFold(n_splits=num_fold, shuffle=True, random_state=12345)
    model1 = OneVsRestClassifier(SVC(probability=True))

    y_real = []
    y_pred = []

    for i, (trainIndex, testIndex) in enumerate(k_fold.split(df_x)):
        xtrain,xtest = df_x.iloc[trainIndex,:],df_x.iloc[testIndex,:]
        ytrain,ytest = dummies_y1.iloc[trainIndex,:],dummies_y1.iloc[testIndex,:]

        model1.fit(xtrain,ytrain)
        y_predict = model1.predict_proba(xtest)

        y_pred.append(y_predict)
        y_real.append(ytest) #appending ytest data as real data set

        ##For PR Curve
        precision, recall, _ = precision_recall_curve(np.array(ytest).ravel(), y_predict.ravel())
        pr_auc = auc(recall,precision)
        plt.plot(recall,precision)
        print('Precision Recall Curve:')
        print('Fold: {} - AUC: {}'.format(i+1, pr_auc))




    y_real = np.concatenate(y_real)
    y_pred = np.concatenate(y_pred)
    precision, recall, _ = precision_recall_curve(y_real.ravel(), y_pred.ravel())
    auc_overall = auc(recall,precision)
    print('Overall AUC for PR:',auc_overall)
    print()
    plt.plot(recall,precision,color='black')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(['Fold1','Fold2','Fold3','Fold4','Fold5','Fold6','Fold7','Fold8','Fold9','Fold10','Overall'],loc='lower center')
    plt.show()




def p4ROC(x,y):

    df_x = x
    df_y1 = y


    dummies_y1 = pd.get_dummies(df_y1)

    num_fold = 10
    k_fold = KFold(n_splits=num_fold, shuffle=True, random_state=12345)
    model1 = OneVsRestClassifier(SVC(probability=True))

    y_real = []
    y_pred = []
    for i, (trainIndex, testIndex) in enumerate(k_fold.split(df_x)):
        xtrain,xtest = df_x.iloc[trainIndex,:],df_x.iloc[testIndex,:]
        ytrain,ytest = dummies_y1.iloc[trainIndex,:],dummies_y1.iloc[testIndex,:]

        model1.fit(xtrain,ytrain)
        y_predict = model1.predict_proba(xtest)

        y_pred.append(y_predict)
        y_real.append(ytest) #appending ytest data as real data set

      #For ROC Curve

        fp_rate, tp_rate, _ = roc_curve(np.array(ytest).ravel(), y_predict.ravel())
        roc_auc = auc(fp_rate, tp_rate)
        plt.plot(fp_rate, tp_rate)
        print('ROC Curve')
        print('Fold: {} - AUC: {}'.format(i+1, roc_auc))

    y_real = np.concatenate(y_real)
    y_pred = np.concatenate(y_pred)
    fp_rate, tp_rate, _ = roc_curve(np.array(y_real).ravel(), y_pred.ravel())
    auc_overall = auc(fp_rate,tp_rate)
    print('Overall AUC for ROC:',auc_overall)
    print()
    plt.plot(fp_rate,tp_rate,color='black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(['Fold1','Fold2','Fold3','Fold4','Fold5','Fold6','Fold7','Fold8','Fold9','Fold10','Overall'],loc='lower center')
    plt.show()

#
# p4PR(df_x,y1)
# p4ROC(df_x,y1)
#
# p4PR(df_x,y2)
# p4ROC(df_x,y2)
#
# p4PR(df_x,y3)
# p4ROC(df_x,y3)
#
# p4PR(df_x,y4)
# p4ROC(df_x,y4)
#




