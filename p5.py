from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from data import loadData
from sklearn.metrics import precision_recall_curve,auc, roc_curve
from collections import Counter


data = loadData()

df_x = data.loc[:,'b3356':'b4703']
df_y = data.loc[:,'GrowthRate']

Lasso_model = linear_model.Lasso(alpha = 1e-5,tol = 0.001)
Lasso_model.fit(df_x,df_y)

coef = Lasso_model.coef_
print('Number of features:',len(coef != 0))
df_x = df_x.loc[:,coef != 0]

y1 = data.loc[:,'Medium']
y2 = data.loc[:,'Stress']

y_comp = y1 + ',' + y2


most_common = Counter(y_comp).most_common(1)
num_common = most_common[0][1]
print('The most common class:',most_common)
print('Baseline Percentage:',num_common/194)


def aucpr(x,y):

    df_x = x
    y_comp = y

    dummies_y1 = pd.get_dummies(y_comp)

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


    y_real = np.concatenate(y_real)
    y_pred = np.concatenate(y_pred)

    precision, recall, _ = precision_recall_curve(np.array(y_real).ravel(), y_pred.ravel())

    print('AUC For PR:',auc(recall,precision))

def aucroc(x,y):

    df_x = x
    y_comp = y

    dummies_y1 = pd.get_dummies(y_comp)

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


    y_real = np.concatenate(y_real)
    y_pred = np.concatenate(y_pred)
    fp_rate, tp_rate, _ = roc_curve(np.array(y_real).ravel(), y_pred.ravel())

    print('AUC For ROC:',auc(fp_rate,tp_rate))


#
# aucpr(df_x,y_comp)
# aucroc(df_x,y_comp)
#

