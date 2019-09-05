from data import loadData
import numpy as np
from sklearn import linear_model
from sklearn.utils import resample

def p2(x):
    data = loadData()
    xtest = x
    #data_left = data.loc[~data.index.isin(data_train.index)]
    num_iter = 50
    predictions = []

    for i in range(num_iter):

        data_train = resample(data, replace=True, n_samples=round(0.7*194),random_state=i)

        ytrain = data_train.loc[:,'GrowthRate']
        xtrain = data_train.loc[:,'b3356':'b4703']

        Lasso_model = linear_model.Lasso(alpha = 1e-5,tol = 0.001)
        Lasso_model.fit(xtrain,ytrain)

        predict = np.ndarray.tolist(Lasso_model.predict(xtest))
        predictions.append(predict)

    return(predictions)


