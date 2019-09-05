from data import loadData
from sklearn import linear_model
from sklearn.model_selection import cross_validate
import numpy as np

def p1_Ridge():

    data = loadData()

    df_x = np.array(data.loc[:,'b3356':'b4703'])
    df_y = np.array(data.loc[:,'GrowthRate'])

    coef = []
    errors = []
    alphas = [0.0001,0.001,0.1, 1, 10, 100,1000,100000]
    for i in range(len(alphas)):
        ridge_model = linear_model.Ridge(alpha = alphas[i])
        ridge_model.fit(df_x,df_y)

        scores = cross_validate(ridge_model, df_x,df_y, cv=10,scoring='neg_mean_squared_error')
        errors.append(np.mean(-scores['test_score']))
        coef.append(ridge_model.coef_)

    errors = np.array(errors)
    minIndex = np.argmin(errors)

    model_ridge = linear_model.Ridge(alpha = alphas[minIndex])
    model_ridge.fit(df_x,df_y)
    feature_nozeros = np.count_nonzero(model_ridge.coef_)



    print('Method:Ridge Regression')
    print('Number of features that have non-zero coefficients:',feature_nozeros)
    print('10-fold Cross Validation MSE:',errors)
    print('Optimal Lambda Value:',alphas[minIndex])

#
# p1_Ridge()


def p1_Lasso():

    data = loadData()

    df_x = np.array(data.loc[:,'b3356':'b4703'])
    df_y = np.array(data.loc[:,'GrowthRate'])

    coef = []
    errors = []

    alphas = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for i in range(len(alphas)):
        Lasso_model = linear_model.Lasso(alpha = alphas[i],tol = 0.001)
        Lasso_model.fit(df_x,df_y)

        scores = cross_validate(Lasso_model, df_x,df_y, cv=10,scoring='neg_mean_squared_error')
        errors.append(np.mean(-scores['test_score']))
        coef.append(Lasso_model.coef_)

    errors = np.array(errors)
    minIndex = np.argmin(errors)

    model_lasso = linear_model.Lasso(alpha = alphas[minIndex])
    model_lasso.fit(df_x,df_y)
    feature_zeros = np.count_nonzero(model_lasso.coef_)


    print('Method:Lasso Regression')
    print('Number of features that have Non-zero coefficients:',feature_zeros)
    print('10-fold Cross Validation MSE:',errors)
    print('Optimal Lambda Value:',alphas[minIndex])
#
# p1_Lasso()


