from p2 import p2
import numpy as np
from data import loadData

data = loadData()

df_x = data.loc[:,'b3356':'b4703']
num_col = len(df_x.columns)
mean_val = np.array(np.mean(df_x,axis=0))
reshaped = mean_val.reshape(1,num_col)

result = p2(reshaped)

n_iterations = 50

stdv = np.std(result)
mean_val = np.mean(result)

cf_upper = mean_val + 2*stdv
cf_lower = mean_val - 2*stdv

cf_interval = [cf_lower,cf_upper]

print('Confidence Interval:',cf_interval)


