
from sklearn.decomposition import PCA
from data import loadData
import pandas as pd
from p4 import p4PR,p4ROC

data = loadData()

x_set = data.loc[:,'b3356':'b4703']

pca = PCA(n_components=3)
pca.fit(x_set)
new_x = pd.DataFrame(pca.transform(x_set))

y1 = data.loc[:,'Strain']
y2 = data.loc[:,'Medium']
y3 = data.loc[:,'Stress']
y4 = data.loc[:,'GenePerturbed']


# p4PR(new_x,y1)
# p4ROC(new_x,y1)
#
# p4PR(new_x,y2)
# p4ROC(new_x,y2)
#
# p4PR(new_x,y3)
# p4ROC(new_x,y3)
#
# p4PR(new_x,y4)
# p4ROC(new_x,y4)
#
#
