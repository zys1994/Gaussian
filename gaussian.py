import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.mixture import GMM

data = pd.read_csv('Twenty_2.csv')
prob = np.array(data['1'])
data1=np.reshape(prob,[prob.__len__(),1])
#y_zero=np.zeros([prob.__len__(),1])
#d=np.concatenate((data1,y_zero),axis=1)

fig, ax = plt.subplots(2,2) #
n, bins,patches = ax[0,0].hist(data1,np.arange(0.5,1,0.001))

n_o=n.copy()
n=np.reshape(n,[n.__len__(),1])
n[n==0]=1

n_o[n_o==0]=1
n_o=stats.boxcox(n_o)[0]
n_o=np.reshape(n_o,[n_o.__len__(),1])

n=np.log(n)

bins=np.reshape(bins,[bins.__len__(),1])
bins=np.delete(bins,-1,axis=0)
d=np.concatenate((bins,n),axis=1)

gmm = GMM(n_components=2).fit(d)
labels = gmm.predict(d)
ax[0,1].scatter(d[:,0], d[:,1],c=labels, s=10, cmap='viridis')


d2=np.concatenate((bins,n_o),axis=1)
gmm2 = GMM(n_components=2).fit(d2)
labels2 = gmm2.predict(d2)
ax[1,0].scatter(d2[:,0], d2[:,1],c=labels2, s=10, cmap='viridis')

plt.show()
N=1

'''
fig, ax = plt.subplots(2,2) #
n, bins,patches = ax[0,0].hist(prob,np.arange(0.5,1,0.001))

gmm = GMM(n_components=2).fit(d)
labels = gmm.predict(d)
labels=np.reshape(labels,[d.__len__(),1])
ax[0,1].scatter(data1, y_zero,c=labels, s=40, cmap='viridis');
'''

'''
fig, ax = plt.subplots(2,2)
n, bins,patches = ax[0,1].hist(prob,np.arange(0.5,1,0.001))
plt.show()
'''