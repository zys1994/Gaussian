import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
import math

def calcGaussian(x,m,u): #m-means u-方差
    v=math.exp(-pow((x-m),2)/(2*pow(u,2)))/(math.sqrt(2*3.1415)*u)
    return v
def calcLogGaussian(x,m,u):
    temp=calcGaussian(x,m,u)
    if temp>10:
        temp=math.log(temp)
    return temp

data = pd.read_csv('Twenty_2.csv')
prob = np.array(data['1'])
data1=np.reshape(prob,[prob.__len__(),1])


fig, ax = plt.subplots(2,2) #
n, bins,patches = ax[0,0].hist(data1,np.arange(0.5,1,0.001))

gmm = GMM(n_components=2).fit(data1)
labels = gmm.predict(data1)
m1=gmm.means_[0,0]
m2=gmm.means_[1,0]
u1=gmm.covars_[0,0]
u2=gmm.covars_[1,0]
#u1=gmm.

x1= [i/1000.0 for i in range(500, 1000)]
y1=np.zeros(x1.__len__())
y2=np.zeros(x1.__len__())
for i in range(x1.__len__()):
    y1[i]=calcLogGaussian(x1[i],m1,u1)
    y2[i] = calcLogGaussian(x1[i],m2,u2)

ax[0,1].plot(x1,y2,color='green')
ax[0,1].plot(x1,y1,color='blue',linestyle='--')
ax[1,0].plot(x1,y1)
ax[1,1].plot(x1,y2)



plt.show()
N=1

'''
n_o=n.copy()
n=np.reshape(n,[n.__len__(),1])
n[n==0]=1

n_o[n_o==0]=1
n_o=stats.boxcox(n_o)[0]
n_o=np.reshape(n_o,[n_o.__len__(),1])

n=np.log(n)

bins=np.reshape(bins,[bins.__len__(),1])
bins=np.delete(bins,-1,axis=0)

ax[0,1].plot(bins,n)
'''