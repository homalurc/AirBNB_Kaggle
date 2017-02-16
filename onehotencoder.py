import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = np.genfromtxt(fname="../data_final_2/autoencoder_output_2lakhs.csv",delimiter=",",skip_header=1, usecols=(0,1,4,5,6,7,8,9,10,11,12))
data1 = np.genfromtxt(fname="../data_final_2/autoencoder_output_2lakhs.csv",delimiter=",",skip_header=1, usecols=(2,3,13))
enc =  OneHotEncoder()

print data

X = enc.fit_transform(data).toarray()
print enc.n_values_
#print X[1][1:35]
#print X[0][1:35]
print len(X)
#print len(data1[1])
print len(X[1][:])
#print np.r_[X[1][1:35],data1[1][0]]
print X.shape


Y = np.empty((213320,184))

for i in range(1,len(X)):
	Y[i] = np.r_[X[i,:67],data1[i,0:1],X[i][67:],data1[i,1]]


np.savetxt("../data_final_2/ohetrain_50.csv", Y, delimiter=",")
