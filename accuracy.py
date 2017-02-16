import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.genfromtxt(fname="output.csv",delimiter=",",skip_header=0)
y = np.genfromtxt(fname="../Data_Final/Int_OneHotEncoding.csv",delimiter=",",skip_header=1, usecols=(160))
error =0
for i in range(len(X)):
	
	if X[i]!=y[i]:
		error = error+1
print error
print len(X)
percent =  1 - float(error)/len(X)
print percent*100

neigh = KNeighborsClassifier(n_neighbors=5)