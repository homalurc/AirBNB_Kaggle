import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.genfromtxt(fname="output1.csv",delimiter=",",skip_header=0)
y = np.genfromtxt(fname="../Nishi Shah/trained2.csv",delimiter=",",skip_header=1, usecols=(13))
error =0
for i in range(len(X)):
	
	if X[i]!=y[i]:
		error = error+1
print error
print len(X)
percent =  1 - float(error)/len(X)
print percent*100

neigh = KNeighborsClassifier(n_neighbors=5)