import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.genfromtxt(fname="../Nishi Shah/trained2.csv",delimiter=",",skip_header=1, usecols=range(1,12))
y = np.genfromtxt(fname="../Nishi Shah/trained2.csv",delimiter=",",skip_header=1, usecols=(13))

for k in range(1,10):
	neigh = KNeighborsClassifier(n_neighbors=k)
	l = 0.9*len(X)
	neigh.fit(X[:l][:], y[:l][:])
	P = neigh.predict(X[l:][:])
	error = 0;
	for i in range(len(P)):
		if P[i] != y[l+i]:
			error=error+1
	print error
	print len(P)
	acc= 1 - float(error)/len(P);
	print acc*100
np.savetxt('output1.csv',P,delimiter=',')

