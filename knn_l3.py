import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.genfromtxt(fname="../data_final_2/autoencoder_output_2lakhs.csv",delimiter=",",skip_header=0, usecols=range(0,14))
#y = np.genfromtxt(fname="../data_final_2/autoencoder_output_2lakhs.csv",delimiter=",",skip_header=0, usecols=(13))
np.random.shuffle(X) 

y = X[:,13]
X = X[:,:12]

for frac in (0.8,0.9):
	print str(frac*100) +"% of training data:"
	
	neigh = KNeighborsClassifier(n_neighbors=6, p=2, algorithm='kd_tree')
	l = frac*len(X)
	neigh.fit(X[:l][:], y[:l][:])
	P = neigh.predict(X[l:][:])
	error = 0;
	for i in range(len(P)):
		if P[i] != y[l+i]:
			error=error+1
	print "k ="+str(2)
	print error
	print len(P)
	acc= float(error)/len(P);
	print acc*100
np.savetxt('output.csv',P,delimiter=',')

