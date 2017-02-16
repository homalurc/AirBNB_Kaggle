import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.genfromtxt(fname="trained2.csv",delimiter=",",skip_header=1, usecols=range(1,14))
#y = np.genfromtxt(fname="../data_final_2/autoencoder_output_2lakhs.csv",delimiter=",",skip_header=0, usecols=(13))
np.random.shuffle(X) 

y = X[:,12]
X = X[:,:11]
frac =0.8;
print str(frac*100) +"% of training data:"
print "Ball Tree"
for k in range(1,13,2):
	neigh = KNeighborsClassifier(n_neighbors=k,algorithm='ball_tree')
	l = frac*len(X)
	neigh.fit(X[:l][:], y[:l][:])
	P = neigh.predict(X[l:][:])
	error = 0;
	for i in range(len(P)):
		if P[i] != y[l+i]:
			error=error+1
	print "k ="+str(k)
	print error
	print len(P)
	acc= float(error)/len(P);
	print acc*100

#np.savetxt('output.csv',P,delimiter=',')

