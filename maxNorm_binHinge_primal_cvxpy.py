"""
"Maximum-Margin Matrix Factorization (Srebro, Rennie, Jaakkola)
-> Original Matlab-code: http://ttic.uchicago.edu/~nati/mmmf/code.html
"""

from cvxpy import *
import numpy as np
import random
import time
random.seed(1)
np,random.seed(1)

# generate test data
#X,Y = 100, 30
#Z = 500
#data = np.zeros((X,Y), dtype=int)
#for i in range(Z):
#	finished = False
#	while not finished:
#		x = random.randint(0,X-1)
#		y = random.randint(0,Y-1)
#		if data[x,y] == 0:
#			val = -1
#			if random.randint(0,1):
#				val = 1
#			data[x,y] = val
#			finished = True
#print(data)

import scipy.io
mat = scipy.io.loadmat('y_100_30_204_bin.mat')['y']
matlab = scipy.io.loadmat('y_100_30_204_bin_COMPLETED_x.mat')['x']

def learnMMMF(y, c):
	n,m = y.shape
	n_obs = np.count_nonzero(y)
	obs = np.nonzero(y)

	A = Variable(n,n)
	B = Variable(m,m)
	X = Variable(n,m)

	t = Variable()
	e = Variable(n_obs)

	objective = Minimize(t + c * sum_entries(e))
	
	constraints = []
	constraints.append(bmat([[A,X],[X.T,B]]) >> 0)
	constraints.append(diag(A) <= t)
	constraints.append(diag(B) <= t)
	constraints.append(e >= 0)

	for x in enumerate(zip(obs[0], obs[1])):	# each observation
		ind = x[0]
		i, a = x[1][0], x[1][1]
		constraints.append(y[i,a] * X[i,a] >= 1 - e[ind])

	prob = Problem(objective, constraints)

	time_start = time.clock()
	result = prob.solve(kktsolver=ROBUST_KKTSOLVER, verbose=True)
	time_end = time.clock()

	print('used time for SDP-solving: ' + str(time_end - time_start))
	
	ORIG = mat
	MATLAB = matlab
	PYTHON = X.value

	print('orig')
	print(ORIG)
	print('matlab')
	print(MATLAB)
	print('python')
	print(PYTHON)
	print('diff')
	print(MATLAB - PYTHON)

learnMMMF(mat, 1.0)
