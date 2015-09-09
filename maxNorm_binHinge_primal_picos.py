import cvxopt as cvx
import picos as pic
import time
import scipy.io
import numpy as np
mat = scipy.io.loadmat('y_100_30_204_bin.mat')['y']
matlab = scipy.io.loadmat('y_100_30_204_bin_COMPLETED_x.mat')['x']

def learnMMMF(y, c):
	n,m = y.shape
	n_obs = np.count_nonzero(y)
	obs = np.nonzero(y)

	prob = pic.Problem()    #create a Problem instance

	# vars
	A = prob.add_variable('A', (n,n))
	B = prob.add_variable('B', (m,m))
	X = prob.add_variable('X', (n,m))

	t = prob.add_variable('t', 1)
	e = []
	for i in range(n_obs):
		e.append(prob.add_variable('e[{0}]'.format(i)))

	# constraints
	prob.add_constraint(((A & X)//(X.T & B)) >> 0)
	prob.add_constraint(pic.diag_vect(A) <= t)
	prob.add_constraint(pic.diag_vect(B) <= t)
	for i in e:
		prob.add_constraint(i >= 0)

	for x in enumerate(zip(obs[0], obs[1])):	# each observation
		ind = x[0]
		i, a = int(x[1][0]), int(x[1][1])

		prob.add_constraint((y[i,a] * X[i,a]) >= (1 - e[ind]))

	# objective
	prob.set_objective('min', t + c * pic.sum(e))

	# solve
	time_start = time.clock()
	prob.solve(verbose=1, solver='sdpa', solve_via_dual = False, noduals=True)
	time_end = time.clock()

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