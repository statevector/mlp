import numpy as np
import pandas as pd

from scipy import optimize
from scipy.optimize import minimize, rosen, rosen_der

class Logistic_Regression():

	def __init__(self):
		# set model hyper parameters: 
		# initialize weights with gaussian random numbers, include bias term
		self.inputLayerSize = 3 
		self.W = np.random.randn(self.inputLayerSize + 1) # 1D array

	def parameters(self):
		print(" input size: {}\n output size: {}\n weights: {}\n".format(
			self.inputLayerSize, self.W))

	# propagate input through the network
	# vector in scalar out (or feature matrix in vector out)
	def forward(self, X):
		ones = np.ones([X.shape[0], 1])
		X = np.append(ones, X, axis = 1)
		#print('X: {} \nX.shape {}:'.format(X, X.shape))
		#print('W: {} \nW.shape {}:'.format(self.W, self.W.shape))
		#print('WXT', np.matmul(self.W, X.T))
		return self.sigmoid(np.matmul(self.W, X.T)) # I think it goes like [1,k] x [n,k].T ~ [1,n]

	# compute sigmoid function
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	# compute derivative of sigmoid function
	def sigmoidPrime(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	# mean squared error loss
	#def costFunction(self, X, y):
	#	yhat = self.forward(X)
	#   N = X.shape[0]
	#	return 1/(2*N) * np.sum((yhat - y) ** 2)

	# binary cross entropy loss
	# return single scalar value from vector of targets and predictions
	def costFunction(self, X, y):
		yhat = self.forward(X)
		N = X.shape[0]
		#print('y: ', y)
		#print('yT: ', y.T)
		#print('yhat: ', yhat)
		#print('1-yT: ', 1-y.T)
		#print('1-yhat: ', 1-yhat)
		return -1/N * (np.matmul(y.T, np.log(yhat)) + np.matmul(1-y.T, np.log(1-yhat)))

	# compute gradient of cost function wrt W and b parameters (i.e. 4 params here)
	# will have same number of elements as columns k in X
	def costFunctionGradient(self, X, y):
		yhat = self.forward(X)
		ones = np.ones([X.shape[0], 1])
		X = np.append(ones, X, axis = 1)
		N = X.shape[0]
		return 1/N * np.matmul((yhat - y).T, X) # [n,1].T x [n,k] ~ [1,k]
		# negative here?

	def getParameters(self):
		return self.W

	def setParameters(self, params):
		#self.W = np.reshape(params[0:self.inputLayerSize+1])
		self.W = params

	#def setParameters(self, params):
	#	pass
		#W_start = 0
		#W_end = self.hiddenLayerSize * self.inputLayerSize
		#self.W = np.reshape(params[W1_start:W1_end])

	def _cost(self, x, *args):
		X, y = args[0], args[1]
		self.setParameters(x)
		cost = self.costFunction(X, y)
		print('cost', cost)
		#print('W', self.W)
		return cost

	def _grad(self, x, *args):
		X, y = args[0], args[1]
		self.setParameters(x)
		grad = self.costFunctionGradient(X, y)
		#print('grad', grad)
		return grad

	def train(self, X, y):

		# initial pamameter guess. Array of real elements of size (n,)
		x0 = self.W
		print('x0:', x0, x0.shape)

		# If jac is a Boolean and is True, 
		# fun is assumed to return the gradient along with the objective function
		res = optimize.minimize(fun = self._cost, 
			x0 = x0,
			args = (X, y),
			jac = self._grad, 
			method = 'BFGS', 
			options = {'maxiter' : 1000, 'disp' : True})
				
		# update parameters with the optimized result
		#print('parameters: ', res.x)
		self.setParameters(res.x)

		return res






if __name__ == '__main__':

	#data = pd.read_csv('data/HTRU_2.csv', header = None)
	#x = data[[0,1,2,3,4,5,6,7]].values # normalize this...
	#y = data[[8]].values

	# use iris without 3rd class! make binary!

	lr = Logistic_Regression()
	#lr.parameters()

	# feature matrix X
	x = np.array([[1.0, 2.0, 3.0], 
				  [1.1, 2.2, 3.3],
				  [0.9, 1.9, 2.9],
				  [-1, -2, -3]])
	#x = np.array([1, 2, 3])
	#x.shape = (3, 3)
	print('X: {} \nX.shape {}:'.format(x, x.shape))

	# predict X
	yhat = lr.forward(x)
	print('yhat:', yhat)

	# target labels y
	y = np.array([1, 1, 1, 0])
	#y.shape = (2, 1)
	print('y: {} \ny.shape {}:'.format(y, y.shape))

	# compute cost function
	J = lr.costFunction(x, y)
	print('cost: ', J)

	dJ = lr.costFunctionGradient(x, y)
	print('grad: ', dJ)

	result = lr.train(x, y)
	print('result: ', result)






