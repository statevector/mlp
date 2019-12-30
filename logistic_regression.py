import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

class LogisticRegression():

	def __init__(self, X, y):
		# set model hyper parameters: 
		# initialize weights with gaussian random numbers, include bias term with +1
		self.inputLayerSize = 3
		self.W = np.random.randn(self.inputLayerSize + 1) # 1D array
		#self.W = np.array([0.09302552, 1.29829595, 2.29181623, 2.43873838]) # min
		#self.W = np.array([-0.43539474,  1.31641954, 2.32551838,  2.48801908]) # another min
		#self.W = np.array([1, -1, -1, 1]) # test
		self.orig = self.W
		# empty list to store callback costs
		self.J = []

	def resetWeights(self):
		#print('self.W:', self.W)
		#print('self.orig', self.orig)
		self.W = self.orig

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
		return -1/N * (np.matmul(y.T, np.log(yhat)) + np.matmul(1-y.T, np.log(1-yhat)))

	# compute gradient of cost function wrt W and b parameters
	# will have same number of elements as columns k in X
	def costFunctionGradient(self, X, y):
		yhat = self.forward(X)
		ones = np.ones([X.shape[0], 1])
		X = np.append(ones, X, axis = 1)
		N = X.shape[0]
		return 1/N * np.matmul((yhat - y).T, X) # [n,1].T x [n,k] ~ [1,k]

	def objectiveFunction(self, x, *args):
		X, y = args[0], args[1]
		self.W = x # update parameters
		cost = self.costFunction(X, y)
		grad = self.costFunctionGradient(X, y)
		#print(cost) # easy callback
		return cost, grad

	def callbackFunction(self, x):
		#print(self.costFunction(self.X, self.y))
		self.J.append(self.costFunction(self.X, self.y))
		
	def train(self, X, y):

		# internal variables for the callback function
		self.X = X
		self.y = y
		
		# starting values for parameters
		x0 = self.W
		print('x0:', x0, x0.shape)

		# If 'jac' is set to 'True', then fun is assumed to return
		# the objective function along with the gradient
		res = optimize.minimize(fun = self.objectiveFunction, 
			x0 = x0,
			args = (X, y), # extra args passed to the objective function
			jac = True, 
			method = 'BFGS', 
			callback = self.callbackFunction,
			options = {'maxiter' : 1000, 'disp' : True})
		
		# update parameters with the final optimized result
		self.W = res.x

		return res

	def gradient_descent(self, X, y):
	
		rate = 0.1
		print('Rate {}:'.format(rate))

		n = 0
		print('Epoch {}:'.format(n))
		print("loss: {}".format(self.costFunction(X, y)))

		# Perform a gradient descent algorithm
		#while self.costFunction(X, y) > 0.01 and n < 10000:
		while n < 1000000:
			n = n + 1
			loss = self.costFunction(X, y)
			W_new = self.W - rate * self.costFunctionGradient(X, y)
			self.W = W_new
			#print(self.W)
			#print(n, self.costFunction(X, y))
			#if (np.abs(loss - self.costFunction(X, y))/loss < 1e-6) and (loss < 2e-6):
			if (loss < 2e-6):
				print(loss, self.costFunction(X, y))
				break

		print("Epoch {}:".format(n))
		print("loss: {}".format(self.costFunction(X, y)))
		print("W: {}".format(self.W))


if __name__ == '__main__':

	#data = pd.read_csv('data/HTRU_2.csv', header = None)
	#x = data[[0,1,2,3,4,5,6,7]].values # normalize this...
	#y = data[[8]].values

	# feature matrix X
	x = np.array([[1.0, 2.0, 3.0], 
				  [1.1, 2.2, 3.3],
				  [0.9, 1.9, 2.9],
				  [-1, -2, -3]])
	#x = np.array([1, 2, 3])
	#x.shape = (3, 3)
	print('X: {} \nX.shape {}:'.format(x, x.shape))

	# target labels y
	y = np.array([1, 1, 1, 0])
	#y.shape = (2, 1)
	print('y: {} \ny.shape {}:'.format(y, y.shape))

	lr = LogisticRegression(x, y)

	# predict X
	yhat = lr.forward(x)
	print('yhat:', yhat)

	# compute cost function
	J = lr.costFunction(x, y)
	print('cost: ', J)

	dJ = lr.costFunctionGradient(x, y)
	print('grad: ', dJ)

	result = lr.train(x, y)
	print('result: ', result)

	# check if never before seen test case works
	#x_test = np.array([0.8, 1.8, 2.8]).reshape(1,3)
	#print('X: {} \nX.shape {}:'.format(x_test, x_test.shape))
	#test = lr.forward(x_test)
	#print(test)

	# alternative training algorithm
	#lr.resetWeights()
	#lr.gradient_descent(x, y)

	# result of callback function... annoyingly missing initial cost function eval...
	print('costs: ', lr.J)

	# plot the results
	xvar = np.arange(0, len(lr.J), 1)
	yvar = lr.J
	
	plt.plot(xvar, yvar, 'r--')
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.title('Cost vs. BFGS Iterations')
	plt.xticks(xvar)
	plt.show()




