import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

#np.random.seed(1234)
np.random.seed(1235)

class LogisticRegression():

	# set model hyper parameters: 
	def __init__(self, nfeatures):
		# initialize weights with gaussian random numbers
		# bias term incliuded with +1
		self.W = np.random.randn(nfeatures + 1) # 1D array
		self.orig = self.W
		# empty list to store callback costs
		self.J = []

	def resetWeights(self):
		self.W = self.orig

	# sigmoid activation function
	def sigmoid(self, z):
		'''
		z: a scalar, vector, or matrix
		'''
		return 1 / (1 + np.exp(-z))

	# derivative of sigmoid function
	def sigmoidPrime(self, z):
		'''
		z: a scalar, vector, or matrix
		'''
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	# propagate input through the network
	# vector in scalar out (or feature matrix in vector out)
	def forward(self, X, verbose=False):
		ones = np.ones([X.shape[0], 1])
		X = np.append(ones, X, axis = 1)
		if verbose:
			print('X: {}, X.shape {}:'.format(X, X.shape))
			print('W: {}, W.shape {}:'.format(self.W, self.W.shape))
			print('WXt: {}:'.format(np.matmul(self.W, X.T)))
		output = self.sigmoid(np.matmul(self.W, X.T)) # goes like [1,k] x [n,k].T ~ [1,n]
		return output

	# binary cross entropy loss
	# return single scalar value from vector of targets and predictions
	# this is the negative log likelihood for N observations
	def costFunction(self, X, y):
		yhat = self.forward(X)
		N = X.shape[0]
		cost = - (1/N) * (np.matmul(y, np.log(yhat)) + np.matmul(1-y, np.log(1-yhat)))
		return cost

	# compute gradient of cost function wrt W and b parameters
	# has same number of elements as columns k in X_jk
	def costFunctionGradient(self, X, y):
		yhat = self.forward(X)
		ones = np.ones([X.shape[0], 1])
		X = np.append(ones, X, axis = 1)
		N = X.shape[0]
		grad = (1/N) * np.matmul((yhat - y).T, X) # [n,1].T x [n,k] ~ [1,k]
		return grad

	# used with scipy minimization
	def objectiveFunction(self, x, *args):
		X, y = args[0], args[1]
		# update model parameters
		self.W = x
		# recompute cost and gradient
		cost = self.costFunction(X, y)
		grad = self.costFunctionGradient(X, y)
		return cost, grad

	def callbackFunction(self, x):
		self.J.append(self.costFunction(self.X, self.y))
	
	def train(self, X, y):
		'''
		X: a feature matrix of training data
		y: a vector of associated target labels
		'''
		# callback variables
		self.X = X
		self.y = y
		res = optimize.minimize(fun = self.objectiveFunction, 
			x0 = self.W,   # starting values for model weights
			args = (X, y), # extra args passed to the objective
			jac = True,    # if TRUE, then objective returns both cost and grad
			method = 'BFGS', 
			callback = self.callbackFunction,
			options = {'maxiter' : 1000, 'disp' : True})
		self.W = res.x
		return res

	def gradient_descent(self, X, y, alpha=0.3, max_iters=5000):
		'''
		X: a feature matrix of training data
		y: a vector of associated target labels
		alpha: the gradient descent learning rate
		max_iters: the max number of gradient descent iterations
		'''
		print('Learning Rate {}:'.format(alpha))
		for i in range(max_iters):
			self.W = self.W - alpha * self.costFunctionGradient(X, y)
			if i%500 == 0:
				print('Epoch {}, loss {}:'.format(i, self.costFunction(X, y)))
		print("W: {}".format(self.W))

if __name__ == '__main__':

	# load the dataset
	data = pd.read_csv('ex2data1.csv')

	# extract the feature matrix X
	X = data.drop(['y'], axis='columns').values
	print('X: {} \nX.shape {}:'.format(X, X.shape))
	print(X)

	# ... and target labels y
	y = data['y'].values
	print('y: {} \ny.shape {}:'.format(y, y.shape))

	# normalize X for training
	X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
	nfeatures = X.shape[1]

	lr = LogisticRegression(nfeatures=nfeatures)
	print('W: {} \nW.shape {}:'.format(lr.W, lr.W.shape))

	# predict X
	yhat = lr.forward(X)
	print('yhat:', yhat)

	# compute cost function
	J = lr.costFunction(X, y)
	print('J: {} \nJ.shape {}:'.format(J, J.shape))

	# compute gradients
	dJ = lr.costFunctionGradient(X, y)
	print('dJ: {} \ndJ.shape {}:'.format(dJ, dJ.shape))

	# fit model parameters W according to data (X,y)
	print('performing BFGS optimization:')
	result = lr.train(X, y)
	print('result: ', result)

	# print result of callback function
	print('costs: ', lr.J, len(lr.J))

	# plot the results
	xvar = np.arange(0, len(lr.J), 1)
	yvar = lr.J
	plt.plot(xvar, yvar, 'r--')
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.title('Cost vs. BFGS Iterations')
	plt.xticks(xvar)
	plt.show()

	# try gradient descent. Do we get same result as with BFGS? Yes!
	print('performing gradient descent:')
	print('W: {} \nW.shape {}:'.format(lr.W, lr.W.shape))
	lr.resetWeights()
	lr.gradient_descent(X, y)

	# check if never before seen test case works
	#x_test = np.array([0.8, 1.8, 2.8]).reshape(1,3)
	#print('X: {} \nX.shape {}:'.format(x_test, x_test.shape))
	#test = lr.forward(x_test)
	#print(test)






