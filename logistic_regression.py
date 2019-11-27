import numpy as np
from scipy import optimize

from scipy.optimize import minimize, rosen, rosen_der

class Logistic_Regression():

	def __init__(self):
		# model hyper parameters
		self.inputLayerSize = 3
		#self.outputLayerSize = 1
		# model parameters
		self.W = np.random.randn(self.inputLayerSize + 1) # include bias term

	def parameters(self):
		print(" input size: {}\n output size: {}\n weights: {}\n".format(
			self.inputLayerSize, self.W))

	# 2x3 * 1x3
	# 2x4 * 1x4

	# push input through network
	# vector in scalar out (matrix in vector out)
	def forward(self, X):
		ones = np.ones([X.shape[0],1])
		X = np.append(ones, X, axis=1)
		#print('W', self.W)
		#print('b', self.b)
		#print('X', X)
		#print('XW', np.dot(X, self.W.T))
		#print('XW+b', np.dot(X, self.W.T) + self.b) # this is broadcasting
		#print('yhat', self.sigmoid(np.dot(X, self.W.T) + self.b))
		#yhat = self.sigmoid(np.dot(X, self.W) + self.b)
		#return self.sigmoid(np.dot(X, self.W.T) + self.b)
		return self.sigmoid(np.dot(X, self.W.T))

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	# compute derivative of sigmoid function
	def sigmoidPrime(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	#def costFunction(self, X, y):
	#	yhat = self.forward(X)
	#	return 1/2 * np.sum((yhat - y) ** 2)

	# binary cross entropy loss
	def costFunction(self, X, y):
		yhat = self.forward(X)
		#print('y:', y, y.shape)
		#print('X:', X, X.shape)
		print(y.T)
		print(yhat)
		print(np.dot(y.T, yhat))
		#print(np.dot(y.T, np.log(self.forward(X))) + np.dot(1-y.T, np.log(1-self.forward(X))))
		return np.matmul(y.T, np.log(yhat)) + np.matmul(1-y.T, np.log(1-yhat))

	# compute gradient of cost function wrt W and b
	# should have same number of elements as cols j in X
	def costFunctionGradient(self, X, y):
		yhat = self.forward(X) # do this first
		ones = np.ones([X.shape[0], 1])
		X = np.append(ones, X, axis = 1)
		#print(np.dot((yhat - y).T, X))
		return 1/len(X) * np.dot((yhat-y).T, X)

	def getParameters(self):
		#print(self.W.ravel())
		return self.W

	def setParameters(self, params):
		pass
		#self.W = np.reshape(params[0:self.inputLayerSize+1])

	#def setParameters(self, params):
	#	pass
		#W_start = 0
		#W_end = self.hiddenLayerSize * self.inputLayerSize
		#self.W = np.reshape(params[W1_start:W1_end])

    #def computeGradients(self, X, y):
    #    grad = self.costFunctionGradient(X, y)
    #    return grad.ravel()

	def _cost(self, x, *args):
		X, y = args[0], args[1]
		self.setParameters(x)
		cost = self.costFunction(X, y)
		print('cost', cost)
		return cost[0] # how to remove this [0] ???

	def _grad(self, x, *args):
		X, y = args[0], args[1]
		self.setParameters(x)
		grad = self.costFunctionGradient(X, y)
		print('grad', grad)
		return grad[0] # same ???

	# def cost_grad(self, x, *args):
	# 	print('x:', x)
	# 	X = args[0]
	# 	print('X:', X)
	# 	y = args[1]
	# 	print('y:', y)
	# 	self.setParameters(x)
	# 	print(self.getParameters())
	# 	cost = self.costFunction(X, y)
	# 	print('cost', cost)
	# 	grad = self.costFunctionGradient(X, y)
	# 	print('grad', grad)
	# 	return cost, grad

	def train(self, X, y):

		params = self.getParameters()
		#print('params:', params, params.shape)

		# Initial guess. Array of real elements of size (n,),
		# where ‘n’ is the number of independent variables
		x0 = self.W
		print('x0:', x0, x0.shape)
		#x0 = 1

		# If jac is a Boolean and is True, 
		# fun is assumed to return the gradient along with the objective function
		res = optimize.minimize(self._cost, 
			x0,
			args = (X, y),
			jac = self._grad, 
			method = 'BFGS', 
			options = {'maxiter' : 1000, 'disp' : True})

		print(res.x)
		#setParameters(res.x)

		return res.success






if __name__ == '__main__':

	lr = Logistic_Regression()
	#lr.parameters()

	# two examples
	#x = np.array([[-1, -2, 3], 
	#			  [ 1,  1, 1]]).reshape(2,-1)
	x = np.array([-1, -2, 3]).reshape(1,-1)
	print('X:', x)

	yhat = lr.forward(x)
	print('yhat:', yhat)
	print('end forward')

	# targets	
	#y = np.array([1, 
	#			  0]).reshape(2,-1)
	y = np.array([1]).reshape(1,-1)
	print('y:', y)

	print('cool')

	J = lr.costFunction(x, y)
	#print(J)

	dJ = lr.costFunctionGradient(x, y)
	#print(dJ)


	#params = [1,1,1,1]
	#q = lr.cost_grad(params, x, y)
	#print(q)

	lr.train(x, y)


#res = minimize()








