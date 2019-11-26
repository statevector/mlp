import numpy as np
from scipy import optimize
#from scipy.optimize import minimize

class Logistic_Regression():

	def __init__(self):
		# model hyper parameters
		self.inputLayerSize = 3
		self.outputLayerSize = 1
		# model parameters
		self.W = np.random.randn(self.inputLayerSize)
		self.b = np.random.randn(1)

	def parameters(self):
		print(" input size: {}\n output size: {}\n weights: {}\n".format(
			self.inputLayerSize, self.outputLayerSize, self.W))

	# push input through network
	def forward(self, X):
		print('W', self.W)
		print('b', self.b)
		print('X', X)
		print('XW', np.dot(X, self.W))
		print('XW+b', np.dot(X, self.W) + self.b) # this is broadcasting
		print('yhat', self.sigmoid(np.dot(X, self.W) + self.b))
		#yhat = self.sigmoid(np.dot(X, self.W) + self.b)
		return self.sigmoid(np.dot(X, self.W) + self.b)

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
		#print(np.dot(y.T, np.log(self.forward(X))))
		#print(np.dot(y.T, np.log(self.forward(X))) + np.dot(1-y.T, np.log(1-self.forward(X))))
		return np.dot(y.T, np.log(yhat)) + np.dot(1-y.T, np.log(1-yhat))

	# compute gradient of cost function wrt W and b
	# should have same number of elements as cols j in X
	def costFunctionGradient(self, X, y):
		yhat = self.forward(X)
		#print(np.dot((yhat - y).T, X))
		return 1/len(X) * np.dot((yhat-y).T, X)

	def getParameters(self):
		return np.concatenate((self.W.ravel(), self.b.ravel()))
    
    #def setParameters(self, params):
    #    #Set W1 and W2 using single paramater vector.
    #    W1_start = 0
    #    W1_end = self.hiddenLayerSize * self.inputLayerSize
    #    self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
    #    W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
    #    self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

	def train(self, X, y):

		params = self.getParameters()
		x0 = y

		res = minimize(f, x0, method='BFGS', tol=1e-6)

		res.x
		setParameters(res.x)

		return res.success





if __name__ == '__main__':

	lr = Logistic_Regression()
	#lr.parameters()

	# two examples
	x = np.array([[-1, -2, 3], 
				  [ 1,  1, 1]]).reshape(2,-1)
	print('X:', x)

	yhat = lr.forward(x)
	print('yhat:', yhat)
	print('end forward')

	# targets	
	y = np.array([1, 
				  0]).reshape(2,-1)
	print('y:', y)

	print('cool')

	J = lr.costFunction(x, y)
	#print(J)

	dJ = lr.costFunctionGradient(x, y)
	print(dJ)




#res = minimize()








