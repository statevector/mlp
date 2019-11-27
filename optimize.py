from scipy import optimize

def objective(x, coeffs):
	return coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]

if __name__ == '__main__':

	x0 = 3.0
	mycoeffs = [1.0, -2.0, 0.0]
	myoptions = {'disp':True}

	results = optimize.minimize(objective, x0, args = mycoeffs, options = myoptions)

	print("Solution: x=%f" % results.x)