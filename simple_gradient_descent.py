import numpy as np

def get_gradient_at_b(X, y, b, m):
	"""
	Parameters
	----------
	X : array_like of int or float
		The X coordinates.
	y : array_like of int or float
		The Y coordinates.
	b : int or float
		The intercept value.
	m : int or float
		The slope value.

	Returns
	-------
	b_gradient : float
		The gradient value at the intercept.
	"""
	X, y = np.array(X), np.array(y)
	N = len(y)
	diff = np.sum(y - (m * X + b))
	b_gradient = -2/N * diff
	return b_gradient

def get_gradient_at_m(X, y, b, m):
	"""
	Parameters
	----------
	X : array_like of int or float
		The X coordinates.
	y : array_like of int or float
		The Y coordinates.
	b : int or float
		The intercept value.
	m : int or float
		The slope value.

	Returns
	-------
	m_gradient : float
		The gradient value at the slope.
	"""
	X, y = np.array(X), np.array(y)
	N = len(y)
	diff = np.sum(X * (y - (m * X + b)))
	m_gradient = -2/N * diff
	return m_gradient

def step_gradient(X, y, b_current, m_current, learning_rate=0.01):
	"""
	Parameters
	----------
	X : array_like of int or float
		The X coordinates.
	y : array_like of int or float
		The Y coordinates.
	b_current : int or float
		The current intercept guess.
	m_current : int or float
		The current slope guess.
	learning_rate : float
		The proportional size of step to take (the default is 0.01).

	Returns
	-------
	step : array of float
		The new intercept value and the new slope value.
	"""
	b_gradient = get_gradient_at_b(X, y, b_current, m_current)
	m_gradient = get_gradient_at_m(X, y, b_current, m_current)
	b = b_current - (learning_rate * b_gradient)
	m = m_current - (learning_rate * m_gradient)
	step = np.array([b, m])
	return step

def gradient_descent(X, y, b=0, m=0, learning_rate=0.01, num_iterations=1000, y_predict=False):
	"""
	Parameters
	----------
	X : array_like of int or float
		The X coordinates.
	y : array_like of int or float
		The Y coordinates.
	b : int or float
		The starting intercept guess (the default is 0).
	m : int or float
		The starting intercept guess (the default is 0).
	learning_rate : float
		The proportional size of step to take (the default is 0.01).
	num_iterations : int
		The number of steps to descend (the default is 1000).
	y_predict : bool
		Whether or not to calculate the y_predictions
		and return them (the default is False).

	Returns
	-------
	b, m, y_predictions : tuple of float and optional array
		The final intercept value, the final slope value,
		and optionally the predicted values of y.
	"""
	for _ in range(num_iterations):
		b, m = step_gradient(X, y, b, m, learning_rate)
	if y_predict:
		y_predictions = m * X + b
		return b, m, y_predictions
	else:
		return b, m

def mean_squared_error(y, y_predictions, root=True):
	"""
	Parameters
	----------
	y : array of float
		The true Y coordinates also known as theta values.
	y_predictions : array of float
		The predicted Y coordinates also known as theta hat values.
	root : bool
		Whether or not to calculate and return the
		sqaure root of the mean squared error (the default is True).

	Returns
	-------
	root_mean_squared_error or mean_squared_error : float
		The square root of the mean squared error or
		optionally the mean squared error.
	"""
	mean_squared_error = np.average(np.square(y - y_predictions))
	if root:
		root_mean_squared_error = np.sqrt(mean_squared_error)
		return root_mean_squared_error
	else:
		return mean_squared_error

def r_squared(y, y_predictions):
	"""
	Parameters
	----------
	y : array of float
		The true Y coordinates also known as theta values.
	y_predictions : array of float
		The predicted Y coordinates also known as theta hat values.

	Returns
	-------
	r_squared : float
		The coefficient of determination.
	"""
	ssr = np.sum(np.square(y - y_predictions))
	sst = np.sum(np.square(y - np.mean(y)))
	r_squared = 1 - (ssr / sst)
	return r_squared
