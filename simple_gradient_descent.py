import numpy as np

def get_gradient_at_b(x, y, b, m):
	"""
	Parameters
	----------
	x : array_like of int or float
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
	x, y = np.array(x), np.array(y)
	N = len(x)
	diff = np.sum(y - (m * x + b))
	b_gradient = -2/N * diff
	return b_gradient

def get_gradient_at_m(x, y, b, m):
	"""
	Parameters
	----------
	x : array_like of int or float
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
	x, y = np.array(x), np.array(y)
	N = len(x)
	diff = np.sum(x * (y - (m * x + b)))
	m_gradient = -2/N * diff
	return m_gradient

def step_gradient(x, y, b_current, m_current, learning_rate=0.01):
	"""
	Parameters
	----------
	x : array_like of int or float
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
	b_gradient = get_gradient_at_b(x, y, b_current, m_current)
	m_gradient = get_gradient_at_m(x, y, b_current, m_current)
	b = b_current - (learning_rate * b_gradient)
	m = m_current - (learning_rate * m_gradient)
	step = np.array([b, m])
	return step
