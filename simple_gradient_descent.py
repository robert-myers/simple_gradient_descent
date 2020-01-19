import numpy as np

def get_gradient_at_b(x, y, m, b):
	"""
	INPUT
	x: array of x values
	y: array of y values
	m: slope
	b: intercept

	OUTPUT
	gradient at intercept
	"""
	N = len(x)
	diff = np.sum(y - (m * x + b))
	b_gradient = -2/N * diff
	return b_gradient

def get_gradient_at_m(x, y, m, b):
	"""
	INPUT
	x: array of x values
	y: array of y values
	m: slope
	b: intercept

	OUTPUT
	gradient at slope
	"""
	N = len(x)
	diff = np.sum(x * (y - (m * x + b)))
	m_gradient = -2/N * diff
	return m_gradient

def step_gradient(x, y, b_current, m_current, learning_rate=0.01):
	b_gradient = get_gradient_at_b(x=x, y=y, b=b_current, m=m_current)
	m_gradient = get_gradient_at_m(x=x, y=y, b=b_current, m=m_current)
	b = b_current - (learning_rate * b_gradient)
	m = m_current - (learning_rate * m_gradient)
	return np.array([b, m])
