import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	N, M = X.shape
	X_new = np.hstack((np.ones((N,1)), X))
	W = np.zeros((M+1,))
	losses = []

	for i in range(iterations):
		Z = np.dot(X_new, W.T)
		S = np.divide(1, (1+np.exp(-Z)))
		loss = -np.sum(y * np.log(S + 1e-15) + (1 - y) * np.log(1 - S + 1e-15))
		losses.append(loss)
		g = X_new.T @ (S - y)
		W = W - learning_rate * g

	return (W.tolist(), losses)
	pass