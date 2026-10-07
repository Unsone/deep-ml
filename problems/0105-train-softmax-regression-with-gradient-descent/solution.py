import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	N, M = X.shape
	C = max(y) + 1
	X_0 = np.hstack((np.ones((N,1)), X))
	W = np.zeros((C, M+1))
	losses = []

	for i in range(iterations):
		Z = np.dot(X_0, W.T)
		Z_max = np.max(Z, axis=1, keepdims=True)
		exp_Z = np.exp(Z - Z_max)
		P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

		correct_logprobs = -np.log(P[np.arange(N), y] + 1e-15) 
		loss = sum(correct_logprobs)
		losses.append(loss)

		Y_onehot = np.zeros((N, C))
		Y_onehot[np.arange(N), y] = 1
		Error = P - Y_onehot
		gradient = np.dot(Error.T, X_0)
		W = W - learning_rate * gradient

	coefficients = W.tolist()
	return coefficients, losses
	pass