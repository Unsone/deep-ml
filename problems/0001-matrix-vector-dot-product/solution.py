import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	A = np.array(a)
	B = np.array(b)
	num1 = A.shape[1]
	num2 = B.shape[0]
	if num1 != num2:
		return -1
	return A @ B
	pass