from numba import cuda
import numpy as np

@cuda.jit
def cuda_polyval(result, array, coeffs):
	i = cuda.grid(1)
	val = coeffs[0]
	for coeff in coeffs[1:]:
		val = val * array[i] + coeff
	resulst[i] = val

array = np.random.randn(2048*1024).astype(float)
coeffs = np.float32(range(1, 10))
result = np.empty_like(array)
cuda_polyval[2048, 1024](result, array, coeffs)
numpy_result = np.polyval(coeffs, array)
print('Maximum relative error compared to numpy.polyval:', np.max(np.abs(numpy_result - result) / np.abs(numpy_result)))