from numba import cuda
import numpy as np

@cuda.jit
def cudaKernal1(array):
	th_pos = cuda.grid(1)
	array[th_pos] += 0.5
