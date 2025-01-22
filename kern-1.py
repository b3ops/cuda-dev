from numba import cuda
import numpy as np 

@cuda.jit
def add_kernal(x, y, out):
	tx = cuda.threadIdx.x 
	ty = cuda.blockidx.x 

	block_size = cuda.blockDim.x 
	grid_size = cuda.gridDim.x 

	start = tx + ty * block_size 
	stride = block_size * grid_size

	for i in range(start, x.shape[0], stride):
		out[i] = x[i] + y[i]

n = 10000
x = np.arange(n).astype(np.float32)
y = 2 * x
out = np.empty_like(x)

threads_per_block = 128
blocks_per_grid = 30

add_kernal[blocks_per_grid, threads_per_block](x, y, out)
print(out[:10])