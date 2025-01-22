import numpy as np
from numba import cuda

@cuda.jit
def simple_kernel(A):
    i = cuda.grid(1)
    if i < A.size:
        A[i] = A[i] * 2

# Initialize data on the host
n = 1024 * 1024  # 1 million elements
A = np.ones(n, dtype=np.float32)

# Allocate memory on the device
d_A = cuda.to_device(A)

# Set block and grid sizes
block_size = 256
grid_size = (n + block_size - 1) // block_size

# Launch the kernel
simple_kernel[grid_size, block_size](d_A)

# Copy the result back to the host
A = d_A.copy_to_host()