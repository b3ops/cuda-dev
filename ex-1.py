from numba import cuda
import numpy as np

# CUDA kernal
@cuda.jit
def kernal_compute(x, y):
    i = cuda.grid(1)
    if i < x.size:
        if x[i] < 10.0:
            x[i] += 1.0
        elif x[i] == 10.0:
            x[i] = -10.0
        elif x[i] > -10.0:
            x[i] -= 1.0
        elif x[i] == -10.0:
            x[i] = 1.0
        y[i] = x[i] * x[i]   
        #print(f'x[{idx}] = {x[idx]}, y[{idx}] = {y[idx]}]')

def recursive_compute(x, y, n, depth=0):
    if depth >= 10:
        return

    tpb = 1024
    bpg = (n + tpb - 1)//tpb

    # copy to gpu
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)

    kernal_compute[bpg, tpb](d_x, d_y)
    # sync
    cuda.synchronize()

    d_x.copy_to_host(x)
    d_y.copy_to_host(y)

    print('Depth:', depth)
    print('x:', x[:10])
    print('y:', x[:10])
    
    recursive_compute(x, y, n, depth + 1)

# init
n = 1000000
x = np.ones(n, dtype=np.float32)
y = np.zeros(n, dtype=np.float32)

# CUDA Kernal 
TPB = 1024
blocks = (n + TPB - 1)//TPB 

recursive_compute(x, y, n)
