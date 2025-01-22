from numba import cuda
from colorama import Fore, Back, Style
from numba import vectorize, float64
import numpy as np
import math, time, sys
import matplotlib.pyplot as plt

# golden ratio
phi = (1 + np.sqrt(5))/2

# CUDA kernal
@cuda.jit
def kernal_compute(x, y, phase):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = np.sin(phase + i * 0.01)
        y[i] = x[i] * x[i]  * phi 

def recursive_compute(x, y, n, depth=0, direction=1, phase=0):
    if depth >= 100:
        return

    tpb = 1024
    bpg = (n + tpb - 1)//tpb

    # copy to gpu
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)

    kernal_compute[bpg, tpb](d_x, d_y, phase)
    # sync
    cuda.synchronize()

    d_x.copy_to_host(x)
    d_y.copy_to_host(y)

    phase += direction * 0.01
    if phase > np.pi or phase < -np.pi:
        direction *= -1

    print(Fore.RED + 
    '********************** start the compilation ***************************')
    print('Depth:', depth)
    print(Style.RESET_ALL)
    print('x:', x[:10])
    print('y:', x[:10])
    
    recursive_compute(x, y, n, depth + 1, direction, phase)

# init
n = 1000000
x = np.ones(n, dtype=np.float32)
y = np.zeros(n, dtype=np.float32)

# CUDA Kernal 
TPB = 1024
blocks = (n + TPB - 1)//TPB 

start =  time.time()
recursive_compute(x, y, n)
print('Compute time:', time.time() - start)


plt.figure(figsize=(12, 6))
plt.plot(x, label= 'sine wave')
plt.plot(y, label='golden ratio squared')
plt.legend()
plt.show()
print('Plotting time:', time.time()-start)