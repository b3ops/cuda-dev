from numba import cuda
import numpy

@cuda.jit
def add_kernel(a, b, c):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

def test_add():
    n = 1000000
    A = cuda.to_device(numpy.arange(n))
    B = cuda.to_device(numpy.arange(n))
    C = numpy.zeros_like(A)
    add_kernal(A,B,C)

test_add()