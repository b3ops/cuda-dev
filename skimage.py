import numpy as np 
import cupy as cp 
import math
from numba import cuda
from numba import jit, int32

x = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fast(a):
	trace = 0.0
	for i in range(a.shape[0]):
		trace += np.tanh(a[i, i])
	return a + trace

print('\n go_fast: \n\n', go_fast(x))
print('\n', cuda.gpus, '\n')

print('nopython')
@jit(nopython=True)
def f(x, y):
	return x + y 

f(1,2)
print('@jit(nopython added to def f(1, 2) = ' , f(1,2))
print('def f(2**31, 2**31+1)', f(2**31, 2**31+1))

print('\nnogil ')
@jit(nogil=True)
def f(x, y):
	return x + y 

f(1,2)
print('@jit(nogil added to def f(1, 2) = ' , f(1,2))
print('def f(2**31, 2**31+1)', f(2**31, 2**31+1))

print('\ncache..')
@jit(cache=True)
def f(x, y):
	return x + y 

f(1,2)
print('@jit cache added to def f(1, 2) = ' , f(1,2))
print('def f(2**31, 2**31+1)', f(2**31, 2**31+1))

print('\nsquare..')
@jit
def square(x):
	return x**2

print('\nsquare(4): ', square(4))
print('square(6): ', square(6))

@jit
def hypot(x, y):
	return math.sqrt(square(x)+square(y))

print('\nhypot(2, 5):', hypot(2, 5))
