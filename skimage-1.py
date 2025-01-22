import numpy as np 
import cupy as cp 
import math, time, sys
from termcolor import colored, cprint
from colorama import Fore, Back, Style
from numba import vectorize, float64
from numba import jit, int32

x = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

#print('\nstart the compilation....')
print(Fore.RED + 
'********************** start the compilation ***************************')
print(Style.RESET_ALL)

start = time.time()
go_fast(x)
end = time.time()
print('Elasped (with compilation) = %s' % (end - start))

start = time.time()
go_fast(x)
end = time.time()
print('Elasped (after compilation) = %s' % (end - start))

print(Fore.RED + 
'\n********************** the vectorize decorator ***************************')
print(Style.RESET_ALL)

@vectorize([float64(float64, float64)])
def f(x, y):
    return x + y

a = np.arange(6)
print('f(a, a) arange:', f(a, a))

a = np.linspace(0, 1, 6)
print('f(a, a) linspace:', f(a, a))

print(Fore.GREEN + 
 '\n *********************** GPU SLAM ************************************')
print(Style.RESET_ALL)