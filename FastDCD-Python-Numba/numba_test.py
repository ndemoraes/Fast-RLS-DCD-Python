from numba import jit
import numpy as np
import time

N = 1000
rng = np.random.default_rng()
A = rng.normal(size = (N,N))
b = rng.normal(size = N)

x = np.arange(100).reshape(10,10)
def test(b):
    trace = 0.0
    for i in range(b.shape[0]):
        trace += np.tanh(b[i, i])
    return b + trace

@jit
def test1(b):
    trace = 0.0
    for i in range(b.shape[0]):
        trace += np.tanh(b[i, i])
    return b + trace


time0=time.perf_counter()
test(x)
time1=time.perf_counter()
print("Sem JIT: ", time1-time0)

time0=time.perf_counter()
test1(x)
time1=time.perf_counter()
print("Com JIT: ", time1-time0)