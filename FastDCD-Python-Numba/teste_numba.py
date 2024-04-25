time0=time.perf_counter()
test(x)
time1=time.perf_counter()
print("Sem JIT: ", time1-time0)

time0=time.perf_counter()
test1(x)
time1=time.perf_counter()
print("Com JIT: ", time1-time0)