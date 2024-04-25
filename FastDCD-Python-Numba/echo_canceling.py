# to run echo canceling demonstration
# run on terminal exec(open("echo_canceling.py").read())



from numba import jit
import numpy as np
from scipy import signal
import time
import pandas as pd

exec(open("DCBF_RLSDCD.py").read())
exec(open("standard_RLSDCD.py").read())
exec(open("DCBF_RLSDCD.py").read())



########

@jit
def RLS(hi, x, s, M, Nit, lambdarls, delta):
    """RLS algorithm"""
    P=delta*np.eye(M)
    w=np.zeros(M)
    g=np.zeros(M)
    e=np.zeros(Nit)
    lambdai=1.0/lambdarls
    for n in range(Nit):
        u=x[np.arange(n+M-1,n-1,-1)]
        y=np.dot(u,w)            # dot product
        e[n]=s[n]-y
        g = P@u
        gamma=lambdarls+np.dot(u, g)
        gammainv=1.0 / gamma
        w = w + (e[n] * gammainv) * g
        g = g * np.sqrt(np.abs(gammainv))
        P = (lambdai) * (P - np.outer(g, g) )
    return (w, e)
       
def test_DCBF():
    print("running test_DCBF \n")
    sigmav=0.1
    lambdarls = 0.99
    delta = 1.0
    rng = np.random.default_rng()

    L=5
    M=np.array([10, 100, 1000]) 
    N=5_000
    NM=np.size(M)
    TRLS=np.zeros((NM,L))
    TRLSDCD=np.zeros((NM,L))
    TRLSDCD4=np.zeros((NM,L))
    TfRLSDCD=np.zeros((NM,L))
    TfRLSDCD4=np.zeros((NM,L))
    for k in range(np.size(M)):
        hi=rng.normal(size=M[k])
        x=rng.normal(size=N)
        s=signal.lfilter(hi,1,x)+sigmav*rng.normal(size=N)
        x=np.concatenate((np.zeros(M[k]-1),x))
        Mk = M[k]
        print("M= ", M[k], "\n")
        for i in range(L):
            time0=time.perf_counter()
            fDCBF_RLSDCD(hi, x, s, M[k], N, lambdarls, delta, 1)
            time1=time.perf_counter()
            TfRLSDCD[k,i]=time1-time0
            time0=time.perf_counter()
            fDCBF_RLSDCD(hi, x, s, M[k], N, lambdarls, delta, 4)
            time1=time.perf_counter()
            TfRLSDCD4[k,i]=time1-time0
            time0=time.perf_counter()
            RLS(hi, x, s, M[k], N, lambdarls, delta)
            time1=time.perf_counter()
            TRLS[k,i]= time1-time0 #timeit.timeit('RLS(hi, x, s, M[k], lambdarls, delta)',  
                                   # setup='from test_DCBF import RLS, hi, x, s, M, k, lambdarls, delta', number=1)
            time0=time.perf_counter()
            RLSDCD(hi, x, s, M[k], N, lambdarls, delta, 1)
            time1=time.perf_counter()

            TRLSDCD[k,i]=time1-time0
            time0=time.perf_counter()
            RLSDCD(hi, x, s, M[k], N, lambdarls, delta, 4)
            time1=time.perf_counter()
            TRLSDCD4[k,i]=time1-time0
            print("i= ", i, "\nTempo RLS\t\t", TRLS[k,i],
                "\nTempo RLSDCD Nu = 1\t", TRLSDCD[k,i],
                "\nTempo RLSDCD Nu = 4\t", TRLSDCD4[k,i],
                "\nTempo fRLSDCD Nu = 1\t", TfRLSDCD[k,i],
                "\nTempo fRLSDCD Nu = 4\t", TfRLSDCD4[k,i], 
                  "\n\n")
        print("\n")
    MefRLSDCD=np.mean(TfRLSDCD,axis=1)
    MeRLSDCD=np.mean(TRLSDCD,axis=1)
    MefRLSDCD4=np.mean(TfRLSDCD4,axis=1)
    MeRLSDCD4=np.mean(TRLSDCD4,axis=1)
    MeRLS=np.mean(TRLS,axis=1)
    MifRLSDCD=np.min(TfRLSDCD,axis=1)
    MiRLSDCD=np.min(TRLSDCD,axis=1)
    MifRLSDCD4=np.min(TfRLSDCD4,axis=1)
    MiRLSDCD4=np.min(TRLSDCD4,axis=1)
    MiRLS=np.min(TRLS,axis=1)
    ResPython = pd.DataFrame(
        {
            "M": M,
            "Mean RLS": MeRLS,
            "Mean RLSDCD1": MeRLSDCD,
            "Mean RLSDCD4": MeRLSDCD4,
            "Mean fRLSDCD1": MefRLSDCD,
            "Mean fRLSDCD4": MefRLSDCD4,
            "Min RLS": MiRLS,
            "Min RLSDCD1": MiRLSDCD,
            "Min RLSDCD4": MiRLSDCD4,
            "Min fRLSDCD1": MifRLSDCD,
            "Min fRLSDCD4": MifRLSDCD4,
        }
    )
    ResPython.to_csv("ResultsPythonL100Vector.csv")
    #return (MefRLSDCD, MeRLSDCD, MefRLSDCD4, MeRLSDCD4, MeRLS)
    return MeRLS


def test_RLS(N, M):
    """Tests RLS function with a simple example"""
    sigmav=0.01 # 
    lambdarls = 0.9999
    delta = 1.0
    rng = np.random.default_rng()
    hi = rng.normal(size=M)
    w = np.zeros(M)
    x = rng.normal(size=N)
    xx = np.concatenate((np.zeros(M-1), x))
    s = signal.lfilter(hi, 1, x) + sigmav*rng.normal(size=N)
    (w, e) = RLS(hi, xx, s, M, N, lambdarls, delta)
    print("MSD = ", np.linalg.norm(w-hi)) # MSD should be a small number, of the order of 1e-5
    return (hi, w)


def test_RLSDCD(N, M):
    """Tests RLSDCD function with a simple example"""
    sigmav=0.01 # 
    lambdarls = 0.9999
    delta = 1.0
    Nu = 1
    rng = np.random.default_rng()
    hi = rng.normal(size=M)
    w = np.zeros(M)
    x = rng.normal(size=N)
    xx = np.concatenate((np.zeros(M-1), x))
    s = signal.lfilter(hi, 1, x) + sigmav*rng.normal(size=N)
    (w, e) = RLSDCD(hi, xx, s, M, N, lambdarls, delta, Nu)
    print("MSD = ", np.linalg.norm(w-hi))
    return (hi, w)


def test_fRLSDCD(N, M):
    """Tests fRLSDCD function with a simple example"""
    sigmav=0.01 # 
    lambdarls = 0.9999
    delta = 1.0
    Nu = 1
    rng = np.random.default_rng()
    hi = rng.normal(size=M)
    w = np.zeros(M)
    x = rng.normal(size=N)
    xx = np.concatenate((np.zeros(M-1), x))
    s = signal.lfilter(hi, 1, x) + sigmav*rng.normal(size=N)
    (w, e) = fDCBF_RLSDCD(hi, xx, s, M, N, lambdarls, delta, Nu)
    print("MSD = ", np.linalg.norm(w-hi))
    return (hi, w)



