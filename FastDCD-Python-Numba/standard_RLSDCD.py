# @@@@@@@@@@@@
# DCD
# calculates the residue 
# called by RLSDCD
#
# @@@@@@@@@@@@
exec(open("Standard_operations.py").read())
@jit
def DCD(M, H, B, Nu, Rphi, res):

    h = H/2
    b = 1
    Deltaw = np.zeros(M)  # vector of zeros
    for k in range(0,Nu):
        p = np.argmax(np.abs(res)) # can be anything from 0 to M-1
        
        while np.abs(res[p]) <= (h/2)*Rphi[p,p]:
            b=b+1
            h=h/2
            if b > B:
                break

        deltawp = np.sign(res[p])*h
        Deltaw[p]=Deltaw[p] + deltawp   # vector element, p is index
        res = update_vector(res, Rphi, deltawp, p)

    return (Deltaw, res)
# fn DCD
@jit
def RLSDCD(hi, x, s, M, Nit, lambdarls, delta, Nu):

    w=np.zeros(M)
    beta=np.zeros(M)
    Deltaw=np.zeros(M)

    # defining and initializing the matrix
    R = initialize_matrix(M, delta)

    e=np.zeros(Nit)
    lambdai=1.0/lambdarls


    for n in range(0,Nit):
        u=x[np.arange(n+M-1,n-1,-1)]
        y=np.dot(u,w)            # dot product            
        e[n]=s[n]-y
        update_matrix(R, u, lambdarls)

        beta = lambdarls * beta + e[n] * u   #equilavent to β[N] = λrls * res[N] + e[n] * u[N]
        (Deltaw, beta) = DCD(M, 4.0, 16, Nu, R, beta)
        w = w + Deltaw


    return (w, e)




