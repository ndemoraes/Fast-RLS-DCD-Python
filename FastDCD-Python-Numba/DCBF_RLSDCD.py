exec(open("DCBF_operations.py").read())

@jit
def fDCD(M, 
            H, 
            B, 
            Nu,
            Rphi,
            top,
            cb_head,
            diag_num_elements,  
            res):

    # The arguments Δw and β are also the outputs.  β will contain the residue on exit

    h = H/2
    b = 1
    Deltaw = np.zeros(M)

    for k in range(0, Nu):
        p = np.argmax(np.abs(res))
        pp = (int(cb_head[0]) + p) % int(diag_num_elements[0]) 

        while np.abs(res[p]) <= (h/2) * Rphi[pp]:  # should be top[0]+Rphi[pp], but top[0]=0
            b=b+1
            h=h/2

            if b > B:
                break

        deltawp = np.sign(res[p])*h
        Deltaw[p]=Deltaw[p]+deltawp
        
        res=update_DCBF_Vector(res, M, Rphi, top, cb_head, diag_num_elements, deltawp, p)


    return (Deltaw, res)

@jit
def fDCBF_RLSDCD(hi, 
                    x, 
                    s,
                    M, 
                    Nit,
                    lambdarls, 
                    delta, 
                    Nu):

    w=np.zeros(M)
    beta=np.zeros(M)
    Deltaw=np.zeros(M)
    e=np.zeros(Nit)
    R=np.zeros(M*(M+1)//2)  # Define matrix, stored in a vector with M(M+1)/2 elements
    for i in range(M):
        R[i] = delta
    top=np.zeros(M)         # top points to the first element in the vector corresponding to a given diagonal
    cb_head=np.zeros(M)     # cb_head points to the element in the diagonal corresponding to the first row
    diag_num_elements=np.zeros(M) # number of elements in each diagonal (equals M-diagonal#)
    index = 0
    for i in range(M):
        diag_num_elements[i] = M-i
        top[i] = index
        index += diag_num_elements[i]

    for n in range(0,Nit):
        u=x[np.arange(n+M-1,n-1,-1)]
        y=np.dot(u,w)            # dot product 
        e[n]=s[n]-y
        update_DCBF_Matrix(R, top, cb_head, diag_num_elements, u, M, lambdarls)
        beta = lambdarls * beta + e[n] * u

        (Deltaw, beta)=fDCD(M, 4.0, 16, Nu, R, top, cb_head, diag_num_elements, beta) 
        w = w + Deltaw
    return (w, e)

