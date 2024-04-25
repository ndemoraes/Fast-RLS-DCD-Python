exec(open("DCBF_operations.py").read())


def fDCD(M, 
            H, 
            B, 
            Nu,
            Rphi,  
            res):

    # The arguments Δw and β are also the outputs.  β will contain the residue on exit

    h = H/2
    b = 1
    Deltaw = np.zeros(M)

    for k in range(0, Nu):
        p = np.argmax(np.abs(res))
        pp = (Rphi.cb_head[0] + p) % Rphi.diag_num_elements[0]

        while np.abs(res[p]) <= (h/2) * Rphi.dmtrx[0][pp]:
            b=b+1
            h=h/2

            if b > B:
                break

        deltawp = np.sign(res[p])*h
        Deltaw[p]=Deltaw[p]+deltawp
        
        res=update_DCBF_Vector(res, Rphi, deltawp, p)


    return (Deltaw, res)


def fDCBF_RLSDCD(hi, 
                    x, 
                    s,
                    M, 
                    lambdarls, 
                    delta, 
                    Nu):

    Nit=len(s)
    w=np.zeros(M)
    beta=np.zeros(M)
    Deltaw=np.zeros(M)
    e=np.zeros(Nit)

    R = initialize_DCBF_Matrix(M, delta)

    lambdai=1.0/lambdarls

    for n in range(0,Nit):
        u=x[np.arange(n+M-1,n-1,-1)]
        y=np.dot(u,w)            # dot product 
        e[n]=s[n]-y
        update_DCBF_Matrix(R, u, lambdarls)
        # the macro @. is provided to convert every function call, operation, and assignment in an expression into the "dotted" version.
        beta = lambdarls * beta + e[n] * u

        (Deltaw, beta)=fDCD(M, 4.0, 16, Nu, R, beta) 
        w = w + Deltaw
    return (w, e)

