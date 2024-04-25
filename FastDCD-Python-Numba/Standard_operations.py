#@jit(nopython=False)
def initialize_matrix(size, delta):
    # initialize major diagonal to all "deltas", rest to zeros
    initialized = delta * np.eye(size)
    return initialized



# ***********************************************************************
# update regular matrix by shifting down to the right by one column + row 
# (used in RLSDCD function)
# ***********************************************************************
# input: matrix, vector and scalar for update equation
# output: updated matrix
# ***********************************************************************
@jit
def update_matrix(mtrx, vtr, scalar):
    M = len(vtr)
    # Update matrix lower block
    mtrx[1:, 1:] = mtrx[:M-1, :M-1]

    # Update first column
    mtrx[0,:] = scalar * mtrx[0,:] + vtr[0] * vtr
    # Copy first column to first row to keep matrix symmetric
    mtrx[1:,0] = mtrx[0, 1:]  

# *****************************************************************************************
# multiply column of the regular matrix by a factor and add result to a vector 
# (used in DCD function)
# *****************************************************************************************
# input: vector, matrix, factor, column number
# output: vector <- vector + factor * matrix[:,column number]
# *****************************************************************************************
@jit
def update_vector(vtr, mtrx, scalar, col_numb):
    # mtrx[row, col]
    vtr = vtr - scalar * mtrx[col_numb, :] # mtrx is symmetric.  this should be faster.
    return vtr

def print_matrix(mtrx):
    # mtrx[row, col]
    for row in range(0,mtrx.shape[0]):
        for col in range(0,mtrx.shape[0]):
        		print(mtrx[row,col], "  ")  
        print("\n")
 
