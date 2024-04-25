
# ****************************************************************
# update DIAGONAL CIRCULAR-BUFFER FORMAT MATRIX by shifting down 
# and to the right by one column and one row 
# (used in fRLSDCD function)
# ****************************************************************
# input: DCBF matrix, vector and scalar for update equation
# output: updated DCBF matrix
# ****************************************************************
@jit
def update_DCBF_Matrix(dmtrx, top, cb_head, diag_num_elements, vtr, M, scalar):

    for i in range(0, M):

        # index of head of circular buffer before updating
        temp=int(top[i]) + int(cb_head[i])

        # shift diagonal up by moving pointer within diagonal (mod is to make it wrap around)
        cb_head[i]= (int(cb_head[i]) - 1) % int(diag_num_elements[i])

        # update a single element
        dmtrx[temp] = scalar * dmtrx[temp] + vtr[0] * vtr[i]  
    return 1

# update_DCBF_Matrix
    
    
    
# *********************************************************************
# multiply column of the DIAGONAL CIRCULAR-BUFFER FORMAT MATRIX by a  
# factor and add result to a vector (used in fDCD function)
# *********************************************************************
# input: vector, matrix, factor, column number
# output: vector <- vector + factor * matrix[:,column number]
# ********************************************************************
@jit
def update_DCBF_Vector(vtr, M, dmtrx, top, cb_head, diag_num_elements, scalar, col_num):
    
    diagonal = col_num          # diagonal starts out as column in the original standard matrix format
    diagonal_step=-1             # The algorithm steps through the diagonals until it reaches the first, 
                                 # then changes direction
    cb_offset=0                  # cb_offset is the offset from the head element in the circular buffer
                                 # inside the circular buffer for that diagonal
    kstep=1                      # The matrix is stored in a vector.  top points to the vector element
                                 # corresponding to the beginning of each diagonal
    
    # iterate over elements in the vector
    for i in range(0, M):   
        # given the diagonal, select the circular buffer element   
        cb_element = (int(cb_head[diagonal]) + cb_offset) % int(diag_num_elements[diagonal])
        # update the vector   
        vtr[i] = vtr[i] - scalar * dmtrx[int(top[diagonal])+int(cb_element)] 

        # select the next column
        diagonal = diagonal + diagonal_step
        cb_offset = cb_offset + kstep
        if diagonal == -1:
            diagonal = 1
            diagonal_step = +1
            cb_offset = col_num
            kstep = 0
    return vtr
