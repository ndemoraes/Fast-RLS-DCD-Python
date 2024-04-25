# **************************************************************
# structure to store matrix in DIAGONAL, CIRCULAR-BUFFER format 
# ***************************************************************
# dmtrx - matrix in diagonal circular-buffer format (DCBF)
# cb_head - index to location of head of circular buffer in each diagonal column
# diag_num_elements - number of elements in diagonal column, and thus 
#                 also the number of columns in jagged array
# ***************************************************************
from collections import namedtuple

DCBF_Matrix = namedtuple('DCBFMatrix', ['dmtrx', 'cb_head', 'diag_num_elements'])



def initialize_DCBF_Matrix(size, delta):	
    """ initialize first diagonal to delta, rest of dmtrx to all zeros"""
    idmtrx=[np.full(size, delta)]
    for j in range(0,size):
        idmtrx.append(np.zeros(size-j))
    icb_head=np.zeros(size, dtype=int)
    idiag_num_elements=range(size,0,-1)
    idcbm = DCBF_Matrix(idmtrx, icb_head, idiag_num_elements)
	
    return idcbm


def initialize_DCBF_Matrix_with_RegularMatrix(size, regular_matrix):
    """used by testing functions"""
	
    idmtrx=[np.zeros(size)]
    for j in range(0,size):
        idmtrx.append(np.zeros(size-j))
    
    for row in range(0,size):
        for col in range(row, size):   
            diag = col - row
            idmtrx[diag][row] = regular_matrix[row,col]  
    icb_head=np.zeros(size, dtype=int)
    idiag_num_elements=range(size,0,-1)
    idcbm = DCBF_Matrix(idmtrx, icb_head, idiag_num_elements)
	
    return idcbm


def print_DCBF_Matrix(mtrx):
    for diagonal in range(0, len(mtrx.dmtrx)):
        print("diagonal: ", diagonal, ", values in CB: ", mtrx.dmtrx[diagonal], "\n")

    print("\n")
    print("Indices of head of circular buffer in each diagonal: ", mtrx.cb_head, "\n")
    print("Number of elements in each diagonal", mtrx.diag_num_elements, "\n")
	

# ****************************************************************
# update DIAGONAL CIRCULAR-BUFFER FORMAT MATRIX by shifting down 
# and to the right by one column and one row 
# (used in fRLSDCD function)
# ****************************************************************
# input: DCBF matrix, vector and scalar for update equation
# output: updated DCBF matrix
# ****************************************************************

def update_DCBF_Matrix(mtrx, vtr, scalar):

    for i in range(0, len(vtr)):

        # index of head of circular buffer before updating
        temp=mtrx.cb_head[i]

        # shift diagonal up by moving pointer within diagonal (mod is to make it wrap around)
        mtrx.cb_head[i]= (mtrx.cb_head[i] - 1) % mtrx.diag_num_elements[i] 

        # update a single element
        mtrx.dmtrx[i][mtrx.cb_head[i]] = scalar * mtrx.dmtrx[i][temp] + vtr[0] * vtr[i]  
    return 1

# update_DCBF_Matrix
    
    
    
# *********************************************************************
# multiply column of the DIAGONAL CIRCULAR-BUFFER FORMAT MATRIX by a  
# factor and add result to a vector (used in fDCD function)
# *********************************************************************
# input: vector, matrix, factor, column number
# output: vector <- vector + factor * matrix[:,column number]
# ********************************************************************

def update_DCBF_Vector(vtr, mtrx, scalar, col_num):
    
    diagonal = col_num          # diagonal starts out as column in the original standard matrix format
    diagonal_step=-1             # The algorithm steps through the diagonals until it reaches the first, 
                                 # then changes direction
    cb_offset=0                  # cb_offset is the offset from the head element in the circular buffer
                                 # inside the circular buffer for that diagonal
    kstep=1                       
    
    # iterate over elements in the vector
    for i in range(0, len(vtr)):   
        # given the diagonal, select the circular buffer element   
        cb_element = (mtrx.cb_head[diagonal] + cb_offset) % mtrx.diag_num_elements[diagonal]
        # update the vector   
        vtr[i] = vtr[i] - scalar * mtrx.dmtrx[diagonal][cb_element] 

        # select the next column
        diagonal = diagonal + diagonal_step
        cb_offset = cb_offset + kstep
        if diagonal == -1:
            diagonal = 1
            diagonal_step = +1
            cb_offset = col_num
            kstep = 0
    return vtr
