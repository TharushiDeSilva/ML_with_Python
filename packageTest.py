# testing the numpy installation 
import numpy as np 

x = np.array([[1,2,3], [4,5,6]])
print("x:\n{}".format(x))

#testing the scipy installtion 
 
from scipy import sparse
eye = np.eye(4) # identity matrix 
print("NumPy array:\n{}".format(eye))

# convert numpy array ro Scipy sparse matric is CSR format 
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

# COO representation of the above Identity matrix. similar to CSR
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo))


