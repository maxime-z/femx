import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix, dok_matrix, dia_matrix


# Create a 4x4 sparse matrix in COO format
row = np.array([0, 0, 2, 2, 3])
col = np.array([0, 2, 0, 1, 3])
data = np.array([1, 2, 3, 4, 5])
A = coo_matrix((data, (row, col)), shape=(4, 4))
print("COO Matrix:\n", A.todense())

# Convert to CSR format
A_csr = A.tocsr()
print("\nCSR Matrix:\n", A_csr.todense())


# Convert to lil format
A_lil = A.tolil()
print("\nLIL Matrix:\n", A_lil.todense())

print(A_lil[0,:])
print(A_lil[1,:])
print(A_lil[2,:])
