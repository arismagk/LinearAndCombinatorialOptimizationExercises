import numpy as np
from scipy import linalg
import sympy
import CalculateBasis as cb


A = np.array([[1, 0, 0, 1, 0, 0],
              [4, 1, 0, 0, 1, 0],
              [8, 4, 1, 0, 0, 1]])

b = np.array([5, 25, 125])
c = np.array([4, 2, 1])

list_of_linear_independent_columns = cb.Calculator.CalculateAllBasisOfMatrix(A)
#print(list_of_linear_independent_columns)
A_transpose = A.transpose()

bases = []
for i in list_of_linear_independent_columns:
    #print(A_transpose[i])
    bases.append(A_transpose[i].transpose())

for base in bases:
    b_inv = linalg.inv(base)
    print("Solution for base ", base, "is: ")
    print(b_inv.dot(b))  # if >= 0 we have solution










