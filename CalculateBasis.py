import numpy as np
from scipy import linalg
import sympy

class Calculator:

    def CalculateAllBasisOfMatrix(A):

        columns_list = []
        A = np.array([[1, 0, 0, 1, 0, 0],
                      [4, 1, 0, 0, 1, 0],
                      [8, 4, 1, 0, 0, 1]])

        A_transpose = A.transpose()

        vectors_list = []

        for i in range (0, 6):
            vectors_list.append(A_transpose[i])

        #print(vectors_list)

        counter = 0
        for i in range(0, 6):

            for j in range(i+1, 6):

                for k in range(j+1, 6):
                    local_list = []
                    A = np.row_stack([vectors_list[i], vectors_list[j], vectors_list[k]])
                    U, s, V = np.linalg.svd(A)

                    if 0 not in s:
                        local_list.append(i)
                        local_list.append(j)
                        local_list.append(k)
                        counter = counter + 1
                        columns_list.append(local_list)
        print(counter)

        for i in columns_list:
            if linalg.det(A_transpose[i]) == 0:
                print("Error Found. Det equal to 0")

        return columns_list