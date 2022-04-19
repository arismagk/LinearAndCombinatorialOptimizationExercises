import numpy as np
from itertools import combinations
from scipy import linalg
from math import comb
import pandas as pd

As = np.array([[1, -2, 3],
              [5, 1, 2],
              [1, 1, 1],
              [2, -1, 1],
              [3, -1, 1]])
bs = np.array([5, 25, 125])
cs = np.array([4, 2, 1])

A = np.array([[1, 0, 0, 1, 0, 0],
              [4, 1, 0, 0, 1, 0],
              [8, 4, 1, 0, 0, 1]])

b = np.array([5, 25, 125])
c = np.array([4, 2, 1, 0, 0, 0])

m, n = A.shape
print(n,m)
print(A)
print(b)
print(c)
print("\n")
feas = 0
nfeas = 0
ninv = 0
zl = [["x1", "x2", "x3", "z"]]
zm = []
for e in combinations(range(n), 3):
    print(e)
    B = A[:, e]
    det = linalg.det(B)
    if det != 0:
        x = np.linalg.solve(B, b)
        s = np.zeros(n)
        for i in range(len(e)):
            s[e[i]] = x[i]
        print(s)
        if np.any(x[1:] < 0):
            print("Solution non feasible")
            z = c.dot(s)
            print(z)
            zm.append(z)
            nfeas += 1
        else:
            feas += 1
            z = c.dot(s)
            zl.append([s[0], s[1], s[2], z])
            print(z)
        print("\n")
    else:
        ninv += 1
        print("Table cannot be inverted\n")

print("There are", n, "variables and", m, "constrains (", n, "choose", m, "=", comb(n, m), "combinations )")
print("There are", ninv, "arrays that cannot be inverted,", feas, "feasible solutions and", nfeas, "non feasible")

print(pd.DataFrame(zl[1:], columns=zl[0]))

print(zm)

for e2 in combinations(range(m), 3):
    B = As[e2, :]
    bl=[]
    for i in e2:
        bl.append(bs[i])
    x = np.linalg.solve(B, bl)
    print(e2,"\n",x,"\n",x.dot(cs))
