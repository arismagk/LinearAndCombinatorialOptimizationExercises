from scipy.optimize import linprog

#obj holds the coefficients from the objective function.
#lhs_ineq holds the left-side coefficients from the inequality (red, blue, and yellow) constraints.
#rhs_ineq holds the right-side coefficients from the inequality (red, blue, and yellow) constraints.

obj = [3, 1] # Objective function coefficients 3x1 + x2

lhs_ineq = [[-6,  -3],  # -6x1 - 3x2 <= -12  red left side
            [-4,  -8],  # -4x1 - 8x2 <= -16  green left side
            [6,    5],  #  6x1 + 5x2 <=  30  yellow left side
            [6,    7]]  #  6x1 + 7x2 <=  36  orange left side

rhs_ineq = [-12,   # -6x1 - 3x2 <= -12  red    right side
            -16,   # -4x1 - 8x2 <= -16  green  right side
            30,    #  6x1 + 5x2 <=  30  yellow right side
            36]    #  6x1 + 7x2 <=  36  orange right side

opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, method="revised simplex")

print(opt)

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

# Create the model
model = LpProblem(name="exercise1", sense=LpMaximize)

# Initialize the decision variables
x1 = LpVariable(name="x1", lowBound=0, cat = "Integer")
x2 = LpVariable(name="x2", lowBound=0, cat = "Integer")

# Add the objective function to the model
obj_func = 3 * x1 + x2
model += obj_func

# Add the constraints to the model
model += (6 * x1 + 3 * x2 >= 12, "red_constraint")
model += (4 * x1 + 8 * x2 >= 16, "green_constraint")
model += (6 * x1 + 5 * x2 <= 30, "yellow_constraint")
model += (6 * x1 + 7 * x2 <= 36, "orange_constraint")

print(model)
status = model.solve()
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")
for var in model.variables():
    print(f"{var.name}: {var.value()}")

import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol

import numpy as np
import matplotlib.pyplot as plt

d = np.linspace(-2, 7, 300)
x1, x2 = np.meshgrid(d, d)
plt.imshow(((x1 >= 0) & (x2 >= 0) & (6 * x1 + 3* x2 >= 12) & (4 * x1 + 8 * x2 >= 16) & (6 * x1 + 5 * x2 <= 30) & (6 * x1 + 7 * x2 <= 36)).astype(int) ,
                extent=(x1.min(),x1.max(),x2.min(),x2.max()),origin="lower", cmap="Greys", alpha = 0.3);

x1 = np.linspace(0, 7, 2000)
# x2 >= -6x1/3 +12/3
x2 = -2 * x1 + 4
# x2 >= 16/8 -4x1/8
x3 = (16 - 4* x1) / 8
# 6x1 + 5x2 <= 30
x4 = (30 - 6 * x1) / 5
# 6x1 + 7x2 <=36
x5 = (36 - 6 * x1) / 7
# objective function wih calculated maximum value: 3x1 + x2 = 15
x6 = 15 - 3 * x1

# objective function wih  less than the calculated maximum value: 3x1 + x2 = 13
x7 = 13 - 3 * x1

plt.plot(x1, x2, label=r'$6x1 +3x2\geq12$')
plt.plot(x1, x3, label=r'$4x1 +8x2\geq16$')
plt.plot(x1, x4, label=r'$6x1 +5x2\leq30$')
plt.plot(x1, x5, label=r'$6x1 +7x2\leq36$')
plt.plot(x1, x6, label=r'$3x1 + x2 = 15$')
plt.plot(x1, x7, label=r'$3x1 + x2 = 13$')
plt.xlim(0,8)
plt.ylim(0,10)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel(r'$x1$')
plt.ylabel(r'$x2$')

plt.show()
