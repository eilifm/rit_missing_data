from sympy import *

x1, x2, y, b0, b1, b2, b12 = symbols('x1 x2 y b0 b1 b2 b12')

eq = Eq(b0 + (b1*x1) + (b2*x2) + (b12*x1*x2), y)

expr = solve(eq, x1)[0]

print(expr)

# str1 = "x1:x2"
# str1 = "(" + str1 + ")"
# print(str1.replace(':', '*'))
