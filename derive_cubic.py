import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols, solve

a, b, c, d, x, α, β = symbols("a b c d x α β")

# polynomial function f(x) = ax³ + bx² + cx + d
f = a * x ** 3 + b * x ** 2 + c * x + d

fp = f.diff(x)  # derivative f'(x)

# evaluate both at x=0 and x=1
f0, f1 = [f.subs(x, i) for i in range(2)]
fp0, fp1 = [fp.subs(x, i) for i in range(2)]

# we want a, b, c, d such that the following conditions hold:
#
#  f(0) = 0
#  f(1) = 0
#  f'(0) = α
#  f'(1) = β

S = solve([f0, f1, fp0 - α, fp1 - β], [a, b, c, d])

# print the analytic solution and plot a graphical example
coeffs = []

num_α = 0.3
num_β = -0.03

for key in [a, b, c, d]:
    print(key, "=", S[key])
    coeffs.append(S[key].subs(dict(α=num_α, β=num_β)))

xvals = np.linspace(0, 1, 101)
yvals = np.polyval(coeffs, xvals)

plt.plot(xvals, yvals)
plt.show()
