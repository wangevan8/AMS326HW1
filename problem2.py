import numpy as np
from scipy.interpolate import lagrange


# Data points
t = np.array([1, 2, 3, 4, 5])
y = np.array([412, 407, 397, 398, 417])

# (1) Polynomial interpolation using Lagrange
poly = lagrange(t, y)
P4_at_6 = np.polyval(poly, 6)

# Results
print("Polynomial Interpolation:")
print(f"P4(t) coefficients: {poly.coef}")
print(f"P4(6) = {P4_at_6}")

# (2) Quadratic fit using least squares
A = np.vstack([np.ones_like(t), t, t**2]).T
a0, a1, a2 = np.linalg.lstsq(A, y, rcond=None)[0]

# Define the quadratic function
def Q2(t):
    return a0 + a1*t + a2*(t**2)

Q2_at_6 = Q2(6)

# Results
print("\nQuadratic Fit:")
print(f"Q2(t) = {a0:.1f} + {a1:.1f}t + {a2:.1f}t²")
print(f"Q2(6) = {Q2_at_6:.1f}")
