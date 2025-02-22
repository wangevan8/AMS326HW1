import numpy as np
from scipy.interpolate import lagrange
# Data points
t = np.array([1, 2, 3, 4, 5])
y = np.array([412, 407, 397, 398, 417])

# Code modified from: numpy.org API reference
# (1) Polynomial interpolation using Lagrange method
poly = lagrange(t, y)
P4_at_6 = np.polyval(poly, 6)

# Display the polynomial coefficients and result
print("Polynomial Interpolation:")
print(f"P4(t) coefficients: {poly.coef}")
print(f"P4(6) = {P4_at_6}")

# Code modified from GeeksforGeeks
# (2) Quadratic fit using least squares
quad_coeff = np.polyfit(t, y, 2)  # Fit a degree-2 polynomial
Q2_6 = np.polyval(quad_coeff, 6)

# Display the quadratic fit coefficients and result
print("\nQuadratic Fit:")
print(f"Q2(t) = {quad_coeff[0]:.1f}t² + {quad_coeff[1]:.1f}t + {quad_coeff[2]:.1f}")
print(f"Q2(6) = {Q2_6:.1f}")
