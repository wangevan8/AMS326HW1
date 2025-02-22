import numpy as np
import random
import time

def f(x):
    #The function f(x) = e^(-x³) - x⁴ - sin(x)
    return np.exp(-x**3) - x**4 - np.sin(x)

def df(x):
    #The derivative of f(x)
    return -3 * x**2 * np.exp(-x**3) - 4 * x**3 - np.cos(x)

def count_operations(func, *args, **kwargs):
    #Count floating-point operations (FLOPs)
    global op_count
    op_count = 0
    
    # Function to track operations
    def flop_f(x):
        global op_count
        # Count operations in f(x) = e^(-x³) - x⁴ - sin(x)
        # e^(-x³): 1 cube + 1 negation + 1 exponent = 3 ops
        # x⁴ = 1 operation
        # sin(x) = 1 operation
        # 2 subtractions = 2 operations
        op_count += 7
        return np.exp(-x**3) - x**4 - np.sin(x)
    
    def flop_df(x):
        global op_count
        # Count operations in df(x) = -3x²e^(-x³) - 4x³ - cos(x)
        # -3x²: 1 square + 1 multiplication + 1 negation = 3 ops
        # e^(-x³): 1 cube + 1 negation + 1 exponent = 3 ops
        # -3x²e^(-x³): 1 multiplication = 1 op
        # 4x³: 1 cubing + 1 multiplication = 2 ops
        # cos(x): 1 operation
        # 2 subtractions: 2 operations
        op_count += 12
        return -3 * x**2 * np.exp(-x**3) - 4 * x**3 - np.cos(x)
    
    # Replace the original function with version that has FLOP tracking based on function name
    if func.__name__ == 'bisection_method':
        result = func(flop_f, *args, **kwargs)
    # Newton needs both f(x) and df(x)
    elif func.__name__ == 'newton_method':
        result = func(flop_f, flop_df, *args, **kwargs)
    elif func.__name__ == 'secant_method':
        result = func(flop_f, *args, **kwargs)
    elif func.__name__ == 'monte_carlo_method':
        result = func(flop_f, *args, **kwargs)
    else:
        result = func(*args, **kwargs)
        
    return result, op_count

# Method 1: Bisection Method
# Modified code from GeeksforGeeks
def bisection_method(f, a=-1, b=1, tol=0.5e-4, max_iter=1000):
    if f(a) * f(b) >= 0: # Error check
        raise ValueError("Function values at interval endpoints must have opposite signs")
    
    iteration = 0
    r_true = 0.641583  # True root
    
    while iteration < max_iter:
        c = (a + b) / 2  # Midpoint
        fc = f(c)
        
        # Check if within range of true root equation
        if abs(c - r_true) < tol:
            return c, iteration + 1
        
        if fc == 0:  # Exact root
            return c, iteration + 1
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
            
        iteration += 1
    
    return c, iteration

# Method 2: Newton's Method
# Modified code from GeeksforGeeks
def newton_method(f, df, x0=0, tol=0.5e-4, max_iter=1000):
    x = x0
    iteration = 0
    r_true = 0.641583  
    
    while iteration < max_iter:
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-10:  # Error check
            raise ValueError("Derivative too close to zero")
        
        x_new = x - fx / dfx
        
        # Check if within range of true root equation
        if abs(x_new - r_true) < tol:
            return x_new, iteration + 1
        
        x = x_new
        iteration += 1
    
    return x, iteration

# Method 3: Secant Method
# Modified code from GeeksforGeeks
def secant_method(f, x0=-1, x1=1, tol=0.5e-4, max_iter=1000):
    x_prev = x0
    x = x1
    iteration = 0
    r_true = 0.641583
    
    while iteration < max_iter:
        f_prev = f(x_prev)
        fx = f(x)
        
        if abs(fx - f_prev) < 1e-10: 
            raise ValueError("Difference in values too close to zero")
        
        x_new = x - fx * (x - x_prev) / (fx - f_prev)
        
        # Check if within range of true root equation
        if abs(x_new - r_true) < tol:
            return x_new, iteration + 1
        
        x_prev = x
        x = x_new
        iteration += 1
    
    return x, iteration

# Method 4: Monte Carlo Method
# Modified code from GeeksforGeeks
def monte_carlo_method(f, a=0.5, b=0.75, tol=0.5e-4, max_samples=100000):
    samples = 0
    best_x = None
    best_f_value = float('inf')
    r_true = 0.641583
    
    while samples < max_samples:
        set_size = min(1000, max_samples - samples) 
        # Generate random points in the interval
        test_points = [random.uniform(a, b) for _ in range(set_size)]
        
        for x in test_points:
            fx = abs(f(x))  # Want |f(x)| close to 0
            
            if fx < best_f_value:
                best_f_value = fx
                best_x = x
                
                # Check if within range of true root equation
                if abs(best_x - r_true) < tol:
                    return best_x, samples + 1
        
        samples += set_size
    
    return best_x, samples

methods = [bisection_method, newton_method, secant_method, monte_carlo_method]
# For loop to print out results
for method in methods:
    try:
        (root, iterations), ops = count_operations(method)
        print(f"{method.__name__}:")
        print(f"  Root: {root:.6f}")
        print(f"  Iterations: {iterations}")
        print(f"  Floating-point operations: {ops}\n")
    except Exception as e:
        print(f"{method.__name__} failed: {e}\n")
