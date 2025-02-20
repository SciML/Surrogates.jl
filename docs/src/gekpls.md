# GEKPLS Surrogate Tutorial

Gradient Enhanced Kriging with Partial Least Squares Method (GEKPLS) is a surrogate modeling technique that brings down computation time and returns improved accuracy for high-dimensional problems. The Julia implementation of GEKPLS is adapted from the Python version by [SMT](https://github.com/SMTorg) which is based on this [paper](https://arxiv.org/pdf/1708.02663.pdf).

The following are the inputs when building a GEKPLS surrogate:

 1. x - The vector containing the training points
 2. y - The vector containing the training outputs associated with each of the training points
 3. grads - The gradients at each of the input X training points
 4. n_comp - Number of components to retain for the partial least squares regression (PLS)
 5. delta_x -  The step size to use for the first order Taylor approximation
 6. lb - The lower bound for the training points
 7. ub - The upper bound for the training points
 8. extra_points - The number of additional points to use for the PLS
 9. theta - The hyperparameter to use for the correlation model

## Basic GEKPLS Usage

The following example illustrates how to use GEKPLS:

```@example gekpls_water_flow
using Surrogates
using Zygote

function water_flow(x)
    r_w = x[1]
    r = x[2]
    T_u = x[3]
    H_u = x[4]
    T_l = x[5]
    H_l = x[6]
    L = x[7]
    K_w = x[8]
    log_val = log(r / r_w)
    return (2 * pi * T_u * (H_u - H_l)) /
           (log_val * (1 + (2 * L * T_u / (log_val * r_w^2 * K_w)) + T_u / T_l))
end

n = 1000
lb = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
ub = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]
x = sample(n, lb, ub, SobolSample())
grads = gradient.(water_flow, x)
y = water_flow.(x)
n_test = 100
x_test = sample(n_test, lb, ub, GoldenSample())
y_true = water_flow.(x_test)
n_comp = 2
delta_x = 0.0001
extra_points = 2
initial_theta = [0.01 for i in 1:n_comp]
g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
y_pred = g.(x_test)
rmse = sqrt(sum(((y_pred - y_true) .^ 2) / n_test))
```

## Using GEKPLS With Surrogate Optimization

GEKPLS can also be used to find the minimum of a function with the optimization function.
This next example demonstrates how this can be accomplished.

```@example gekpls_optimization
using Surrogates
using Zygote

function sphere_function(x)
    return sum(x .^ 2)
end

lb = [-5.0, -5.0, -5.0]
ub = [5.0, 5.0, 5.0]
n_comp = 2
delta_x = 0.0001
extra_points = 2
initial_theta = [0.01 for i in 1:n_comp]
n = 100
x = sample(n, lb, ub, SobolSample())
grads = gradient.(sphere_function, x)
y = sphere_function.(x)
g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
x_point, minima = surrogate_optimize(sphere_function, SRBF(), lb, ub, g,
    RandomSample(); maxiters = 20,
    num_new_samples = 20, needs_gradient = true)
minima
```
