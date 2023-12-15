# Latin Hypercube Sampling (LHS) for Surrogate Modeling in Julia

Latin Hypercube Sampling (LHS) is a method used for generating representative samples across multiple dimensions of an input space. This README demonstrates how to use LHS for surrogate modeling in Julia, specifically replacing Sobol Sampling in the context of GEKPLS (Gradient Enhanced Kriging with Partial Least Squares) surrogate modeling.

### Requirements
- Julia programming language
- LatinHypercubeSampling.jl package

### Installation
Make sure you have the LatinHypercubeSampling.jl package installed in your Julia environment:

```julia
using Pkg
Pkg.add("LatinHypercubeSampling")
```

# Usage Example
Consider a scenario where you want to create a surrogate model using GEKPLS for a given function, **water_flow**, with an input space defined by lower bounds **lb** and upper bounds **ub**. Here's how to use LHS instead of Sobol Sampling:

```
using Surrogates
using LatinHypercubeSampling

# Function definition for water_flow
function water_flow(x)
    r_w = x[1]
    r = x[2]
    T_u = x[3]
    H_u = x[4]
    T_l = x[5]
    H_l = x[6]
    L = x[7]
    K_w = x[8]
    log_val = log(r/r_w)
    return (2*pi*T_u*(H_u - H_l))/ ( log_val*(1 + (2*L*T_u/(log_val*r_w^2*K_w)) + T_u/T_l))
end

n_samples = 1000  # Number of LHS samples
lb = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
ub = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]

# Generating Latin Hypercube Samples
lhs_samples = lhsdesign(n_samples, length(lb))
x = [(ub - lb) .* lhs_samples[i, :] .+ lb for i in 1:n_samples]
```
# Surrogate Optimization

```
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
println(minima)
```


# Conclusion

Latin Hypercube Sampling (LHS) provides a more evenly distributed set of samples across the input space compared to purely random methods. Experiment with different sampling sizes and assess the surrogate model's performance to find the optimal sampling strategy for your problem.

For more information, refer to this link: 
https://github.com/MrUrq/LatinHypercubeSampling.jl


