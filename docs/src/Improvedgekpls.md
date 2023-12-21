# GEKPLS Function

Gradient Enhanced Kriging with Partial Least Squares Method (GEKPLS) is a surrogate modelling technique that brings down computation time and returns improved accuracy for high-dimensional problems. The Julia implementation of GEKPLS is adapted from the Python version by [SMT](https://github.com/SMTorg) which is based on this [paper](https://arxiv.org/pdf/1708.02663.pdf).  

# Modifications for Improved GEKPLS Function:

To enhance the GEKPLS function, sampling method was changed from ```SobolSample()``` to ```HaltonSample()```.


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
    log_val = log(r/r_w)
    return (2*pi*T_u*(H_u - H_l))/ ( log_val*(1 + (2*L*T_u/(log_val*r_w^2*K_w)) + T_u/T_l))
end

n = 1000
lb = [0.05,100,63070,990,63.1,700,1120,9855]
ub = [0.15,50000,115600,1110,116,820,1680,12045]
x = sample(n,lb,ub,HaltonSample())
grads = gradient.(water_flow, x)
y = water_flow.(x)
n_test = 100 
x_test = sample(n_test,lb,ub,GoldenSample()) 
y_true = water_flow.(x_test)
n_comp = 2
delta_x = 0.0001
extra_points = 2
initial_theta = [0.01 for i in 1:n_comp]
g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
y_pred = g.(x_test)
rmse = sqrt(sum(((y_pred - y_true).^2)/n_test)) #root mean squared error
println(rmse) #0.0347
```

<br>
<br>



| **Sampling Method** | **RMSE**                | **Differences**                                                                                                                                                                                                                                                                                     |
|----------------------|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Sobol Sampling**               | 0.021472963465423097 | Utilizes digital nets to generate quasi-random numbers, offering low-discrepancy points for improved coverage. - Requires careful handling, especially in higher dimensions.                                                                                                                      |
| **Halton Sampling**               | 0.02144270998045834  | Uses a deterministic sequence based on prime numbers to generate points, allowing for quasi-random, low-discrepancy sampling. - Simpler to implement but may exhibit correlations in some dimensions affecting coverage.                                                                               |
