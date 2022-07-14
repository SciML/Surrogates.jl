## GEKPLS Surrogate Tutorial

Gradient Enhanced Kriging with Partial Least Squares Method (GEKPLS) is a surrogate modelling technique that brings down computation time and returns improved accuracy for high-dimensional problems. The Julia implementation of GEKPLS is adapted from the Python version by [SMT](https://github.com/SMTorg) which is based on this [paper](https://arxiv.org/pdf/1708.02663.pdf).  

The following are the inputs when building a GEKPLS surrogate: 

1. X - The matrix containing the training points
2. y - The vector containing the training outputs associated with each of the training points
3. grads - The gradients at each of the input X training points
4. n_comp - Number of components to retain for the partial least squares regression (PLS)
5. delta_x -  The step size to use for the first order Taylor approximation
6. xlimits - The lower and upper bounds for the training points
7. extra_points - The number of additional points to use for the PLS 
8. theta - The hyperparameter to use for the correlation model

The following example illustrates how to use GEKPLS:

```@example gekpls_water_flow

using Surrogates
using Zygote

function vector_of_tuples_to_matrix(v)
    #helper function to convert training data generated by surrogate sampling into a matrix suitable for GEKPLS
    num_rows = length(v)
    num_cols = length(first(v))
    K = zeros(num_rows, num_cols)
    for row in 1:num_rows
        for col in 1:num_cols
            K[row, col]=v[row][col]
        end
    end
    return K
end

function vector_of_tuples_to_matrix2(v)
    #helper function to convert gradients into matrix form
    num_rows = length(v)
    num_cols = length(first(first(v)))
    K = zeros(num_rows, num_cols)
    for row in 1:num_rows
        for col in 1:num_cols
            K[row, col] = v[row][1][col]
        end
    end
    return K
end

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
d = 8
lb = [0.05,100,63070,990,63.1,700,1120,9855]
ub = [0.15,50000,115600,1110,116,820,1680,12045]
x = sample(n,lb,ub,SobolSample())
X = vector_of_tuples_to_matrix(x)
grads = vector_of_tuples_to_matrix2(gradient.(water_flow, x))
y = reshape(water_flow.(x),(size(x,1),1))
xlimits = hcat(lb, ub)
n_test = 100 
x_test = sample(n_test,lb,ub,GoldenSample()) 
X_test = vector_of_tuples_to_matrix(x_test) 
y_true = water_flow.(x_test)
n_comp = 2
delta_x = 0.0001
extra_points = 2
initial_theta = [0.01 for i in 1:n_comp]
g = GEKPLS(X, y, grads, n_comp, delta_x, xlimits, extra_points, initial_theta)
y_pred = g(X_test)
rmse = sqrt(sum(((y_pred - y_true).^2)/n_test)) #root mean squared error
println(rmse) #0.0347

```
