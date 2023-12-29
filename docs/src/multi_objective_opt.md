# Multi-objective optimization 

## Case 1: Non-colliding objective functions

```@example multi_obj
using Surrogates

m = 10
f  = x -> [x^i for i = 1:m]
lb = 1.0
ub = 10.0
x  = sample(50, lb, ub, GoldenSample())
y  = f.(x)
my_radial_basis_ego = RadialBasis(x, y, lb, ub)
pareto_set, pareto_front = surrogate_optimize(f, SMB(),lb,ub,my_radial_basis_ego,SobolSample(); maxiters = 10, n_new_look = 100)

m = 5
f  = x -> [x^i for i =1:m]
lb = 1.0
ub = 10.0
x  = sample(50, lb, ub, SobolSample())
y  = f.(x)
my_radial_basis_rtea = RadialBasis(x, y, lb, ub)
Z = 0.8
K = 2
p_cross = 0.5
n_c = 1.0
sigma = 1.5
surrogate_optimize(f,RTEA(Z,K,p_cross,n_c,sigma),lb,ub,my_radial_basis_rtea,SobolSample())

```

## Case 2: objective functions with conflicting minima

```@example multi_obj

f  = x -> [sqrt((x[1] - 4)^2 + 25*(x[2])^2),
           sqrt((x[1]+4)^2 + 25*(x[2])^2),
           sqrt((x[1]-3)^2 + (x[2]-1)^2)]
lb = [2.5,-0.5]
ub = [3.5,0.5]
x  = sample(50, lb, ub, SobolSample())
y  = f.(x)
my_radial_basis_ego = RadialBasis(x, y, lb, ub)
#I can find my pareto set and pareto front by calling again the surrogate_optimize function:
pareto_set, pareto_front = surrogate_optimize(f,SMB(),lb,ub,my_radial_basis_ego, SobolSample(); maxiters = 10, n_new_look = 100);
```
