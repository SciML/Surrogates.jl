## Radial Surrogates
Let's start with something easy to get our hands dirty.
I want to build a surrogate for ``f(x) = log(x)*x^2+x^3``.
Let's choose the Radial Basis Surrogate for 1D.

```@example
using Surrogates
f = x -> log(x)*x^2+x^3
lb = 1.0
ub = 10.0
x = sample(50,lb,ub,SobolSample())
y = f.(x)
my_radial_basis = RadialBasis(x,y,lb,ub)

#I want an approximation at 5.4
approx = my_radial_basis(5.4)
```

- For each Surrogates we can call it with different inputs: either ``(x,y,lb,ub)`` or with it's parameters,
different for each Surrogates. Let's see for Radial Basis Surrogates:

```@example
my_radial_basis = RadialBasis(x,y,lb,ub,rad=thinplateRadial)

#We want an approximation at 5.4
approx = my_radial_basis(5.4)
```

Now, Let's choose the Radial Basis Surrogate for 2D.

```@example
using Surrogates
using LinearAlgebra
f = x -> x[1]*x[2]
lb = [1.0,2.0]
ub = [10.0,8.5]
x = sample(50,lb,ub,SobolSample())
y = f.(x)
my_radial_basis = RadialBasis(x,y,lb,ub)

#I want an approximation at (1.0,1.4)
approx = my_radial_basis((1.0,1.4))
```


- Now we will call an Optimization Method for RadialBasis Surrogates in 1D and ND.
Let's see an Optimization method for 1D:

```@example
using Surrogates, LinearAlgebra, Flux
using Flux: @epochs
##### For 1D #####
lb = 0.0
ub = 15.0
objective_function = x -> 2*x+1
x = [2.5,4.0,6.0]
y = [6.0,9.0,13.0]

# In 1D values of p closer to 2 make the det(R) closer and closer to 0,
#this does not happen in higher dimensions because p would be a vector and not
#all components are generally C^inf
p = 1.99
a = 2
b = 6

my_rad_SRBF1 = RadialBasis(x,y,a,b,rad = linearRadial)
surrogate_optimize(objective_function,SRBF(),a,b,my_rad_SRBF1,UniformSample())
```

Now, let's see an optimization method for ND:

```@example
using Surrogates, LinearAlgebra, Flux
using Flux: @epochs
##### For ND #####
objective_function_ND = z -> 3*norm(z)+1
lb = [1.0,1.0]
ub = [6.0,6.0]
x = sample(5,lb,ub,SobolSample())
y = objective_function_ND.(x)
p = [1.5,1.5]
theta = [1.0,1.0]

x = sample(5,lb,ub,SobolSample())
objective_function_ND = z -> 3*norm(z)+1
y = objective_function_ND.(x)
my_rad_SRBFN = RadialBasis(x,y,lb,ub,rad = linearRadial)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_rad_SRBFN,UniformSample())
```
