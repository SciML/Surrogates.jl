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
my_radial_basis = RadialBasis(x,y,lb,ub,rad=thinplateRadial)

#I want an approximation at 5.4
approx = my_radial_basis(5.4)
```

- For each Surrogates we can call it with different inputs: either ``(x,y,lb,ub)`` or with it's parameters,
different for each Surrogates. Let's see for Radial Basis Surrogates:

```@example
my_radial_basis = RadialBasis(x,y,lb,ub)

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

Again, Let's choose the Radial Basis Surrogate for ND.

```@example
using Surrogates, LinearAlgebra
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
lb = [0.0,3.0,6.0]
ub = [4.0,7.0,10.0]

#bounds = [[0.0, 3.0, 6.0], [4.0, 7.0, 10.0]]

my_radial_basis = RadialBasis(x, y, lb, ub)

#We want an approximation at (1.0,2.0,3.0)  
est = my_radial_basis((1.0,2.0,3.0))
```
