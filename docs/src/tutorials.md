## Surrogates 101
Let's start with something easy to get our hands dirty.
I want to build a surrogate for ``f(x) = log(x)*x^2+x^3``.
Let's choose the radial basis surrogate.
```
using Surrogates
f = x -> log(x)*x^2+x^3
lb = 1.0
ub = 10.0
x = sample(50,lb,ub,SobolSample())
y = f.(x)
thin_plate_spline = x -> x^2*log(x)
q = 2
my_radial_basis = RadialBasis(x,y,lb,ub,thin_plate_spline,q)

#I want an approximation at 5.4
approx = my_radial_basis(5.4)
```

Let's now see an example in 2D.
```
using Surrogates
f = x -> x[1]*x[2]
lb = [1.0,2.0]
ub = [10,8.5]
x = sample(50,lb,ub,SobolSample())
y = f.(x)
thin_plate_spline = x -> x^2*log(x)
q = 2
my_radial_basis = RadialBasis(x,y,[lb,ub],thin_plate_spline,q)

#I want an approximation at (1.0,1.4)
approx = my_radial_basis((1.0,1.4))
```

## Kriging standard error
Let's now use the Kriging surrogate, which is a single output Gaussian process.
This surrogate has a nice feature:not only it approximates the solution at a
point, it also calculates the standard error at such point.
Let's see an example:
```
using Surrogates
f = x -> exp(x)*x^2+x^3
lb = 0.0
ub = 10.0
x = sample(100,lb,ub,UniformSample())
y = f.(x)
p = 1.9
my_krig = Kriging(x,y,p)

#I want an approximation at 5.4
approx = my_radial_basis(5.4)

#I want to find the standard error at 5.4
std_err = std_error_at_point(my_krig,5.4)
```

Let's now optimize the Kriging surrogate using Lower confidence bound method, this is just a one-liner:
```
surrogate_optimize(f,LCBS(),a,b,my_krig,UniformSample())
```
## Lobachesky integral

The Lobachesky surrogate has the nice feature of having a closed formula for its
integral, which is something that other surrogates are missing.
Let's compare it with QuadGK.
```
using Surrogates
using QuadGK
obj = x -> 3*x + log(x)
a = 1.0
b = 4.0
x = sample(2000,a,b,SobolSample())
y = obj.(x)
alpha = 2.0
n = 6
my_loba = LobacheskySurrogate(x,y,alpha,n,a,b)

#1D integral
int_1D = lobachesky_integral(my_loba,a,b)
int = quadgk(obj,a,b)
int_val_true = int[1]-int[2]
@test abs(int_1D - int_val_true) < 10^-5
```
