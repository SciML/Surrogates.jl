# Surrogates 101

Let's start with something easy to get our hands dirty. Let's say we want to build a surrogate for ``f(x) = \log(x) \cdot x^2+x^3``.

## RBF

We will first use the radial basis surrogate for demonstrations.

```@example
# Importing the package
using Surrogates

# Defining the function
f = x -> log(x) * x^2 + x^3

# Sampling points from the function
lb = 1.0
ub = 10.0
x = sample(50, lb, ub, SobolSample())
y = f.(x)

# Constructing the surrogate
my_radial_basis = RadialBasis(x, y, lb, ub)

# Predicting at x=5.4
approx = my_radial_basis(5.4)
```

We can plot to see how well the surrogate performs compared to the true function.

```@example
using Plots

plot(x, y, seriestype = :scatter, label = "Sampled points",
    xlims = (lb, ub), legend = :top)
xs = 1.0:0.001:10.0
plot!(xs, f.(xs), label = "True function", legend = :top)
plot!(xs, my_radial_basis.(xs); label = "RBF", legend = :top)
```

It fits quite well! Now, let's now see an example in 2D.

```@example
using Surrogates
using LinearAlgebra

f = x -> x[1] * x[2]

lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(50, lb, ub, SobolSample())
y = f.(x)

my_radial_basis = RadialBasis(x, y, lb, ub)

# Predicting at x=(1.0,1.4)
approx = my_radial_basis((1.0, 1.4))
```

## Kriging

Let's now use the Kriging surrogate, which is a single-output Gaussian process. This surrogate has a nice feature - not only does it approximate the solution at a point, it also calculates the standard error at such a point.

```@example kriging
using Surrogates

f = x -> exp(-x) * x^2 + x^3

lb = 0.0
ub = 10.0
x = sample(50, lb, ub, RandomSample())
y = f.(x)

p = 1.9
my_krig = Kriging(x, y, lb, ub, p = p)

# Predicting at x=5.4
approx = my_krig(5.4)

# Predicting error at x=5.4
std_err = std_error_at_point(my_krig, 5.4)
```

Let's now optimize the Kriging surrogate using the lower confidence bound method. This is just a one-liner:

```@example kriging
surrogate_optimize(
    f, LCBS(), lb, ub, my_krig, RandomSample(); maxiters = 10, num_new_samples = 10)
```

Surrogate optimization methods have two purposes: they both sample the space in unknown regions and look for the minima at the same time.

## Lobachevsky integral

The Lobachevsky surrogate has the nice feature of having a closed formula for its integral, which is something that other surrogates are missing. Let's compare it with QuadGK:

```@example loba
using Surrogates
using QuadGK

obj = x -> 3 * x + log(x)
a = 1.0
b = 4.0
x = sample(2000, a, b, SobolSample())
y = obj.(x)
alpha = 2.0
n = 6
my_loba = LobachevskySurrogate(x, y, a, b, alpha = alpha, n = n)

#1D integral
int_1D = lobachevsky_integral(my_loba, a, b)
int = quadgk(obj, a, b)
int_val_true = int[1] - int[2]
println(int_1D)
println(int_val_true)
```

## NeuralSurrogate

Basic example of fitting a neural network on a simple function of two variables.

```@example nns
using Surrogates
using Flux
using Statistics
using SurrogatesFlux

f = x -> x[1]^2 + x[2]^2
# Flux models are in single precision by default.
# Thus, single precision will also be used here for our training samples.
bounds = Float32[-1.0, -1.0], Float32[1.0, 1.0]

x_train = sample(500, bounds..., SobolSample())
y_train = f.(x_train)

# Perceptron with one hidden layer of 20 neurons.
model = Chain(Dense(2, 20, relu), Dense(20, 1))

# Training of the neural network
learning_rate = 0.1
optimizer = Descent(learning_rate)  # Simple gradient descent. See Flux documentation for other options.
n_epochs = 50
sgt = NeuralSurrogate(x_train, y_train, bounds..., model = model,
    opt = optimizer, n_epochs = n_epochs)

# Testing the new model
x_test = sample(30, bounds..., RandomSample())
test_error = mean(abs2, sgt(x)[1] - f(x) for x in x_test)
```
