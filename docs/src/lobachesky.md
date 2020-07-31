## Lobachevsky surrogate tutorial

Lobachevsky splines function is a function that used for univariate and multivariate scattered interpolation. Introduced by Lobachevsky in 1842 to investigate errors in astronomical measurements.

We are going to use a Lobachevsky surrogate to optimize $f(x)=sin(x)+sin(10/3 * x)$.

First of all import `Surrogates` and `Plots`.
```@example LobachevskySurrogate_tutorial
using Surrogates
using Plots
default()
```
### Sampling

We choose to sample f in 4 points between 0 and 4 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example LobachevskySurrogate_tutorial
f(x) = sin(x) + sin(10/3 * x)
n_samples = 5
lower_bound = 1.0
upper_bound = 4.0
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function", xlims=(lower_bound, upper_bound))
```
### Building a surrogate

With our sampled points we can build the Lobachevsky surrogate using the `LobachevskySurrogate` function.

`lobachevsky_surrogate` behaves like an ordinary function which we can simply plot. Alpha is the shape parameters and n specify how close you want lobachevsky function to radial basis function.

```@example LobachevskySurrogate_tutorial
alpha = 2.0
n = 6
lobachevsky_surrogate = LobacheskySurrogate(x, y, lower_bound, upper_bound, alpha = 2.0, n = 6)
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(lobachevsky_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```
### Optimizing
Having built a surrogate, we can now use it to search for minimas in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example LobachevskySurrogate_tutorial
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, lobachevsky_surrogate, SobolSample())
scatter(x, y, label="Sampled points")
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(lobachevsky_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```


In the example below, it shows how to use `lobachevsky_surrogate` for higher dimension problems.

## Lobachevsky Surrogate Tutorial (ND):

First of all we will define the `Schaffer` function we are going to build surrogate for. Notice, one how its argument is a vector of numbers, one for each coordinate, and its output is a scalar.

```@example LobachevskySurrogate_ND
using Plots # hide
default(c=:matter, legend=false, xlabel="x", ylabel="y") # hide
using Surrogates # hide

function schaffer(x)
    x1=x[1]
    x2=x[2]
    fact1 = x1 ^2;
    fact2 = x2 ^2;
    y = fact1 + fact2;
end
```


### Sampling

Let's define our bounds, this time we are working in two dimensions. In particular we want our first dimension `x` to have bounds `0, 8`, and `0, 8` for the second dimension. We are taking 60 samples of the space using Sobol Sequences. We then evaluate our function on all of the sampling points.

```@example LobachevskySurrogate_ND
n_samples = 60
lower_bound = [0.0, 0.0]
upper_bound = [8.0, 8.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = schaffer.(xys);
```

```@example LobachevskySurrogate_ND
x, y = 0:8, 0:8 # hide
p1 = surface(x, y, (x1,x2) -> schaffer((x1,x2))) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
scatter!(xs, ys, zs) # hide
p2 = contour(x, y, (x1,x2) -> schaffer((x1,x2))) # hide
scatter!(xs, ys) # hide
plot(p1, p2, title="True function") # hide
```


### Building a surrogate
Using the sampled points we build the surrogate, the steps are analogous to the 1-dimensional case.

```@example LobachevskySurrogate_ND
Lobachesky = LobacheskySurrogate(xys, zs,  lower_bound, upper_bound, alpha = [2.4,2.4], n=8)
```

```@example LobachevskySurrogate_ND
p1 = surface(x, y, (x, y) -> Lobachesky([x y])) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> Lobachesky([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2, title="Surrogate") # hide
```


### Optimizing
With our surrogate we can now search for the minimas of the function.

Notice how the new sampled points, which were created during the optimization process, are appended to the `xys` array.
This is why its size changes.

```@example LobachevskySurrogate_ND
size(Lobachesky.x)
```
```@example LobachevskySurrogate_ND
surrogate_optimize(schaffer, SRBF(), lower_bound, upper_bound, Lobachesky, SobolSample(), maxiters=1, num_new_samples=10)
```
```@example LobachevskySurrogate_ND
size(Lobachesky.x)
```

```@example LobachevskySurrogate_ND
p1 = surface(x, y, (x, y) -> Lobachesky([x y])) # hide
xys = Lobachesky.x # hide
xs = [i[1] for i in xys] # hide
ys = [i[2] for i in xys] # hide
zs = schaffer.(xys) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> Lobachesky([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2) # hide
```
