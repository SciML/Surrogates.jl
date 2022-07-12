## Kriging surrogate tutorial (1D)

Kriging or Gaussian process regression is a method of interpolation for which the interpolated values are modeled by a Gaussian process.

We are going to use a Kriging surrogate to optimize $f(x)=(6x-2)^2sin(12x-4)$. (function from Forrester et al. (2008)).

First of all import `Surrogates` and `Plots`.
```@example kriging_tutorial1d
using Surrogates
using Plots
default()
```
### Sampling

We choose to sample f in 4 points between 0 and 1 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example kriging_tutorial1d
# https://www.sfu.ca/~ssurjano/forretal08.html
# Forrester et al. (2008) Function
f(x) = (6 * x - 2)^2 * sin(12 * x - 4)

n_samples = 4
lower_bound = 0.0
upper_bound = 1.0

xs = lower_bound:0.001:upper_bound

x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)

scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound), ylims=(-7, 17))
plot!(xs, f.(xs), label="True function", legend=:top)
```
### Building a surrogate

With our sampled points we can build the Kriging surrogate using the `Kriging` function.

`kriging_surrogate` behaves like an ordinary function which we can simply plot. A nice statistical property of this surrogate is being able to calculate the error of the function at each point, we plot this as a confidence interval using the `ribbon` argument.

```@example kriging_tutorial1d
kriging_surrogate = Kriging(x, y, lower_bound, upper_bound);

plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound), ylims=(-7, 17), legend=:top)
plot!(xs, f.(xs), label="True function", legend=:top)
plot!(xs, kriging_surrogate.(xs), label="Surrogate function", ribbon=p->std_error_at_point(kriging_surrogate, p), legend=:top)
```
### Optimizing
Having built a surrogate, we can now use it to search for minima in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example kriging_tutorial1d
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, kriging_surrogate, SobolSample())

scatter(x, y, label="Sampled points", ylims=(-7, 7), legend=:top)
plot!(xs, f.(xs), label="True function", legend=:top)
plot!(xs, kriging_surrogate.(xs), label="Surrogate function", ribbon=p->std_error_at_point(kriging_surrogate, p), legend=:top)
```


## Kriging surrogate tutorial (ND)

First of all let's define the function we are going to build a surrogate for. Notice how its argument is a vector of numbers, one for each coordinate, and its output is a scalar.

```@example kriging_tutorialnd
using Plots # hide
default(c=:matter, legend=false, xlabel="x", ylabel="y") # hide
using Surrogates # hide

function branin(x)
    x1=x[1]
    x2=x[2]
    a=1;
    b=5.1/(4*π^2);
    c=5/π;
    r=6;
    s=10;
    t=1/(8π);
    a*(x2-b*x1+c*x1-r)^2+s*(1-t)*cos(x1)+s
end
```

### Sampling
Let's define our bounds, this time we are working in two dimensions. In particular we want our first dimension `x` to have bounds `-5, 10`, and `0, 15` for the second dimension. We are taking 50 samples of the space using Sobol Sequences. We then evaluate our function on all of the sampling points.

```@example kriging_tutorialnd
n_samples = 50
lower_bound = [-5.0, 0.0]
upper_bound = [10.0, 15.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = branin.(xys);
```

```@example kriging_tutorialnd
x, y = -5:10, 0:15 # hide
p1 = surface(x, y, (x1,x2) -> branin((x1,x2))) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
scatter!(xs, ys, zs) # hide
p2 = contour(x, y, (x1,x2) -> branin((x1,x2))) # hide
scatter!(xs, ys) # hide
plot(p1, p2, title="True function") # hide
```

### Building a surrogate
Using the sampled points we build the surrogate, the steps are analogous to the 1-dimensional case.

```@example kriging_tutorialnd
kriging_surrogate = Kriging(xys, zs, lower_bound, upper_bound, p=[2.0, 2.0], theta=[0.03, 0.003])
```

```@example kriging_tutorialnd
p1 = surface(x, y, (x, y) -> kriging_surrogate([x y])) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> kriging_surrogate([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2, title="Surrogate") # hide
```

### Optimizing
With our surrogate we can now search for the minima of the branin function.

Notice how the new sampled points, which were created during the optimization process, are appended to the `xys` array.
This is why its size changes.

```@example kriging_tutorialnd
size(xys)
```
```@example kriging_tutorialnd
surrogate_optimize(branin, SRBF(), lower_bound, upper_bound, kriging_surrogate, SobolSample(), maxiters=10)
```
```@example kriging_tutorialnd
size(xys)
```

```@example kriging_tutorialnd
p1 = surface(x, y, (x, y) -> kriging_surrogate([x y])) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
zs = branin.(xys) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> kriging_surrogate([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2) # hide
```
