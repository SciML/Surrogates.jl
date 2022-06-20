## Radial Surrogates
The Radial Basis Surrogate model represents the interpolating function as a linear combination of basis functions, one for each training point. Let's start with something easy to get our hands dirty. I want to build a surrogate for:

`f(x) = log(x)*x^2+x^3``

Let's choose the Radial Basis Surrogate for 1D. First of all we have to import these two packages: `Surrogates` and `Plots`,

```@example RadialBasisSurrogate
using Surrogates
using Plots
default()
```

We choose to sample f in 30 points between 5 to 25 using `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example RadialBasisSurrogate
f(x) = log(x)*x^2 + x^3
n_samples = 30
lower_bound = 5
upper_bound = 25
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)
scatter(x, y, label="Sampled Points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function", scatter(x, y, label="Sampled Points", xlims=(lower_bound, upper_bound), legend=:top)
```


## Building Surrogate

With our sampled points we can build the **Radial Surrogate** using the `RadialBasis` function.

We can simply calculate `radial_surrogate` for any value.

```@example LinearRadialBasisSurrogate
radial_surrogate = RadialBasis(x, y, lower_bound, upper_bound)
val = radial_surrogate(5.4)
```

We can also use cubic radial basis functions.

```@example CubicRadialBasisSurrogate
radial_surrogate = RadialBasis(x, y, lower_bound, upper_bound, cubicRadial)
val = radial_surrogate(5.4)
```

Currently available radial basis functions are `linearRadial` (the default), `cubicRadial`, `multiquadricRadial`, and `thinplateRadial`.

Now, we will simply plot `radial_surrogate`:

```@example RadialBasisSurrogate
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function",  xlims=(lower_bound, upper_bound), legend=:top)
plot!(radial_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound), legend=:top)
```


## Optimizing

Having built a surrogate, we can now use it to search for minima in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example RadialBasisSurrogate
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, radial_surrogate, SobolSample())
scatter(x, y, label="Sampled points", legend=:top)
plot!(f, label="True function",  xlims=(lower_bound, upper_bound), legend=:top)
plot!(radial_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound), legend=:top)
```


## Radial Basis Surrogate tutorial (ND)

First of all we will define the `Booth` function we are going to build surrogate for:

$f(x) = (x_1 + 2*x_2 - 7)^2 + (2*x_1 + x_2 - 5)^2$

 Notice, one how its argument is a vector of numbers, one for each coordinate, and its output is a scalar.

```@example RadialBasisSurrogateND
using Plots # hide
default(c=:matter, legend=false, xlabel="x", ylabel="y") # hide
using Surrogates # hide

function booth(x)
    x1=x[1]
    x2=x[2]
    term1 = (x1 + 2*x2 - 7)^2;
    term2 = (2*x1 + x2 - 5)^2;
    y = term1 + term2;
end
```

### Sampling

Let's define our bounds, this time we are working in two dimensions. In particular we want our first dimension `x` to have bounds `-5, 10`, and `0, 15` for the second dimension. We are taking 80 samples of the space using Sobol Sequences. We then evaluate our function on all of the sampling points.

```@example RadialBasisSurrogateND
n_samples = 80
lower_bound = [-5.0, 0.0]
upper_bound = [10.0, 15.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = booth.(xys);
```

```@example RadialBasisSurrogateND
x, y = -5:10, 0:15 # hide
p1 = surface(x, y, (x1,x2) -> booth((x1,x2))) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
scatter!(xs, ys, zs) # hide
p2 = contour(x, y, (x1,x2) -> booth((x1,x2))) # hide
scatter!(xs, ys) # hide
plot(p1, p2, title="True function") # hide
```

### Building a surrogate
Using the sampled points we build the surrogate, the steps are analogous to the 1-dimensional case.

```@example RadialBasisSurrogateND
radial_basis = RadialBasis(xys, zs,  lower_bound, upper_bound)
```

```@example RadialBasisSurrogateND
p1 = surface(x, y, (x, y) -> radial_basis([x y])) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> radial_basis([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2, title="Surrogate") # hide
```

### Optimizing
With our surrogate we can now search for the minima of the function.

Notice how the new sampled points, which were created during the optimization process, are appended to the `xys` array.
This is why its size changes.

```@example RadialBasisSurrogateND
size(xys)
```
```@example RadialBasisSurrogateND
surrogate_optimize(booth, SRBF(), lower_bound, upper_bound, radial_basis, UniformSample(), maxiters=50)
```
```@example RadialBasisSurrogateND
size(xys)
```

```@example RadialBasisSurrogateND
p1 = surface(x, y, (x, y) -> radial_basis([x y])) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
zs = booth.(xys) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> radial_basis([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2) # hide
```
