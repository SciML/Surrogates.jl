## Gradient Enhanced Kriging

Gradient-enhanced Kriging is an extension of kriging which supports gradient information. GEK is usually more accurate than kriging, however, it is not computationally efficient when the number of inputs, the number of sampling points, or both, are high. This is mainly due to the size of the corresponding correlation matrix that increases proportionally with both the number of inputs and the number of sampling points.

Let's have a look to the following function to use Gradient Enhanced Surrogate:
``f(x) = sin(x) + 2*x^2``

First of all, we will import `Surrogates` and `Plots` packages:

```@example GEK1D
using Surrogates
using Plots
default()
```

### Sampling

We choose to sample f in 8 points between 0 to 1 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example GEK1D
n_samples = 10
lower_bound = 2
upper_bound = 10
xs = lower_bound:0.001:upper_bound
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
f(x) = x^3 - 6x^2 + 4x + 12
y1 = f.(x)
der = x -> 3*x^2 - 12*x + 4
y2 = der.(x)
y = vcat(y1,y2)
scatter(x, y1, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function", xlims=(lower_bound, upper_bound), legend=:top)
```

### Building a surrogate

With our sampled points we can build the Gradient Enhanced Kriging surrogate using the `GEK` function.

```@example GEK1D
my_gek = GEK(x, y, lower_bound, upper_bound, p = 2.9);
```
```@example @GEK1D
plot(x, y1, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function",  xlims=(lower_bound, upper_bound), legend=:top)
plot!(my_gek, label="Surrogate function", ribbon=p->std_error_at_point(my_gek, p), xlims=(lower_bound, upper_bound), legend=:top)
```


## Gradient Enhanced Kriging Surrogate Tutorial (ND)

First of all let's define the function we are going to build a surrogate for.

```@example GEK_ND
using Plots # hide
default(c=:matter, legend=false, xlabel="x", ylabel="y") # hide
using Surrogates # hide
```

Now, let's define the function:

```@example GEK_ND
function shubert(x)
      x1 = x[1]
      x2 = x[2]
      sum1 = 0;
      sum2 = 0;
        for ii in 1:5
	    new1 = ii * cos((ii+1)*x1+ii);
	    new2 = ii * cos((ii+1)*x2+ii);
	    sum1 = sum1 + new1;
	    sum2 = sum2 + new2;
        end
      y = sum1 * sum2;
end
```

### Sampling

Let's define our bounds, this time we are working in two dimensions. In particular we want our first dimension `x` to have bounds `-10, 5.12`, and `-5.12, 10` for the second dimension. We are taking 80 samples of the space using Sobol Sequences. We then evaluate our function on all of the sampling points.

```@example GEK_ND
n_samples = 80
lower_bound = [-10, -5.12]
upper_bound = [5.12, 10.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = shubert.(xys);
```

```@example GEK_ND
x, y = -10:5.12, -5.12:10 # hide
p1 = surface(x, y, (x1,x2) -> shubert((x1,x2))) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
scatter!(xs, ys, zs) # hide
p2 = contour(x, y, (x1,x2) -> shubert((x1,x2))) # hide
scatter!(xs, ys) # hide
plot(p1, p2, title="True function") # hide
```

### Building a surrogate
Using the sampled points we build the surrogate, the steps are analogous to the 1-dimensional case.

```@example GEK_ND
my_GEK = GEK(xys, zs, lower_bound, upper_bound, p=[1.9, 1.9])
```

```@example GEK_ND
p1 = surface(x, y, (x, y) -> my_GEK([x y])) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> my_GEK([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2, title="Surrogate") # hide
```
