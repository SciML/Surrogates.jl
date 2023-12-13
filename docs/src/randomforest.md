## Random forests surrogate tutorial

!!! note
    This surrogate requires the 'SurrogatesRandomForest' module which can be added by inputting "]add SurrogatesRandomForest" from the Julia command line. 

Random forests is a supervised learning algorithm that randomly creates and merges multiple decision trees into one forest.

We are going to use a Random forests surrogate to optimize $f(x)=sin(x)+sin(10/3 * x)$.

First of all import `Surrogates` and `Plots`.
```@example RandomForestSurrogate_tutorial
using Surrogates
using SurrogatesRandomForest
using Plots
default()
```
### Sampling

We choose to sample f in 4 points between 0 and 1 using the `sample` function. The sampling points are chosen using a Sobol sequence, this can be done by passing `SobolSample()` to the `sample` function.

```@example RandomForestSurrogate_tutorial
f(x) = sin(x) + sin(10/3 * x)
n_samples = 5
lower_bound = 2.7
upper_bound = 7.5
x = sample(n_samples, lower_bound, upper_bound, SobolSample())
y = f.(x)
scatter(x, y, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function", xlims=(lower_bound, upper_bound), legend=:top)
```
### Building a surrogate

With our sampled points we can build the Random forests surrogate using the `RandomForestSurrogate` function.

`randomforest_surrogate` behaves like an ordinary function which we can simply plot. Additionally you can specify the number of trees created
using the parameter num_round

```@example RandomForestSurrogate_tutorial
num_round = 2
randomforest_surrogate = RandomForestSurrogate(x ,y ,lower_bound, upper_bound, num_round = 2)
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound), legend=:top)
plot!(f, label="True function",  xlims=(lower_bound, upper_bound), legend=:top)
plot!(randomforest_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound), legend=:top)
```
### Optimizing
Having built a surrogate, we can now use it to search for minima in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example RandomForestSurrogate_tutorial
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, randomforest_surrogate, SobolSample())
scatter(x, y, label="Sampled points")
plot!(f, label="True function",  xlims=(lower_bound, upper_bound), legend=:top)
plot!(randomforest_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound), legend=:top)
```


## Random Forest ND

First of all we will define the `Bukin Function N. 6` function we are going to build surrogate for.

```@example RandomForestSurrogateND
using Plots # hide
default(c=:matter, legend=false, xlabel="x", ylabel="y") # hide
using Surrogates # hide

function bukin6(x)
    x1=x[1]
    x2=x[2]
    term1 = 100 * sqrt(abs(x2 - 0.01*x1^2));
    term2 = 0.01 * abs(x1+10);
    y = term1 + term2;
end
```


### Sampling

Let's define our bounds, this time we are working in two dimensions. In particular we want our first dimension `x` to have bounds `-5, 10`, and `0, 15` for the second dimension. We are taking 50 samples of the space using Sobol Sequences. We then evaluate our function on all of the sampling points.

```@example RandomForestSurrogateND
n_samples = 50
lower_bound = [-5.0, 0.0]
upper_bound = [10.0, 15.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = bukin6.(xys);
```

```@example RandomForestSurrogateND
x, y = -5:10, 0:15 # hide
p1 = surface(x, y, (x1,x2) -> bukin6((x1,x2))) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
scatter!(xs, ys, zs) # hide
p2 = contour(x, y, (x1,x2) -> bukin6((x1,x2))) # hide
scatter!(xs, ys) # hide
plot(p1, p2, title="True function") # hide
```


### Building a surrogate
Using the sampled points we build the surrogate, the steps are analogous to the 1-dimensional case.

```@example RandomForestSurrogateND
using SurrogatesRandomForest
RandomForest = RandomForestSurrogate(xys, zs,  lower_bound, upper_bound)
```

```@example RandomForestSurrogateND
p1 = surface(x, y, (x, y) -> RandomForest([x y])) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> RandomForest([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2, title="Surrogate") # hide
```


### Optimizing
With our surrogate we can now search for the minima of the function.

Notice how the new sampled points, which were created during the optimization process, are appended to the `xys` array.
This is why its size changes.

```@example RandomForestSurrogateND
size(xys)
```
```@example RandomForestSurrogateND
surrogate_optimize(bukin6, SRBF(), lower_bound, upper_bound, RandomForest, SobolSample(), maxiters=20)
```
```@example RandomForestSurrogateND
size(xys)
```

```@example RandomForestSurrogateND
p1 = surface(x, y, (x, y) -> RandomForest([x y])) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
zs = bukin6.(xys) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> RandomForest([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2) # hide
```
