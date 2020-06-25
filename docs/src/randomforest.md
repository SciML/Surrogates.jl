## Random forests surrogate tutorial

Random forests is a supervised learning algorithm that randomly creates and merges multiple decision trees into one forest.

We are going to use a Random forests surrogate to optimize $f(x)=sin(x)+sin(10/3 * x)$.

First of all import `Surrogates` and `Plots`.
```@example RandomForestSurrogate_tutorial
using Surrogates
using Plots
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
plot!(f, label="True function", xlims=(lower_bound, upper_bound))
```
### Building a surrogate

With our sampled points we can build the Random forests surrogate using the `RandomForestSurrogate` function.

`randomforest_surrogate` behaves like an ordinary function which we can simply plot. Addtionally you can specify the number of trees created
using the parameter num_round

```@example RandomForestSurrogate_tutorial
num_round = 2
randomforest_surrogate = RandomForestSurrogate(x ,y ,lower_bound, upper_bound, num_round = 2)
plot(x, y, seriestype=:scatter, label="Sampled points", xlims=(lower_bound, upper_bound))
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(randomforest_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```
### Optimizing
Having built a surrogate, we can now use it to search for minimas in our original function `f`.

To optimize using our surrogate we call `surrogate_optimize` method. We choose to use Stochastic RBF as optimization technique and again Sobol sampling as sampling technique.

```@example RandomForestSurrogate_tutorial
@show surrogate_optimize(f, SRBF(), lower_bound, upper_bound, randomforest_surrogate, SobolSample())
scatter(x, y, label="Sampled points")
plot!(f, label="True function",  xlims=(lower_bound, upper_bound))
plot!(randomforest_surrogate, label="Surrogate function",  xlims=(lower_bound, upper_bound))
```
