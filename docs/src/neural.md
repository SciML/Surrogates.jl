# Neural network tutorial
!!! note
    This surrogate requires the 'SurrogatesFlux' module which can be added by inputting "]add SurrogatesFlux" from the Julia command line. 

It's possible to define a neural network as a surrogate, using Flux.
This is useful because we can call optimization methods on it.

First of all we will define the `Schaffer` function we are going to build surrogate for.

```@example Neural_surrogate
using Plots
default(c=:matter, legend=false, xlabel="x", ylabel="y") # hide
using Surrogates
using Flux
using SurrogatesFlux

function schaffer(x)
    x1=x[1]
    x2=x[2]
    fact1 = x1 ^2;
    fact2 = x2 ^2;
    y = fact1 + fact2;
end
```


## Sampling

Let's define our bounds, this time we are working in two dimensions. In particular we want our first dimension `x` to have bounds `0, 8`, and `0, 8` for the second dimension. We are taking 60 samples of the space using Sobol Sequences. We then evaluate our function on all of the sampling points.

```@example Neural_surrogate
n_samples = 60
lower_bound = [0.0, 0.0]
upper_bound = [8.0, 8.0]

xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = schaffer.(xys);
```

```@example Neural_surrogate
x, y = 0:8, 0:8 # hide
p1 = surface(x, y, (x1,x2) -> schaffer((x1,x2))) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
scatter!(xs, ys, zs) # hide
p2 = contour(x, y, (x1,x2) -> schaffer((x1,x2))) # hide
scatter!(xs, ys) # hide
plot(p1, p2, title="True function") # hide
```


## Building a surrogate
You can specify your own model, optimization function, loss functions and epochs.
As always, getting the model right is hardest thing.

```@example Neural_surrogate
model1 = Chain(
  Dense(2, 5, σ),
  Dense(5,2,σ),
  Dense(2, 1)
)
neural = NeuralSurrogate(xys, zs, lower_bound, upper_bound, model = model1, n_echos = 10)
```

## Optimization
We can now call an optimization function on the neural network:
```@example Neural_surrogate
surrogate_optimize(schaffer, SRBF(), lower_bound, upper_bound, neural, SobolSample(), maxiters=20, num_new_samples=10)
```
