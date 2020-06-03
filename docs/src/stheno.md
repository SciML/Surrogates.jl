## Simple Usage

A Kriging surrogate based on Stheno can work right out of the box, although your mileage
may vary.

```
using Stheno
using Surrogates

num_samples = 10
lb = 0.0
ub = 10.0

#Sampling
x = sample(num_samples,lb,ub,SobolSample())
f = x-> log(x)*x^2+x^3
y = f.(x)

surrogate = SthenoKriging(x, y)

# predicting at a point
surrogate(1.0)

# the standard error at a point
std_error_at_point(surrogate, 1.0)
```

## Tuning Gaussian Process/Kriging Hyperparameters

We may be interested in a surrogate for a multi-dimensional input.

```
using Stheno
using Surrogates

num_samples = 10
lb = [0.0; -10.0]
ub = [1.0;  10.0]

#Sampling
x = sample(num_samples, lb, ub, SobolSample())
f = x-> log(1+x[1]+x[2]^2)
y = f.(x)
```

Additionally, the default hyperparameters of the Gaussian processes may be inappropriate
for our problem. There are a number of ways to do this, as shown in the [Stheno tutorial](https://willtebbutt.github.io/Stheno.jl/dev/getting_started/).

Suppose we have defined our Gaussian process (note that each dimension has a different
scaling):

```
l = [1.0; 5.0]
σ² = 0.05
gp = Stheno.GP(σ² * stretch(matern52(), 1 ./ l), Stheno.GPC())
```

We can then pass this to our surrogate fitting process.

```
surrogate = SthenoKriging(x, y, gp)
```

## A Multidimensional Example

Lastly, if we would like to fit a multi-output surrogate with Kriging, we could do
the following (note that we co-define the two Gaussian processes within the Stheno model):

```
using Stheno
using Surrogates

num_samples = 10
lb = [0.0; -10.0]
ub = [1.0;  10.0]

#Sampling
x = sample(num_samples, lb, ub, SobolSample())
f = x-> [log(1+x[1]+x[2]^2), x[1]]
y = f.(x)

l = [1.0; 5.0]
σ² = 0.05
Stheno.@model function m()
    gp1 = GP(σ² * stretch(matern52(), 1 ./ l))
    gp2 = GP(σ² * stretch(matern52(), 1 ./ l)) + gp1
    return gp1, gp2
end

gps = m()

surrogate = SthenoKriging(x, y, gps)
```

Now, we see that the covariance between the two processes is non-zero:
```
x = ColVecs([3.0; 1.0][:, :])
x′ = ColVecs([1.5; 1.5][:, :])
Stheno.cov(surrogate.gp_posterior[1], surrogate.gp_posterior[2], x, x′)
```
