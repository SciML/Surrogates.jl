# Tensor product function
A tensor product combines multiple functions or vectors. It is a mathematical operation that takes two vector spaces and produces another vector space, capturing their joint behavior across multiple dimensions.

For instance, consider a tensor product function defined as follows:

```\[ f(x) = ∏ᵢ=₁ᵈ cos(aπxᵢ) \]```

Let's import Surrogates and Plots:
```@example tensor
using Surrogates
using Plots
default()
```

Define the 1D objective function:
```@example tensor
function f(x)
    a = 0.5;
    return cos(a*pi*x)
end
```

```@example tensor
n = 30
lb = -5.0
ub = 5.0
a = 0.5
x = sample(n, lb, ub, SobolSample())
y = f.(x)
xs = lb:0.001:ub
scatter(x, y, label="Sampled points", xlims=(lb, ub), ylims=(-1, 1), legend=:top)
plot!(xs, f.(xs), label="True function", legend=:top)
```

Fitting and plotting different surrogates:
```@example tensor
loba_1 = LobachevskySurrogate(x, y, lb, ub)
krig = Kriging(x, y, lb, ub)
scatter(x, y, label="Sampled points", xlims=(lb, ub), ylims=(-2.5, 2.5), legend=:bottom)
plot!(xs,f.(xs), label="True function", legend=:top)
plot!(xs, loba_1.(xs), label="Lobachevsky", legend=:top)
plot!(xs, krig.(xs), label="Kriging", legend=:top)
```

## Kriging Plot

![kriging](https://github.com/Spinachboul/Surrogates.jl/assets/105979087/906e6688-db47-48be-90d1-ea471aacac16)

## Lobachevsky Plot

![lobachevsky](https://github.com/Spinachboul/Surrogates.jl/assets/105979087/678cfc13-0aec-4488-8e4d-39649853ecdd)

## Combined Plot

![combined_plot](https://github.com/Spinachboul/Surrogates.jl/assets/105979087/46762f0d-50c5-4d6c-961a-236fd9fb3ad5)

