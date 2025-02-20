# Surrogate

Every surrogate has a different definition depending on the parameters needed. It uses the interface defined in [SurrogatesBase.jl](https://github.com/SciML/SurrogatesBase.jl). In a nutshell, they use:

 1. `update!(::AbstractDeterministicSurrogate, x_new, y_new)`
 2. `AbstractDeterministicSurrogate(value)`
    
The first function adds a sample point to the surrogate, thus changing the internal coefficients. The second one calculates the approximation at value.

  - Linear surrogate

```@docs
LinearSurrogate(x,y,lb,ub)
```

  - Radial basis function surrogate

```@docs
RadialBasis(x, y, lb, ub; rad::RadialFunction = linearRadial, scale_factor::Real=1.0, sparse = false)
```

  - Kriging surrogate

```@docs
Kriging(x,y,p,theta)
```

  - Lobachevsky surrogate

```@docs
LobachevskySurrogate(x,y,lb,ub; alpha = collect(one.(x[1])),n::Int = 4, sparse = false)
lobachevsky_integral(loba::LobachevskySurrogate,lb,ub)
```

  - Support vector machine surrogate, requires `using LIBSVM` and `using SurrogatesSVM`

```
SVMSurrogate(x,y,lb::Number,ub::Number)
```

  - Random forest surrogate, requires `using XGBoost` and `using SurrogatesRandomForest`

```
RandomForestSurrogate(x,y,lb,ub;num_round::Int = 1)
```

  - Neural network surrogate, requires `using Flux` and `using SurrogatesFlux`

```
NeuralSurrogate(x,y,lb,ub; model = Chain(Dense(length(x[1]),1), first), loss = (x,y) -> Flux.mse(model(x), y),opt = Descent(0.01),n_echos::Int = 1)
```

# Creating another surrogate

It's great that you want to add another surrogate to the library!
You will need to:

 1. Define a new mutable struct and a constructor function
 2. Define update!(your\_surrogate, x\_new, y\_new)
 3. Define your\_surrogate(value) for the approximation

## Example

```julia
mutable struct NewSurrogate{X,Y,L,U,C,A,B} <: AbstractDeterministicSurrogate
  x::X
  y::Y
  lb::L
  ub::U
  coeff::C
  alpha::A
  beta::B
end

function NewSurrogate(x, y, lb, ub, parameters)
    ...
    return NewSurrogate(x, y, lb, ub, calculated\_coeff, alpha, beta)
end

function update!(NewSurrogate, x_new, y_new)
  ...
end

function (s::NewSurrogate)(value)
  return s.coeff*value + s.alpha
end
```
