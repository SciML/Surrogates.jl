# Surrogate
Every surrogate has a different definition depending on the parameters needed.
However, they have in common:

1. ```add_point!(::AbstractSurrogate,x_new,y_new)```
2. ```AbstractSurrogate(value)```
The first function adds a sample point to the surrogate, thus changing the internal
coefficients. The second one calculates the approximation at value.

* Linear surrogate
```@docs
LinearSurrogate(x,y,lb,ub)
```

* Radial basis function surrogate
```@docs
RadialBasis(x,y,bounds,phi::Function,q::Int)
```

* Kriging surrogate
```@docs
Kriging(x,y,p,theta)
```

* Lobachesky surrogate
```@docs
LobacheskySurrogate(x,y,alpha,n::Int,lb,ub)
lobachesky_integral(loba::LobacheskySurrogate,lb,ub)
```

* Support vector machine surrogate, requires `using LIBSVM`
```@docs
SVMSurrogate(x,y,lb,ub)
```

* Random forest surrogate, requires `using XGBoost`
```@docs
RandomForestSurrogate(x,y,lb,ub,num_round)
```

* Neural network surrogate, requires `using Flux`
```@docs
NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)
```

# Creating another surrogate
It's great that you want to add another surrogate to the library!
You will need to:

1. Define a new mutable struct and a constructor function
2. Define add\_point!(your\_surrogate::AbstactSurrogate,x\_new,y\_new)
3. Define your\_surrogate(value) for the approximation

**Example**
```
mutable struct NewSurrogate{X,Y,L,U,C,A,B} <: AbstractSurrogate
  x::X
  y::Y
  lb::L
  ub::U
  coeff::C
  alpha::A
  beta::B
end

function NewSurrogate(x,y,lb,ub,parameters)
    ...
    return NewSurrogate(x,y,lb,ub,calculated\_coeff,alpha,beta)
end

function add_point!(NewSurrogate,x\_new,y\_new)

  nothing
end

function (s::NewSurrogate)(value)
  return s.coeff*value + s.alpha
end
```
