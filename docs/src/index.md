![SurrogatesLogo](images/Surrogates.png)
# Overview
A surrogate model is an approximation method that mimics the behavior of a computationally
expensive simulation. In more mathematical terms: suppose we are attempting to optimize a function
``\; f(p)``, but each calculation of ``\; f`` is very expensive. It may be the case we need to solve a PDE for each point or use advanced numerical linear algebra machinery which is usually costly. The idea is then to develop a surrogate model ``\; g`` which approximates ``\; f`` by training on previous data collected from evaluations of ``\; f``.
The construction of a surrogate model can be seen as a three steps process:
- Sample selection
- Construction of the surrogate model
- Surrogate optimization

The sampling methods are super important for the behaviour of the Surrogate.
At the moment they are:
- Grid sample
- Uniform sample
- Sobol sample
- Latin Hypercupe sample
- Low discrepancy sample

The available surrogates are:
- Linear
- Radial Basis
- Kriging
- Neural Network
- Support vector machine
- Random Forest

After the Surrogate is built, we need to optimize it with respect to some objective function.
That is, simultaneously looking for a minimum **and** sampling the most unknown region.  
The available optimization methods are:
- Stochastic RBF (SRBF)
- Lower confidence bound strategy (LCBS)
- Expected improvement (EI)
- Dynamic coordinate search (DYCORS)


# Installation
In the REPL:
```
]add https://github.com/JuliaDiffEq/Surrogates.jl
```

# Quick example
```
using Surrogates
num_samples = 10
lb = 0.0
ub = 10.0

#Sampling
x = Sample(num_samples,lb,ub,SobolSample())
f = x-> log(x)*x^2+x^3
y = f.(x)

#Creating surrogate
my_lobachesky = LobacheskySurrogate(x,y,lb,ub)

#Approximanting value at 5.0
value = my_lobachesky(5.0)

#Adding more data points
surrogate_optimize(f,SRBF(),lb,ub,my_lobachesky,UniformSample())

#New approximation
value = my_lobachesky(5.0)
```
