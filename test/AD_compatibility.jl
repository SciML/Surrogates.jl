using Surrogates
using ForwardDiff
using LinearAlgebra
using Flux
using Flux: @epochs
using Tracker
using Zygote
#using Zygote: @nograd

#=
#FORWARD
###### 1D ######
lb = 0.0
ub = 10.0
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> x^2
y = f.(x)

#Radials
my_rad = RadialBasis(x,y,lb,ub,x->norm(x),2)
g = x -> ForwardDiff.derivative(my_rad,x)
g(5.0)

#Kriging
p = 1.5
my_krig = Kriging(x,y,p)
g = x -> ForwardDiff.derivative(my_krig,x)
g(5.0)

#Linear Surrogate
my_linear = LinearSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.derivative(my_linear,x)
g(5.0)

#Inverse distance
p = 1.4
my_inverse = InverseDistanceSurrogate(x,y,p,lb,ub)
g = x -> ForwardDiff.derivative(my_inverse,x)
g(5.0)

#Lobachesky
n = 4
α = 2.4
my_loba = LobacheskySurrogate(x,y,α,n,lb,ub)
g = x -> ForwardDiff.derivative(my_loba,x)
g(5.0)

#Second order polynomial
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.derivative(my_second,x)
g(5.0)

###### ND ######
lb = [0.0,0.0]
ub = [10.0,10.0]
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> x[1]*x[2]
y = f.(x)

#Radials
my_rad = RadialBasis(x,y,[lb,ub],z->norm(z),2)
g = x -> ForwardDiff.gradient(my_rad,x)
g([2.0,5.0])

#Kriging
theta = [2.0,2.0]
p = [1.9,1.9]
my_krig = Kriging(x,y,p,theta)
g = x -> ForwardDiff.gradient(my_krig,x)
g([2.0,5.0])

#Linear Surrogate
my_linear = LinearSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.gradient(my_linear,x)
g([2.0,5.0])

#Inverse Distance
p = 1.4
my_inverse = InverseDistanceSurrogate(x,y,p,lb,ub)
g = x -> ForwardDiff.gradient(my_inverse,x)
g([2.0,5.0])

#Lobachesky
alpha = [1.4,1.4]
n = 4
my_loba_ND = LobacheskySurrogate(x,y,alpha,n,lb,ub)
g = x -> ForwardDiff.gradient(my_loba_ND,x)
g([2.0,5.0])

#Second order polynomial
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.gradient(my_second,x)
g([2.0,5.0])

### Tracker ###
#=
#1D
lb = 0.0
ub = 10.0
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> x^2
y = f.(x)

#Radials
my_rad = RadialBasis(x,y,lb,ub,x->norm(x),2)
g = x -> Tracker.gradient(my_rad,x)
g(5.0)

#Kriging
p = 1.5
my_krig = Kriging(x,y,p)
g = x -> Tracker.gradient(my_krig,x)
g(5.0)

#Linear Surrogate
my_linear = LinearSurrogate(x,y,lb,ub)
g = x -> Tracker.gradient(my_linear,x)
g(5.0)

#Inverse distance
p = 1.4
my_inverse = InverseDistanceSurrogate(x,y,p,lb,ub)
g = x -> Tracker.gradient(my_inverse,x)
g(5.0)

#Lobachesky
n = 4
α = 2.4
my_loba = LobacheskySurrogate(x,y,α,n,lb,ub)
g = x -> Tracker.gradient(my_loba,x)
g(5.0)

#Second order polynomial
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
g = x -> Tracker.gradient(my_second,x)
g(5.0)

#ND
lb = [0.0,0.0]
ub = [10.0,10.0]
n = 100
x = sample(n,lb,ub,SobolSample())
f = x -> x[1]*x[2]^2
y = f.(x)

#Radials
my_rad = RadialBasis(x,y,[lb,ub],z->norm(z),2)
g = x -> Tracker.gradient(my_rad,x)
g([2.0,5.0])

#Kriging
theta = [2.0,2.0]
p = [1.9,1.9]
my_krig = Kriging(x,y,p,theta)
g = x -> Tracker.gradient(my_krig,x)
g([2.0,5.0])

#Linear Surrogate
my_linear = LinearSurrogate(x,y,lb,ub)
g = x -> Tracker.gradient(my_linear,x)
g([2.0,5.0])

#Inverse Distance
p = 1.4
my_inverse = InverseDistanceSurrogate(x,y,p,lb,ub)
g = x -> Tracker.gradient(my_inverse,x)
g([2.0,5.0])

#Lobachesky
alpha = [1.4,1.4]
n = 4
my_loba_ND = LobacheskySurrogate(x,y,alpha,n,lb,ub)
g = x -> Tracker.gradient(my_loba_ND,x)
g([2.0,5.0])

#Second order polynomial
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
g = x -> Tracker.gradient(my_second,x)
g([2.0,5.0])
=#

### ZYGOTE ###
=#
###### 1D ######
lb = 0.0
ub = 10.0
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> x^2
y = f.(x)

#Radials
my_rad = RadialBasis(x,y,lb,ub,x->norm(x),2)
g = x -> my_rad'(x)
g(5.0)

Zygote.refresh()
#Kriging
p = 1.5
my_krig = Kriging(x,y,p)
g = x -> my_krig'(x)
g(5.0)

Zygote.refresh()
#Linear Surrogate
my_linear = LinearSurrogate(x,y,lb,ub)
g = x -> my_linear'(x)
g(5.0)

Zygote.refresh()
#Inverse distance
p = 1.4
my_inverse = InverseDistanceSurrogate(x,y,p,lb,ub)
g = x -> my_inverse'(x)
g(5.0)

Zygote.refresh()
#Second order polynomial
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
g = x -> my_second'(x)
g(5.0)

Zygote.refresh()
#Lobachesky
n = 4
α = 2.4
my_loba = LobacheskySurrogate(x,y,α,n,lb,ub)
g = x -> my_loba'(x)
g(0.0)

#NN
Zygote.refresh()
model = Chain(Dense(1,1))
loss(x, y) = Flux.mse(model(x), y)
opt = Descent(0.01)
n_echos = 1
my_neural = NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)
g = x->my_neural'(x)
g(3.4)

###### ND ######

lb = [0.0,0.0]
ub = [10.0,10.0]
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> x[1]*x[2]
y = f.(x)

#Radials
my_rad = RadialBasis(x,y,[lb,ub],z->norm(z),2)
g = x -> Zygote.gradient(my_rad,x)
g((2.0,5.0))

#Kriging
Zygote.refresh()
theta = [2.0,2.0]
p = [1.9,1.9]
my_krig = Kriging(x,y,p,theta)
g = x -> Zygote.gradient(my_krig,x)
g((2.0,5.0))

#Linear Surrogate
Zygote.refresh()
my_linear = LinearSurrogate(x,y,lb,ub)
g = x -> Zygote.gradient(my_linear,x)
g((2.0,5.0))

#Inverse Distance
Zygote.refresh()
p = 1.4
my_inverse = InverseDistanceSurrogate(x,y,p,lb,ub)
g = x -> Zygote.gradient(my_inverse,x)
g((2.0,5.0))

#Lobachesky not working yet weird issue with Zygote @nograd
#=
Zygote.refresh()
alpha = [1.4,1.4]
n = 4
my_loba_ND = LobacheskySurrogate(x,y,alpha,n,lb,ub)
g = x -> Zygote.gradient(my_loba_ND,x)
g((2.0,5.0))
=#

#Second order polynomial mutating arrays
Zygote.refresh()
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
g = x -> Zygote.gradient(my_second,x)
g((2.0,5.0))

#NN
Zygote.refresh()
model = Chain(Dense(2,1), sum)
loss(x, y) = Flux.mse(model(x), y)
opt = Descent(0.01)
n_echos = 1
my_neural = NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)
g = x -> Zygote.gradient(my_neural,x)
g((2.0,5.0))

###### ND -> ND ######

lb = [0.0, 0.0]
ub = [10.0, 2.0]
n = 5
x = sample(n,lb,ub,SobolSample())
f = x -> [x[1]^2, x[2]]
y = f.(x)

#NN
Zygote.refresh()
model = Chain(Dense(2,2))
loss(x, y) = Flux.mse(model(x), y)
opt = Descent(0.01)
n_echos = 1
my_neural = NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)
Zygote.gradient(x -> sum(my_neural(x)), (2.0, 5.0))

Zygote.refresh()
my_rad = RadialBasis(x,y,[lb,ub],z->norm(z),1)
Zygote.gradient(x -> sum(my_rad(x)), (2.0, 5.0))

Zygote.refresh()
p = 1.4
my_inverse = InverseDistanceSurrogate(x,y,p,lb,ub)
my_inverse((2.0, 5.0))
Zygote.gradient(x -> sum(my_inverse(x)), (2.0, 5.0))

Zygote.refresh()
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
Zygote.gradient(x -> sum(my_second(x)), (2.0, 5.0))
