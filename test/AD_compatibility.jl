using Surrogates
using ForwardDiff
using LinearAlgebra
using Flux
using Flux: @epochs

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

#Neural Surrogate StackOverflow ?!?!?
#=
model = Chain(Dense(1,1))
loss(x, y) = Flux.mse(model(x), y)
opt = Descent(0.01)
n_echos = 1
my_neural = NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)
g = x -> ForwardDiff.derivative(my_neural,x)
println(g(5.0))
=#

#Random forest problem in predict
#=
num_round = 2
my_forest_1D = RandomForestSurrogate(x,y,lb,ub,num_round)
g = x -> ForwardDiff.derivative(my_forest_1D,x)
println(g(5.0))
=#

#Second order polynomial
my_second = SecondOrderPolynomialSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.derivative(my_second,x)
g(5.0)

#SVM problem in predict
#=
my_svm = SVMSurrogate(x,y,lb,ub)
g = x -> ForwardDiff.derivative(my_svm,x)
g(5.0)
=#
