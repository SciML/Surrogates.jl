using SafeTestsets
using Surrogates
using Flux
using Flux: @epochs
using QuasiMonteCarlo

@safetestset "SurrogatesFlux" begin

#1D
a = 0.0
b = 10.0
obj_1D = x -> 2*x+3
x = sample(10,0.0,10.,SobolSample())
y = obj_1D.(x);
my_model = Chain(Dense(1,1), first)
my_loss(x, y) = Flux.mse(my_model(x), y)
my_opt = Descent(0.01)
n_echos = 1
my_neural = NeuralSurrogate(x,y,a,b,model = my_model,loss = my_loss,opt = my_opt,n_echos=1)
my_neural_kwargs = NeuralSurrogate(x,y,a,b)
add_point!(my_neural,8.5,20.0)
add_point!(my_neural,[3.2,3.5],[7.4,8.0])
val = my_neural(5.0)

#ND

lb = [0.0,0.0]
ub = [5.0,5.0]
x = sample(5,lb,ub, SobolSample())
obj_ND_neural(x) = x[1]*x[2];
y = obj_ND_neural.(x)
my_model = Chain(Dense(2,1), first)
my_loss(x, y) = Flux.mse(my_model(x), y)
my_opt = Descent(0.01)
n_echos = 1
my_neural = NeuralSurrogate(x,y,lb,ub,model=my_model,loss=my_loss,opt=my_opt,n_echos=1)
my_neural_kwargs = NeuralSurrogate(x,y,lb,ub)
my_neural((3.5, 1.49))
my_neural([3.4,1.4])
add_point!(my_neural,(3.5,1.4),4.9)
add_point!(my_neural,[(3.5,1.4),(1.5,1.4),(1.3,1.2)],[1.3,1.4,1.5])

# Multi-output #98
f  = x -> [x^2, x]
lb = 1.0
ub = 10.0
x  = sample(5, lb, ub, SobolSample())
push!(x, 2.0)
y  = f.(x)
my_model = Chain(Dense(1,2))
my_loss(x, y) = Flux.mse(my_model(x), y)
surrogate = NeuralSurrogate(x,y,lb,ub,model = my_model,loss = my_loss ,opt=my_opt,n_echos=1)
surr_kwargs = NeuralSurrogate(x,y,lb,ub)

f  = x -> [x[1], x[2]^2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x  = sample(20, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y  = f.(x)
my_model = Chain(Dense(2,2))
my_loss(x, y) = Flux.mse(my_model(x), y)
surrogate = NeuralSurrogate(x,y,lb,ub,model=my_model,loss=my_loss,opt=my_opt,n_echos=1)
surrogate_kwargs = NeuralSurrogate(x,y,lb,ub)
surrogate((1.0, 2.0))
x_new = (2.0, 2.0)
y_new = f(x_new)
add_point!(surrogate, x_new, y_new)

end
