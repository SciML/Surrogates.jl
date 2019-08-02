using Surrogates
using Flux
using Flux: @epochs

#1D
a = 0.0
b = 10.0
obj_1D = x -> 2*x+3
x = sample(10,0.0,10.,SobolSample())
y = obj_1D.(x);
model = Chain(Dense(1,1))
loss(x, y) = Flux.mse(model(x), y)
opt = Descent(0.01)
n_echos = 1
my_neural = NeuralSurrogate(x,y,a,b,model,loss,opt,n_echos)
add_point!(my_neural,8.5,20.0)
add_point!(my_neural,[3.2,3.5],[7.4,8.0])
val = my_neural(5.0)
println(val)

#ND

lb = [0.0,0.0]
ub = [5.0,5.0]
s = sample(5,lb,ub, SobolSample())
x = Tuple.(s)
obj_ND_neural(x) = x[1]*x[2];
y = obj_ND_neural.(x)
model = Chain(Dense(2,1))
loss(x, y) = Flux.mse(model(x), y)
opt = Descent(0.01)
n_echos = 1
my_neural = NeuralSurrogate(x,y,lb,ub,model,loss,opt,n_echos)
my_neural((3.5, 1.49))
my_neural([3.4,1.4])
add_point!(my_neural,(3.5,1.4),4.9)
add_point!(my_neural,[(3.5,1.4),(1.5,1.4),(1.3,1.2)],[1.3,1.4,1.5])
