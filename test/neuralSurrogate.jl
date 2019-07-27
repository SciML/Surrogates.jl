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

#ND
X = sample(100,[0.,0.],[5.,5.], SobolSample())
obj(x) = x[1]*x[2];
Y = obj.(X)
X = vcat(map(x->x', X)...)
data = []
for i in 1:size(X,1)
    push!(data, (X[i,:], Y[i]))
end
model = Chain(Dense(2,1))
loss(x, y) = Flux.mse(model(x), y)
opt = Descent(0.01)
ps = Flux.params(model)
@epochs 10 Flux.train!(loss, ps, data, opt)
println(model([3.5, 1.4]))
