using Base
using Test
using LinearAlgebra
using Surrogates


#1D
lb = 0.0
ub = 10.0
obj_1D = x -> log(x)
x = sample(5,lb,ub,UniformSample())
y = obj_1D.(x)
linear = x -> x
cubic = x -> x^3
λ = 2.3
multiquadr = x -> sqrt(x^2+λ^2)
q = 1
my_rad = RadialBasis(x,y,lb,ub,linear,q)
add_point!(my_rad,3.5,1.25)
add_point!(my_rad,[3.7,3.9],[1.30,1.36])
pred = my_rad(4.0)
my_rad = RadialBasis(x,y,lb,ub,cubic,q)
q = 2
my_rad = RadialBasis(x,y,lb,ub,multiquadr,q)


#ND
lb = [0.0,0.0,0.0]
ub = [10.0,10.0,10.0]
bounds = [lb,ub]
obj_ND = x-> x[1]*x[2]*x[3]
x = sample(5,lb,ub,SobolSample())
y = obj_ND.(x)
linear = x -> norm(x)
cubic = x -> norm(x)^3
λ = 2.3
multiquadr = x -> sqrt(norm(x)^2+λ^2)
q = 1
my_rad_ND = RadialBasis(x,y,bounds,linear,q)
add_point!(my_rad_ND,(3.5,4.5,1.2),18.9)
add_point!(my_rad_ND,[(3.2,1.2,6.7),(3.4,9.5,7.4)],[25.72,239.0])
my_rad_ND = RadialBasis(x,y,bounds,cubic,q)
q = 2
my_rad_ND = RadialBasis(x,y,bounds,multiquadr,q)
prediction = my_rad_ND((1.0,1.0,1.0))
