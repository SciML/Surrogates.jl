using Surrogates
using LinearAlgebra
objective_function = x -> 2*x+1
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]
p = 2
a = 2
b = 6

#Using Kriging
my_k = Kriging(x,y,p)
optimization(a,b,my_k,10,UniformSample(),10)

#Using RadialBasis
my_rad = RadialBasis(x,y,a,b,z->norm(z),1)
optimization(a,b,my_rad,10,UniformSample(),10)
