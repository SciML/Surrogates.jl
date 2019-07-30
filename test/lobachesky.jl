using Surrogates
using LinearAlgebra
using Test
using QuadGK
using Cubature

#1D

obj = x -> 3*x + log(x)
a = 1.0
b = 4.0
x = sample(2000,a,b,SobolSample())
y = obj.(x)
alpha = 2.0
n = 6
my_loba = LobacheskySurrogate(x,y,alpha,n,a,b)
val = my_loba(3.83)

#1D integral
int_1D = lobachesky_integral(my_loba,a,b)
int = quadgk(obj,a,b)
int_val_true = int[1]-int[2]
@test abs(int_1D - int_val_true) < 10^-5
add_point!(my_loba,3.7,12.1)
add_point!(my_loba,[1.23,3.45],[5.20,109.67])
=#
#ND
obj = x -> x[1] + log(x[2])
lb = [0.0,0.0]
ub = [8.0,8.0]
alpha = 2.4
n = 8
s = sample(20,lb,ub,SobolSample())
x = Tuple.(s)
y = obj.(x)
my_loba_ND = LobacheskySurrogate(x,y,alpha,n,lb,ub)

#ND
int_ND = lobachesky_integral(my_loba_ND,lb,ub)
int = hcubature(obj,lb,ub)
int_val_true = int[1]-int[2]
#@test abs(int_ND - int_val_true) < 10^-1
add_point!(my_loba_ND,(10.0,11.0),4.0)
add_point!(my_loba_ND,[(12.0, 15.0),(13.0,14.0)],[4.0,5.0])
