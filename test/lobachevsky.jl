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
my_loba = LobachevskySurrogate(x,y,a,b, alpha = 2.0, n = 6)
val = my_loba(3.83)

#1D integral
int_1D = lobachevsky_integral(my_loba,a,b)
int = quadgk(obj,a,b)
int_val_true = int[1]-int[2]
@test abs(int_1D - int_val_true) < 2*10^-5
add_point!(my_loba,3.7,12.1)
add_point!(my_loba,[1.23,3.45],[5.20,109.67])

#ND

obj = x -> x[1] + log(x[2])
lb = [0.0,0.0]
ub = [8.0,8.0]
alpha = [2.4,2.4]
n = 8
x = sample(3200,lb,ub,SobolSample())
y = obj.(x)
my_loba_ND = LobachevskySurrogate(x,y,lb,ub,alpha = [2.4,2.4],n=8)
my_loba_kwargs = LobachevskySurrogate(x,y,lb,ub)
pred = my_loba_ND((1.0,2.0))
#ND

int_ND = lobachevsky_integral(my_loba_ND,lb,ub)
int = hcubature(obj,lb,ub)
int_val_true = int[1]-int[2]
@test abs(int_ND - int_val_true) < 10^-1
add_point!(my_loba_ND,(10.0,11.0),4.0)
add_point!(my_loba_ND,[(12.0, 15.0),(13.0,14.0)],[4.0,5.0])
lobachevsky_integrate_dimension(my_loba_ND,lb,ub,2)

obj = x -> x[1] + log(x[2]) + exp(x[3])
lb = [0.0,0.0,0.0]
ub = [8.0,8.0,8.0]
alpha = [2.4,2.4,2.4]
x = sample(50,lb,ub,SobolSample())
y = obj.(x)
n = 4
my_loba_ND = LobachevskySurrogate(x,y,lb,ub)
lobachevsky_integrate_dimension(my_loba_ND,lb,ub,2)

#Sparse
#1D
obj = x -> 3*x + log(x)
a = 1.0
b = 4.0
x = sample(100,a,b,SobolSample())
y = obj.(x)
alpha = 2.0
n = 6
my_loba = LobachevskySurrogate(x,y,a,b, alpha = 2.0, n = 6,sparse=true)

#ND
obj = x -> x[1] + log(x[2])
lb = [0.0,0.0]
ub = [8.0,8.0]
alpha = [2.4,2.4]
n = 8
x = sample(100,lb,ub,SobolSample())
y = obj.(x)
my_loba_ND = LobachevskySurrogate(x,y,lb,ub,alpha = [2.4,2.4],n=8,sparse=true)
