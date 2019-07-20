using Surrogates
using LinearAlgebra
using Test
#1D
obj = x -> 3*x + log(x)
a = 1.0
b = 4.0
x = sample(2000,a,b,SobolSample())
y = obj.(x)
alpha = 2.0
n = 2
my_loba = LobacheskySurrogate(x,y,alpha,n,a,b)
val = my_loba(3.83)
#1D integral
int_1D = lobachesky_integral(my_loba,a,b)
#Value of integral taken from quadgk for testing purposes    
@test abs(int_1D - 25.04517738347907) < 10^-5
add_point!(my_loba,3.7,12.1)
add_point!(my_loba,[1.23,3.45],[5.20,109.67])


#ND
obj = x -> x[1]*x[2]
lb = [0.0,0.0]
ub = [8.0,8.0]
alpha = 2.0
n = 2
x = sample(10,lb,ub,SobolSample())
y = obj.(x)
my_loba_ND = LobacheskySurrogate(x,y,alpha,n,lb,ub)
#add_point!(my_loba_ND,[2.0,4.2],4.65188)
#add_point!(my_loba_ND,[[2.0, 3.0],[5.4,3.3]],[2.236,6.328])
#ND integral
int_ND = lobachesky_integral(my_loba_ND,lb,ub)
