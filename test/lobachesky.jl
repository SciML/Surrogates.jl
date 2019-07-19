using Surrogates
using LinearAlgebra

#1D
obj = x -> exp(x)*x+1
a = 0.0
b = 4.0
x = sample(20,a,b,SobolSample())
y = obj.(x)
alpha = 2.0
n = 2
my_loba = LobacheskySurrogate(x,y,alpha,n,a,b)
val = my_loba(3.83)
add_point!(my_loba,3.7,12.1)
add_point!(my_loba,[1.23,3.45],[5.20,109.67])

#1D Integral
int = lobachesky_integral(my_loba,a,b)


#ND
obj = x -> 3*norm(x) + 1
lb = [0.0,0.0]
ub = [10.0,10.0]
alpha = 2.0
n = 2
x = sample(5,lb,ub,SobolSample())
y = obj.(x)
my_loba_ND = LobacheskySurrogate(x,y,alpha,n,lb,ub)
add_point!(my_loba_ND,[2.0,4.2],4.65188)
add_point!(my_loba_ND,[[2.0, 3.0],[5.4,3.3]],[2.236,6.328])
