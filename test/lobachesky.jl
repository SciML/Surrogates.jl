using Surrogates

x = [1.0,2.0,3.0]
y = [4.0,5.0,6.0]
a = 0.0
b = 4.0
alpha = 2.0
n = 2
my_loba = LobacheskySurrogate(x,y,alpha,n,a,b)
println(my_loba.coeff)
val = my_loba(4.0)
println(val)
