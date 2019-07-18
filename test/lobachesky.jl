using Surrogates

#1D
obj = x -> 3*x+1
a = 0.0
b = 4.0
x = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
y = obj.(x)
alpha = 2.0
n = 2
my_loba = LobacheskySurrogate(x,y,alpha,n,a,b)
val = my_loba(3.5)
add_point!(my_loba,3.7,12.1)

#ND
