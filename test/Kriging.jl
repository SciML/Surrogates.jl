using LinearAlgebra
using Surrogates
using Test
#1D
lb = 0.0
ub = 10.0
f = x -> log(x)*exp(x)
x = sample(5,lb,ub,SobolSample())
y = f.(x)
p = 1.9
my_k = Kriging(x,y,p)
add_point!(my_k,4.0,75.68)
add_point!(my_k,[5.0,6.0],[238.86,722.84])
pred = my_k(5.5)


#ND
lb = [0.0,0.0,1.0]
ub = [5.0,7.5,10.0]
x = sample(5,lb,ub,SobolSample())
f = x -> x[1]+x[2]*x[3]
y = f.(x)
theta = [2.0,2.0,2.0]
p = [1.9,1.9,1.9]
my_k = Kriging(x,y,p,theta)
add_point!(my_k,(4.0,3.2,9.5),34.4)
add_point!(my_k,[(1.0,4.65,6.4),(2.3,5.4,6.7)],[30.76,38.48])
pred = my_k((3.5,5.5,6.5))


#test sets
#WITHOUT ADD POINT
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [1.0,2.0,3.0]
p = [1.0,1.0,1.0]
theta = [2.0,2.0,2.0]
my_k = Kriging(x,y,p,theta)
est = my_k((1.0,2.0,3.0))
std_err = std_error_at_point(my_k,(1.0,2.0,3.0))


#WITH ADD POINT adding singleton
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [1.0,2.0,3.0]
p = [1.0,1.0,1.0]
theta = [2.0,2.0,2.0]
my_k = Kriging(x,y,p,theta)
add_point!(my_k,(10.0,11.0,12.0),4.0)
est = my_k((10.0,11.0,12.0))
std_err = std_error_at_point(my_k,(10.0,11.0,12.0))
@test std_err < 10^(-6)


#WITH ADD POINT ADDING MORE
x = [(1.0, 2.0, 3.0),(4.0,5.0,6.0),(7.0, 8.0, 9.0)]
y = [1.0,2.0,3.0]
p = [1.0,1.0,1.0]
theta = [2.0,2.0,2.0]
my_k = Kriging(x,y,p,theta)
add_point!(my_k,[(10.0, 11.0, 12.0),(13.0,14.0,15.0)],[4.0,5.0])
est = my_k((10.0,11.0,12.0))
std_err = std_error_at_point(my_k,(10.0,11.0,12.0))
@test std_err < 10^(-6)

#WITHOUT ADD POINT
x = [1.0,2.0,3.0]
y = [4.0,5.0,6.0]
p = 1.3
my_k = Kriging(x,y,p)
est = my_k(1.0)
std_err = std_error_at_point(my_k,1.0)
@test std_err < 10^(-6)

#WITH ADD POINT adding singleton
x = [1.0,2.0,3.0]
y = [4.0,5.0,6.0]
p = 1.3
my_k = Kriging(x,y,p)
add_point!(my_k,4.0,9.0)
est = my_k(4.0)
std_err = std_error_at_point(my_k,4.0)
@test std_err < 10^(-6)


#WITH ADD POINT adding more
x = [1.0,2.0,3.0]
y = [4.0,5.0,6.0]
p = 1.3
my_k = Kriging(x,y,p)
add_point!(my_k,[4.0,5.0,6.0],[9.0,13.0,15.0])
est = my_k(4.0)
std_err = std_error_at_point(my_k,4.0)
@test std_err < 10^(-6)
