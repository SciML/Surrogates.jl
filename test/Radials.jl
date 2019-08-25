using Base
using Test
using LinearAlgebra
using Surrogates


#1D
lb = 0.0
ub = 4.0
x = [1.0,2.0,3.0]
y = [4.0,5.0,6.0]
linear = x -> norm(x)
cubic = x -> x^3
λ = 2.3
multiquadr = x -> sqrt(x^2+λ^2)
q = 1
my_rad = RadialBasis(x,y,lb,ub,linear,q)
est = my_rad(3.0)
@test est ≈ 7.875
add_point!(my_rad,4.0,10.0)
est = my_rad(3.0)
@test est ≈ 6.499999999999991
add_point!(my_rad,[3.2,3.3,3.4],[8.0,9.0,10.0])
est_rad2 = my_rad(3.0)
@test est_rad2 ≈ 7.79959017

my_rad = RadialBasis(x,y,lb,ub,cubic,q)
q = 2
my_rad = RadialBasis(x,y,lb,ub,multiquadr,q)


#ND
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]
lb = [0.0,3.0,6.0]
ub = [4.0,7.0,10.0]
my_rad = RadialBasis(x,y,lb,ub,z->norm(z),1)
est = my_rad((1.0,2.0,3.0))
@test est ≈ 4.0


#WITH ADD_POINT, adding singleton
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]
lb = [0.0,3.0,6.0]
ub = [4.0,7.0,10.0]
my_rad = RadialBasis(x,y,lb,ub,z->norm(z),1)

add_point!(my_rad,(9.0,10.0,11.0),10.0)
est = my_rad((1.0,2.0,3.0))
@test est ≈ 4.0

#WITH ADD_POINT, adding more
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]
lb = [0.0,3.0,6.0]
ub = [4.0,7.0,10.0]
my_rad = RadialBasis(x,y,lb,ub,z->norm(z),1)
add_point!(my_rad,[(9.0,10.0,11.0),(12.0,13.0,14.0)],[10.0,11.0])
est = my_rad((1.0,2.0,3.0))
@test est ≈ 4.0

lb = [0.0,0.0,0.0]
ub = [10.0,10.0,10.0]
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]
linear = x -> norm(x)
cubic = x -> norm(x)^3
λ = 2.3
multiquadr = x -> sqrt(norm(x)^2+λ^2)
q = 1
my_rad_ND = RadialBasis(x,y,lb,ub,linear,q)
add_point!(my_rad_ND,(3.5,4.5,1.2),18.9)
add_point!(my_rad_ND,[(3.2,1.2,6.7),(3.4,9.5,7.4)],[25.72,239.0])
my_rad_ND = RadialBasis(x,y,lb,ub,cubic,q)
q = 2
my_rad_ND = RadialBasis(x,y,lb,ub,multiquadr,q)
prediction = my_rad_ND((1.0,1.0,1.0))
