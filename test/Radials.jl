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
my_rad = RadialBasis(x, y, lb, ub, linear, q)
est = my_rad(3.0)
@test est ≈ 6.0
add_point!(my_rad, 4.0, 10.0)
est = my_rad(3.0)
@test est ≈ 6.0
add_point!(my_rad,[3.2,3.3,3.4],[8.0,9.0,10.0])
est = my_rad(3.0)
@test est ≈ 6.0

my_rad = RadialBasis(x, y, lb, ub, cubic, q)
q = 2
my_rad = RadialBasis(x,y,lb,ub,multiquadr,q)


#ND
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
bounds = [[0.0, 3.0, 6.0], [4.0, 7.0, 10.0]]
my_rad = RadialBasis(x, y, bounds, z->norm(z), 1)
est = my_rad((1.0,2.0,3.0))
@test est ≈ 4.0


#WITH ADD_POINT, adding singleton
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]
bounds = [[0.0,3.0,6.0],[4.0,7.0,10.0]]
my_rad = RadialBasis(x,y,bounds,z->norm(z),1)
add_point!(my_rad,(9.0,10.0,11.0),10.0)
est = my_rad((1.0,2.0,3.0))
@test est ≈ 4.0

#WITH ADD_POINT, adding more
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]
bounds = [[0.0,3.0,6.0],[4.0,7.0,10.0]]
my_rad = RadialBasis(x,y,bounds,z->norm(z),1)
add_point!(my_rad,[(9.0,10.0,11.0),(12.0,13.0,14.0)],[10.0,11.0])
est = my_rad((1.0,2.0,3.0))
@test est ≈ 4.0

lb = [0.0,0.0,0.0]
ub = [10.0,10.0,10.0]
bounds = [lb,ub]
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]
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

# #100
f = x -> x[1]*x[2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(500, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y = f.(x)
linear = z -> norm(z)
my_radial_basis = RadialBasis(x, y, [lb, ub], linear, 1)
@test my_radial_basis((1.0, 2.0)) ≈ 2
my_radial_basis = RadialBasis(x, y, [lb, ub], linear, 5)
@test my_radial_basis((1.0, 2.0)) ≈ 2

# #100
f = x -> x[1]*x[2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(5, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y = f.(x)
linear = z -> norm(z)
my_radial_basis = RadialBasis(x, y, [lb, ub], linear, 0)
@test my_radial_basis((1.0, 2.0)) ≈ 2
