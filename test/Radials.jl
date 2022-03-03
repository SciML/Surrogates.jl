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
my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial)
est = my_rad(3.0)
@test est ≈ 6.0
add_point!(my_rad, 4.0, 10.0)
est = my_rad(3.0)
@test est ≈ 6.0
add_point!(my_rad,[3.2,3.3,3.4],[8.0,9.0,10.0])
est = my_rad(3.0)
@test est ≈ 6.0

my_rad = RadialBasis(x, y, lb, ub, rad = cubicRadial)
q = 2
my_rad = RadialBasis(x,y,lb,ub, rad = multiquadricRadial)


#ND
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
lb = [0.0,3.0,6.0]
ub = [4.0,7.0,10.0]
#bounds = [[0.0, 3.0, 6.0], [4.0, 7.0, 10.0]]
my_rad = RadialBasis(x, y, lb, ub)
est = my_rad((1.0,2.0,3.0))
@test est ≈ 4.0


#WITH ADD_POINT, adding singleton
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]
lb = [0.0,3.0,6.0]
ub = [4.0,7.0,10.0]
#bounds = [[0.0,3.0,6.0],[4.0,7.0,10.0]]
my_rad = RadialBasis(x,y,lb,ub,rad = linearRadial, scale_factor = 1.0)
add_point!(my_rad,(9.0,10.0,11.0),10.0)
est = my_rad((1.0,2.0,3.0))
@test est ≈ 4.0

#WITH ADD_POINT, adding more
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]
#bounds = [[0.0,3.0,6.0],[4.0,7.0,10.0]]
lb = [0.0,3.0,6.0]
ub = [4.0,7.0,10.0]
my_rad = RadialBasis(x,y,lb,ub)
add_point!(my_rad,[(9.0,10.0,11.0),(12.0,13.0,14.0)],[10.0,11.0])
est = my_rad((1.0,2.0,3.0))
@test est ≈ 4.0

lb = [0.0,0.0,0.0]
ub = [10.0,10.0,10.0]
#bounds = [lb,ub]
x = [(1.0,2.0,3.0),(4.0,5.0,6.0),(7.0,8.0,9.0)]
y = [4.0,5.0,6.0]

my_rad_ND = RadialBasis(x,y,lb,ub)
add_point!(my_rad_ND,(3.5,4.5,1.2),18.9)
add_point!(my_rad_ND,[(3.2,1.2,6.7),(3.4,9.5,7.4)],[25.72,239.0])
my_rad_ND = RadialBasis(x,y,lb,ub,rad = cubicRadial)
my_rad_ND = RadialBasis(x,y,lb,ub, rad = multiquadricRadial)
prediction = my_rad_ND((1.0,1.0,1.0))


f = x -> x[1]*x[2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(500, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y = f.(x)
my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial)
@test my_radial_basis((1.0, 2.0)) ≈ 2
my_radial_basis = RadialBasis(x, y, lb, ub, rad =linearRadial)
@test my_radial_basis((1.0, 2.0)) ≈ 2

f = x -> x[1]*x[2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(5, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y = f.(x)
my_radial_basis = RadialBasis(x, y, lb,ub, rad = linearRadial)
@test my_radial_basis((1.0, 2.0)) ≈ 2

# Multi-output
f  = x -> [x^2, x]
lb = 1.0
ub = 10.0
x  = sample(5, lb, ub, SobolSample())
push!(x, 2.0)
y  = f.(x)
my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial)
my_radial_basis(2.0)
@test my_radial_basis(2.0) ≈ [4, 2]

f  = x -> [x[1]*x[2], x[1]+x[2]^2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x  = sample(5, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y  = f.(x)
my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial)
my_radial_basis((1.0, 2.0))
@test my_radial_basis((1.0, 2.0)) ≈ [2, 5]

x_new = (2.0, 2.0)
y_new = f(x_new)
add_point!(my_radial_basis, x_new, y_new)
@test my_radial_basis(x_new) ≈ y_new



#sparse

#1D
lb = 0.0
ub = 4.0
x = [1.0,2.0,3.0]
y = [4.0,5.0,6.0]
my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial, sparse = true)

#ND
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
lb = [0.0,3.0,6.0]
ub = [4.0,7.0,10.0]
#bounds = [[0.0, 3.0, 6.0], [4.0, 7.0, 10.0]]
my_rad = RadialBasis(x, y, lb, ub, sparse = true)


#test to verify multiquadricRadial with default scale_factor
lb = [0.0, 0.0, 0.0]
ub = [3.0, 3.0, 3.0]
n_samples = 100
g(x) = sqrt(x[1]^2 + x[2]^2 + x[3]^2) 
x = sample(n_samples, lb, ub, SobolSample()) 
y = g.(x) 
mq_rad = RadialBasis(x, y, lb, ub, rad = multiquadricRadial) 
@test isapprox( mq_rad([2.0, 2.0, 1.0]), g([2.0, 2.0, 1.0]), atol = .0001)
