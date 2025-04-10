using Base
using Test
using LinearAlgebra
using Surrogates

#1D
lb = 0.0
ub = 4.0
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
linear = x -> norm(x)
cubic = x -> x^3
λ = 2.3
multiquadr = x -> sqrt(x^2 + λ^2)
q = 1
my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
est = my_rad(3.0)
@test est ≈ 6.0
update!(my_rad, 4.0, 10.0)
est = my_rad(3.0)
@test est ≈ 6.0
update!(my_rad, [3.2, 3.3, 3.4], [8.0, 9.0, 10.0])
est = my_rad(3.0)
@test est ≈ 6.0

my_rad = RadialBasis(x, y, lb, ub, rad = cubicRadial())
q = 2
my_rad = RadialBasis(x, y, lb, ub, rad = multiquadricRadial())

# Test that input dimension is properly checked for 1D radial surrogates
@test_throws ArgumentError my_rad(Float64[])
@test_throws ArgumentError my_rad((2.0, 3.0, 4.0))

#ND
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
lb = [0.0, 3.0, 6.0]
ub = [4.0, 7.0, 10.0]
#bounds = [[0.0, 3.0, 6.0], [4.0, 7.0, 10.0]]
my_rad = RadialBasis(x, y, lb, ub)
est = my_rad((1.0, 2.0, 3.0))
@test est ≈ 4.0

#WITH ADD_POINT, adding singleton
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
lb = [0.0, 3.0, 6.0]
ub = [4.0, 7.0, 10.0]
#bounds = [[0.0,3.0,6.0],[4.0,7.0,10.0]]
my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), scale_factor = 1.0)
update!(my_rad, (9.0, 10.0, 11.0), 10.0)
est = my_rad((1.0, 2.0, 3.0))
@test est ≈ 4.0

#WITH ADD_POINT, adding more
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
#bounds = [[0.0,3.0,6.0],[4.0,7.0,10.0]]
lb = [0.0, 3.0, 6.0]
ub = [4.0, 7.0, 10.0]
my_rad = RadialBasis(x, y, lb, ub)
update!(my_rad, [(9.0, 10.0, 11.0), (12.0, 13.0, 14.0)], [10.0, 11.0])
est = my_rad((1.0, 2.0, 3.0))
@test est ≈ 4.0

lb = [0.0, 0.0, 0.0]
ub = [10.0, 10.0, 10.0]
#bounds = [lb,ub]
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]

my_rad_ND = RadialBasis(x, y, lb, ub)
update!(my_rad_ND, (3.5, 4.5, 1.2), 18.9)
update!(my_rad_ND, [(3.2, 1.2, 6.7), (3.4, 9.5, 7.4)], [25.72, 239.0])
my_rad_ND = RadialBasis(x, y, lb, ub, rad = cubicRadial())
my_rad_ND = RadialBasis(x, y, lb, ub, rad = multiquadricRadial())
prediction = my_rad_ND((1.0, 1.0, 1.0))

f = x -> x[1] * x[2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(500, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y = f.(x)
my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
@test my_radial_basis((1.0, 2.0)) ≈ 2
my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
@test my_radial_basis((1.0, 2.0)) ≈ 2

f = x -> x[1] * x[2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(5, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y = f.(x)
my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
@test my_radial_basis((1.0, 2.0)) ≈ 2

# Test that input dimension is properly checked for ND radial surrogates
@test_throws ArgumentError my_radial_basis((1.0,))
@test_throws ArgumentError my_radial_basis((2.0, 3.0, 4.0))

# Multi-output
f = x -> [x^2, x]
lb = 1.0
ub = 10.0
x = sample(5, lb, ub, SobolSample())
push!(x, 2.0)
y = f.(x)
my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
my_radial_basis(2.0)
@test my_radial_basis(2.0) ≈ [4, 2]

f = x -> [x[1] * x[2], x[1] + x[2]^2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(5, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y = f.(x)
my_radial_basis = RadialBasis(x, y, lb, ub, rad = linearRadial())
my_radial_basis((1.0, 2.0))
@test my_radial_basis((1.0, 2.0)) ≈ [2, 5]

x_new = (2.0, 2.0)
y_new = f(x_new)
update!(my_radial_basis, x_new, y_new)
@test my_radial_basis(x_new) ≈ y_new

#sparse

#1D
lb = 0.0
ub = 4.0
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), sparse = true)

#ND
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
lb = [0.0, 3.0, 6.0]
ub = [4.0, 7.0, 10.0]
#bounds = [[0.0, 3.0, 6.0], [4.0, 7.0, 10.0]]
my_rad = RadialBasis(x, y, lb, ub, sparse = true)

#test to verify multiquadricRadial with default scale_factor
lb = [0.0, 0.0, 0.0]
ub = [3.0, 3.0, 3.0]
n_samples = 100
g(x) = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
x = sample(n_samples, lb, ub, SobolSample())
y = g.(x)
mq_rad = RadialBasis(x, y, lb, ub, rad = multiquadricRadial())
@test isapprox(mq_rad([2.0, 2.0, 1.0]), g([2.0, 2.0, 1.0]), atol = 0.0001)
mq_rad = RadialBasis(x, y, lb, ub, rad = multiquadricRadial(0.9)) # different shape parameter should not be as accurate
@test !isapprox(mq_rad([2.0, 2.0, 1.0]), g([2.0, 2.0, 1.0]), atol = 0.0001)

# Issue 316

x = sample(1024, [-0.45, -0.4, -0.9], [0.40, 0.55, 0.35], SobolSample())
lb = [-0.45 -0.4 -0.9]
ub = [0.40 0.55 0.35]

function mockvalues(in)
    x, y, z = in
    p1 = reverse(vec([1.09903695e+01 -1.01500500e+01 -4.06629740e+01 -1.41834931e+01 1.00604784e+01 4.34951623e+00 -1.06519689e-01 -1.93335202e-03]))
    p2 = vec([2.12791877 2.12791877 4.23881665 -1.05464575])
    f = evalpoly(z, p1)
    f += p2[1] * x^2 + p2[2] * y^2 + p2[3] * x^2 * y + p2[4] * x * y^2
    f
end

y = mockvalues.(x)
rbf = RadialBasis(x, y, lb, ub, rad = multiquadricRadial(1.788))
test = (lb .+ ub) ./ 2
@test isapprox(rbf(test), mockvalues(test), atol = 0.001)

# Test regularization parameter
# Check the tests still pass with a small regularization parameter
# 1D
lb = 0.0
ub = 4.0
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), regularization = 1E-12)
est = my_rad(3.0)
@test est ≈ 6.0
update!(my_rad, 4.0, 10.0)
est = my_rad(3.0)
@test est ≈ 6.0
update!(my_rad, [3.2, 3.3, 3.4], [8.0, 9.0, 10.0])
est = my_rad(3.0)
@test est ≈ 6.0

#ND
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
lb = [0.0, 3.0, 6.0]
ub = [4.0, 7.0, 10.0]
my_rad = RadialBasis(x, y, lb, ub)
est = my_rad((1.0, 2.0, 3.0))
@test est ≈ 4.0
#WITH ADD_POINT, adding singleton
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [4.0, 5.0, 6.0]
lb = [0.0, 3.0, 6.0]
ub = [4.0, 7.0, 10.0]
my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), scale_factor = 1.0, regularization = 1E-12)
update!(my_rad, (9.0, 10.0, 11.0), 10.0)
est = my_rad((1.0, 2.0, 3.0))
@test est ≈ 4.0

# Check regularization fixes the SingularException
# 1D
for reg in [0, 1E-12]
    local lb = 0.0
    local ub = 4.0
    # Pass the first point twice to create a singular matrix
    # This should throw a SingularException if regularization is not used
    local x = [1.0, 1.0, 2.0, 3.0]
    local y = [4.0, 4.0, 5.0, 6.0]
    if reg == 0
        @test_throws LinearAlgebra.SingularException RadialBasis(x, y, lb, ub, rad = linearRadial(), regularization = reg)
    else
        local my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), regularization = reg)
        @test my_rad(3.0) ≈ 6.0
        @test my_rad(1.0) ≈ 4.0
    end
end

# ND
for reg in [0, 1E-12]
    local lb = [0.0, 3.0, 6.0]
    local ub = [4.0, 7.0, 10.0]
    local x = [(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    local y = [4.0, 4.0, 5.0, 6.0]
    if reg == 0
        @test_throws LinearAlgebra.SingularException RadialBasis(x, y, lb, ub, rad = linearRadial(), regularization = reg)
    else
        local my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), regularization = reg)
        @test my_rad((1.0, 2.0, 3.0)) ≈ 4.0
        @test my_rad((4.0, 5.0, 6.0)) ≈ 5.0
    end
end
