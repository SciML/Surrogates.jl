using Surrogates
using Test
using QuasiMonteCarlo
#1D
obj = x -> sin(x) + sin(x)^2 + sin(x)^3
lb = 0.0
ub = 10.0
x = sample(5, lb, ub, HaltonSample())
y = obj.(x)
p = 3.5
InverseDistance = InverseDistanceSurrogate(x, y, lb, ub, p = 2.4)
InverseDistance_kwargs = InverseDistanceSurrogate(x, y, lb, ub)
prediction = InverseDistance(5.0)
update!(InverseDistance, 5.0, -0.91)
update!(InverseDistance, [5.1, 5.2], [1.0, 2.0])

# Test that input dimension is properly checked for 1D inverse distance surrogates
@test_throws ArgumentError InverseDistance(Float64[])
@test_throws ArgumentError InverseDistance((2.0, 3.0, 4.0))

#ND

lb = [0.0, 0.0]
ub = [10.0, 10.0]
n = 100
x = sample(n, lb, ub, SobolSample())
f = x -> x[1] * x[2]^2
y = f.(x)
p = 3.0
InverseDistance = InverseDistanceSurrogate(x, y, lb, ub, p = p)
prediction = InverseDistance((1.0, 2.0))
update!(InverseDistance, (5.0, 3.4), -0.91)
update!(InverseDistance, [(5.1, 5.2), (5.3, 6.7)], [1.0, 2.0])

# Test that input dimension is properly checked for 1D inverse distance surrogates
@test_throws ArgumentError InverseDistance(Float64[])
@test_throws ArgumentError InverseDistance(2.0)
@test_throws ArgumentError InverseDistance((2.0, 3.0, 4.0))

# Multi-output #98
f = x -> [x^2, x]
lb = 1.0
ub = 10.0
x = sample(5, lb, ub, SobolSample())
push!(x, 2.0)
y = f.(x)
surrogate = InverseDistanceSurrogate(x, y, lb, ub, p = 1.2)
surrogate_kwargs = InverseDistanceSurrogate(x, y, lb, ub)
@test surrogate(2.0) ≈ [4, 2]

f = x -> [x[1], x[2]^2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(20, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y = f.(x)
surrogate = InverseDistanceSurrogate(x, y, lb, ub, p = 1.2)
surrogate_kwargs = InverseDistanceSurrogate(x, y, lb, ub)
@test surrogate((1.0, 2.0)) ≈ [1, 4]
x_new = (2.0, 2.0)
y_new = f(x_new)
update!(surrogate, x_new, y_new)
@test surrogate(x_new) ≈ y_new
surrogate((0.0, 0.0))
