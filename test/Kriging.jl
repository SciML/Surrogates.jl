using LinearAlgebra
using Surrogates
using Test
using Statistics

include("test_utils.jl")

#1D
lb = 0.0
ub = 10.0
f = x -> log(x) * exp(x)
x = sample(5, lb, ub, SobolSample())
y = f.(x)
my_p = 1.9

# Check hyperparameter validation for constructing 1D Kriging surrogates
@test_throws ArgumentError my_k=Kriging(x, y, lb, ub, p = -1.0)
@test_throws ArgumentError my_k=Kriging(x, y, lb, ub, p = 3.0)
@test_throws ArgumentError my_k=Kriging(x, y, lb, ub, theta = -2.0)

my_k = Kriging(x, y, lb, ub, p = my_p)
@test my_k.theta ≈ 0.5 * std(x)^(-my_p)

# Check to make sure interpolation condition is satisfied
@test _check_interpolation(my_k)

# Check input dimension validation for 1D Kriging surrogates
@test_throws ArgumentError my_k(rand(3))
@test_throws ArgumentError my_k(Float64[])

add_point!(my_k, 4.0, 75.68)
add_point!(my_k, [5.0, 6.0], [238.86, 722.84])
pred = my_k(5.5)

@test 238.86 ≤ pred ≤ 722.84
@test my_k(5.0) ≈ 238.86
@test std_error_at_point(my_k, 5.0) < 1e-3 * my_k(5.0)

#WITHOUT ADD POINT
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
my_p = 1.3
my_k = Kriging(x, y, lb, ub, p = my_p)
est = my_k(1.0)
@test est == 4.0
std_err = std_error_at_point(my_k, 1.0)
@test std_err < 10^(-6)

#WITH ADD POINT adding singleton
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
my_p = 1.3
my_k = Kriging(x, y, lb, ub, p = my_p)
add_point!(my_k, 4.0, 9.0)
est = my_k(4.0)
std_err = std_error_at_point(my_k, 4.0)
@test std_err < 10^(-6)

#WITH ADD POINT adding more
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
my_p = 1.3
my_k = Kriging(x, y, lb, ub, p = my_p)
add_point!(my_k, [4.0, 5.0, 6.0], [9.0, 13.0, 15.0])
est = my_k(4.0)
std_err = std_error_at_point(my_k, 4.0)
@test std_err < 10^(-6)

#Testing kwargs 1D
kwar_krig = Kriging(x, y, lb, ub);

# Check hyperparameter initialization for 1D Kriging surrogates
p_expected = 1.99
@test kwar_krig.p == p_expected
@test kwar_krig.theta == 0.5 / std(x)^p_expected

#ND
lb = [0.0, 0.0, 1.0]
ub = [5.0, 7.5, 10.0]
x = sample(5, lb, ub, SobolSample())
f = x -> x[1] + x[2] * x[3]
y = f.(x)
my_theta = [2.0, 2.0, 2.0]
my_p = [1.9, 1.9, 1.9]
my_k = Kriging(x, y, lb, ub, p = my_p, theta = my_theta)
add_point!(my_k, (4.0, 3.2, 9.5), 34.4)
add_point!(my_k, [(1.0, 4.65, 6.4), (2.3, 5.4, 6.7)], [30.76, 38.48])
pred = my_k((3.5, 5.5, 6.5))

#test sets
#WITHOUT ADD POINT
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [1.0, 2.0, 3.0]
my_p = [1.0, 1.0, 1.0]
my_theta = [2.0, 2.0, 2.0]
my_k = Kriging(x, y, lb, ub, p = my_p, theta = my_theta)
est = my_k((1.0, 2.0, 3.0))
std_err = std_error_at_point(my_k, (1.0, 2.0, 3.0))

#WITH ADD POINT adding singleton
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [1.0, 2.0, 3.0]
my_p = [1.0, 1.0, 1.0]
my_theta = [2.0, 2.0, 2.0]
my_k = Kriging(x, y, lb, ub, p = my_p, theta = my_theta)
add_point!(my_k, (10.0, 11.0, 12.0), 4.0)
est = my_k((10.0, 11.0, 12.0))
std_err = std_error_at_point(my_k, (10.0, 11.0, 12.0))
@test std_err < 10^(-6)

#WITH ADD POINT ADDING MORE
x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
y = [1.0, 2.0, 3.0]
my_p = [1.0, 1.0, 1.0]
my_theta = [2.0, 2.0, 2.0]
my_k = Kriging(x, y, lb, ub, p = my_p, theta = my_theta)
add_point!(my_k, [(10.0, 11.0, 12.0), (13.0, 14.0, 15.0)], [4.0, 5.0])
est = my_k((10.0, 11.0, 12.0))
std_err = std_error_at_point(my_k, (10.0, 11.0, 12.0))
@test std_err < 10^(-6)

#test kwargs ND (hyperparameter initialization)
kwarg_krig_ND = Kriging(x, y, lb, ub)

# Check hyperparameter validation for ND kriging surrogate construction
@test_throws ArgumentError Kriging(x, y, lb, ub, p = 3 * my_p)
@test_throws ArgumentError Kriging(x, y, lb, ub, p = -my_p)
@test_throws ArgumentError Kriging(x, y, lb, ub, theta = -my_theta)

# Check input dimension validation for ND kriging surrogates
@test_throws ArgumentError kwarg_krig_ND(1.0)
@test_throws ArgumentError kwarg_krig_ND([1.0])
@test_throws ArgumentError kwarg_krig_ND([2.0, 3.0])
@test_throws ArgumentError kwarg_krig_ND(ones(5))

# Test hyperparameter initialization
d = length(x[3])
p_expected = 1.99
@test all(==(p_expected), kwarg_krig_ND.p)
@test all(kwarg_krig_ND.theta .≈ [0.5 / std(x_i[ℓ] for x_i in x)^p_expected for ℓ in 1:3])

num_replicates = 100

for i in 1:num_replicates
    # Check that interpolation condition is satisfied when noise variance is zero
    surr = _random_surrogate(Kriging)
    @test _check_interpolation(surr)

    # Check that we do not satisfy interpolation condition when noise variance isn't zero
    surr = _random_surrogate(Kriging, noise_variance = 0.2)
    @test !_check_interpolation(surr)
end
