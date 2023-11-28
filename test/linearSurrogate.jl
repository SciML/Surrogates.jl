using Surrogates

include("test_utils.jl")

#1D
x = [1.0, 2.0, 3.0]
y = [1.5, 3.5, 5.3]
lb = 0.0
ub = 7.0
my_linear_surr_1D = LinearSurrogate(x, y, lb, ub)
val = my_linear_surr_1D(5.0)
add_point!(my_linear_surr_1D, 4.0, 7.2)
add_point!(my_linear_surr_1D, [5.0, 6.0], [8.3, 9.7])

# Test that input dimension is properly checked for 1D Linear surrogates
@test_throws ArgumentError my_linear_surr_1D(Float64[])
@test_throws ArgumentError my_linear_surr_1D((2.0, 3.0, 4.0))

#ND
lb = [0.0, 0.0]
ub = [10.0, 10.0]
x = sample(5, lb, ub, SobolSample())
y = [4.0, 5.0, 6.0, 7.0, 8.0]
my_linear_ND = LinearSurrogate(x, y, lb, ub)
add_point!(my_linear_ND, (10.0, 11.0), 9.0)
add_point!(my_linear_ND, [(8.0, 5.0), (9.0, 9.5)], [4.0, 5.0])
val = my_linear_ND((10.0, 11.0))

# Test that input dimension is properly checked for ND Linear surrogates
@test_throws ArgumentError my_linear_ND(Float64[])
@test_throws ArgumentError my_linear_ND(1.0)
@test_throws ArgumentError my_linear_ND((2.0, 3.0, 4.0))

num_replicates = 100
for i in 1:num_replicates
    # LinearSurrogate should not interpolate the data unless the data/func is linear
    surr = _random_surrogate(LinearSurrogate)
    @test !_check_interpolation(surr)

    linear_func = x -> sum(x)
    surr = _random_surrogate(LinearSurrogate, linear_func)
    @test _check_interpolation(surr)
end
