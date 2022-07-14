using Surrogates
using LinearAlgebra

include("test_utils.jl")

#1D
x = [1.0, 2.0, 3.0]
y = [3.0, 5.0, 7.0]
lb = 0.0
ub = 5.0
my_wend = Wendland(x, y, lb, ub)
add_point!(my_wend, 0.5, 4.0)
val = my_wend(0.5)

# Test that input dimension is properly checked for 1D Wendland surrogates
@test_throws ArgumentError my_wend(Float64[])
@test_throws ArgumentError my_wend((2.0, 3.0, 4.0))

#ND
lb = [0.0, 0.0]
ub = [4.0, 4.0]
x = sample(5, lb, ub, SobolSample())
f = x -> x[1] + x[2]
y = f.(x)
my_wend_ND = Wendland(x, y, lb, ub)
est = my_wend_ND((1.0, 2.0))
add_point!(my_wend_ND, (3.0, 3.5), 4.0)
add_point!(my_wend_ND, [(9.0, 10.0), (12.0, 13.0)], [10.0, 11.0])

# Test that input dimension is properly checked for ND Wendland surrogates
@test_throws ArgumentError my_wend_ND(Float64[])
@test_throws ArgumentError my_wend_ND(2.0)
@test_throws ArgumentError my_wend_ND((2.0, 3.0, 4.0))

#todo
