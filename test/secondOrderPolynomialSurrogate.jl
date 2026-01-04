using Surrogates
using Test

#1D
lb = 0.0
ub = 5.0
obj_1D = x -> log(x) * exp(x)
x = sample(5, lb, ub, SobolSample())
y = obj_1D.(x)
my_second_order_poly = SecondOrderPolynomialSurrogate(x, y, lb, ub)
val = my_second_order_poly(5.0)
update!(my_second_order_poly, 5.0, 238.86)
update!(my_second_order_poly, [6.0, 7.0], [722.84, 2133.94])

# Test that input dimension is properly checked for 1D SecondOrderPolynomial surrogates
@test_throws ArgumentError my_second_order_poly(Float64[])
@test_throws ArgumentError my_second_order_poly((2.0, 3.0, 4.0))

#ND
lb = [0.0, 0.0]
ub = [10.0, 10.0]
obj_ND = x -> log(x[1]) * exp(x[2])
x = sample(10, lb, ub, RandomSample())
y = obj_ND.(x)
my_second_order_poly = SecondOrderPolynomialSurrogate(x, y, lb, ub)
val = my_second_order_poly((5.0, 7.0))
update!(my_second_order_poly, (5.0, 7.0), 1764.96)
update!(my_second_order_poly, [(1.5, 1.5), (3.4, 5.4)], [1.817, 270.95])

# Test that input dimension is properly checked for ND SecondOrderPolynomial surrogates
@test_throws ArgumentError my_second_order_poly(Float64[])
@test_throws ArgumentError my_second_order_poly(2.0)
@test_throws ArgumentError my_second_order_poly((2.0, 3.0, 4.0))

# Multi-output #98
f = x -> [x^2, x]
lb = 1.0
ub = 10.0
x = sample(5, lb, ub, SobolSample())
push!(x, 2.0)
y = f.(x)
surrogate = SecondOrderPolynomialSurrogate(x, y, lb, ub)
# should be exact
@test surrogate.β ≈ [0 0; 0 1; 1 0]

@test surrogate(2.0) ≈ [4, 2]
@test surrogate(1.0) ≈ [1, 1]

f = x -> [x[1], x[2]^2]
lb = [1.0, 2.0]
ub = [10.0, 8.5]
x = sample(20, lb, ub, SobolSample())
push!(x, (1.0, 2.0))
y = f.(x)
surrogate = SecondOrderPolynomialSurrogate(x, y, lb, ub)
@test surrogate.β ≈ [0 0; 1 0; 0 0; 0 0; 0 0; 0 1]
@test surrogate((1.0, 2.0)) ≈ [1, 4]
x_new = (2.0, 2.0)
y_new = f(x_new)
@test surrogate(x_new) ≈ y_new
update!(surrogate, x_new, y_new)
@test surrogate(x_new) ≈ y_new

# surrogate should recover 2nd order polynomial
function second_order_target(x; a = 0.3, b = [0.7, 0.1], c = [0.3 0.4; 0.4 0.1])
    return a + b' * x + x' * c * x
end
second_order_target(x::Tuple; kwargs...) = f([x...]; kwargs...)
lb = fill(-5.0, 2);
ub = fill(5.0, 2);
n = 10^3;
x = sample(n, lb, ub, SobolSample())
y = second_order_target.(x)
sec = SecondOrderPolynomialSurrogate(x, y, lb, ub)
@test y ≈ sec.(x)
