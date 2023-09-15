using Surrogates
using Test
using Revise 


#1D
lb = 0.0
ub = 10.0
f = x -> log(x) * exp(x)
x = sample(5, lb, ub, SobolSample())
y = f.(x)


# Test lengths of new_x and EI (1D)

my_k = Kriging(x, y, lb, ub)
new_x, eis = Ask(EI(), lb, ub, my_k, SobolSample(), 3, CLmean!)

@test length(new_x) == 3
@test length(eis) == 3

# Test lengths of new_x and SRBF (1D)

my_surr = RadialBasis(x, y, lb, ub)
new_x, eis = Ask(SRBF(), lb, ub, my_surr, SobolSample(), 3, CLmean!)
@test length(new_x) == 3
@test length(eis) == 3


# Test lengths of new_x and EI (ND)

lb = [0.0, 0.0, 1.0]
ub = [5.0, 7.5, 10.0]
x = sample(5, lb, ub, SobolSample())
f = x -> x[1] + x[2] * x[3]
y = f.(x)

my_k = Kriging(x, y, lb, ub)

new_x, eis = Ask(EI(), lb, ub, my_k, SobolSample(), 5, CLmean!)

@test length(new_x) == 5
@test length(eis) == 5

@test length(new_x[1]) == 3

# Test lengths of new_x and SRBF (ND)

my_surr = RadialBasis(x, y, lb, ub)
new_x, eis = Ask(SRBF(), lb, ub, my_surr, SobolSample(), 5, CLmean!)

@test length(new_x) == 5
@test length(eis) == 5

@test length(new_x[1]) == 3

# # Check hyperparameter validation for Ask 
@test_throws ArgumentError new_x, eis = Ask(EI(), lb, ub, my_k, SobolSample(), -1, CLmean!)

