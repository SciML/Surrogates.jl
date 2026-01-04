using Surrogates
using LinearAlgebra
using QuasiMonteCarlo
using LIBSVM
#######SRBF############
##### 1D #####

lb = 0.0
ub = 15.0
objective_function = x -> 2 * x + 1
x = [2.5, 4.0, 6.0]
y = [6.0, 9.0, 13.0]

# In 1D values of p closer to 2 make the det(R) closer and closer to 0,
#this does not happen in higher dimensions because p would be a vector and not
#all components are generally C^inf
p = 1.99
a = 2
b = 6

#Using Kriging

x = [2.5, 4.0, 6.0]
y = [6.0, 9.0, 13.0]
my_k_SRBF1 = Kriging(x, y, lb, ub; p)
xstar,
    fstar = surrogate_optimize!(
    objective_function, SRBF(), a, b, my_k_SRBF1,
    RandomSample()
)

#Using RadialBasis

x = [2.5, 4.0, 6.0]
y = [6.0, 9.0, 13.0]
my_rad_SRBF1 = RadialBasis(x, y, a, b, rad = linearRadial())
(
    xstar,
    fstar,
) = surrogate_optimize!(
    objective_function, SRBF(), a, b, my_rad_SRBF1,
    RandomSample()
)

x = [2.5, 4.0, 6.0]
y = [6.0, 9.0, 13.0]
my_wend_1d = Wendland(x, y, lb, ub)
xstar,
    fstar = surrogate_optimize!(
    objective_function, SRBF(), a, b, my_wend_1d,
    RandomSample()
)

x = [2.5, 4.0, 6.0]
y = [6.0, 9.0, 13.0]
my_earth1d = EarthSurrogate(x, y, lb, ub)
xstar,
    fstar = surrogate_optimize!(
    objective_function, SRBF(), a, b, my_earth1d,
    HaltonSample()
)

##### ND #####
objective_function_ND = z -> 3 * norm(z) + 1
lb = [1.0, 1.0]
ub = [6.0, 6.0]
x = sample(5, lb, ub, SobolSample())
y = objective_function_ND.(x)

#Kriging

my_k_SRBFN = Kriging(x, y, lb, ub)
#Every optimization method now returns the y_min and its position
x_min,
    y_min = surrogate_optimize!(
    objective_function_ND, SRBF(), lb, ub, my_k_SRBFN,
    RandomSample()
)

#Radials
lb = [1.0, 1.0]
ub = [6.0, 6.0]
x = sample(5, lb, ub, SobolSample())
objective_function_ND = z -> 3 * norm(z) + 1
y = objective_function_ND.(x)
my_rad_SRBFN = RadialBasis(x, y, lb, ub, rad = linearRadial())
surrogate_optimize!(objective_function_ND, SRBF(), lb, ub, my_rad_SRBFN, RandomSample())

# Lobachevsky
x = sample(5, lb, ub, RandomSample())
y = objective_function_ND.(x)
alpha = [2.0, 2.0]
n = 4
my_loba_ND = LobachevskySurrogate(x, y, lb, ub)
surrogate_optimize!(objective_function_ND, SRBF(), lb, ub, my_loba_ND, RandomSample())

#Linear
lb = [1.0, 1.0]
ub = [6.0, 6.0]
x = sample(500, lb, ub, SobolSample())
objective_function_ND = z -> 3 * norm(z) + 1
y = objective_function_ND.(x)
my_linear_ND = LinearSurrogate(x, y, lb, ub)
surrogate_optimize!(
    objective_function_ND, SRBF(), lb, ub, my_linear_ND, SobolSample(),
    maxiters = 15
)

#SVM
lb = [1.0, 1.0]
ub = [6.0, 6.0]
x = sample(5, lb, ub, SobolSample())
objective_function_ND = z -> 3 * norm(z) + 1
y = objective_function_ND.(x)
my_SVM_ND = SVMSurrogate(x, y, lb, ub)
surrogate_optimize!(
    objective_function_ND, SRBF(), lb, ub, my_SVM_ND, SobolSample(), maxiters = 15
)

#Inverse distance surrogate
lb = [1.0, 1.0]
ub = [6.0, 6.0]
x = sample(5, lb, ub, SobolSample())
objective_function_ND = z -> 3 * norm(z) + 1
my_p = 2.5
y = objective_function_ND.(x)
my_inverse_ND = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
surrogate_optimize!(
    objective_function_ND, SRBF(), lb, ub, my_inverse_ND, SobolSample(),
    maxiters = 15
)

#SecondOrderPolynomialSurrogate
lb = [0.0, 0.0]
ub = [10.0, 10.0]
obj_ND = x -> log(x[1]) * exp(x[2])
x = sample(15, lb, ub, RandomSample())
y = obj_ND.(x)
my_second_order_poly_ND = SecondOrderPolynomialSurrogate(x, y, lb, ub)
surrogate_optimize!(
    obj_ND, SRBF(), lb, ub, my_second_order_poly_ND, SobolSample(),
    maxiters = 15
)

####### LCBS #########
######1D######
objective_function = x -> 2 * x + 1
lb = 0.0
ub = 15.0
x = [2.0, 4.0, 6.0]
y = [5.0, 9.0, 13.0]
p = 1.8
a = 2.0
b = 6.0
my_k_LCBS1 = Kriging(x, y, lb, ub)
surrogate_optimize!(objective_function, LCBS(), a, b, my_k_LCBS1, RandomSample())

##### ND #####
objective_function_ND = z -> 3 * norm(z) + 1
x = [(1.2, 3.0), (3.0, 3.5), (5.2, 5.7)]
y = objective_function_ND.(x)
p = [1.2, 1.2]
theta = [2.0, 2.0]
lb = [1.0, 1.0]
ub = [6.0, 6.0]

#Kriging
my_k_LCBSN = Kriging(x, y, lb, ub)
surrogate_optimize!(objective_function_ND, LCBS(), lb, ub, my_k_LCBSN, RandomSample())

##### EI ######

###1D###
objective_function = x -> (x + 1)^2 - x + 2 # Minimum of this function is at x = -0.5, y = -2.75
true_min_x = -0.5
true_min_y = objective_function(true_min_x)
lb = -5
ub = 5
x = sample(5, lb, ub, SobolSample())
y = objective_function.(x)
my_k_EI1 = Kriging(x, y, lb, ub; p = 2)
surrogate_optimize!(
    objective_function, EI(), lb, ub, my_k_EI1, SobolSample(),
    maxiters = 200, num_new_samples = 155
)

# Check that EI is correctly minimizing the objective
y_min, index_min = findmin(my_k_EI1.y)
x_min = my_k_EI1.x[index_min]
@test norm(x_min - true_min_x) < 0.05 * norm(ub .- lb)
@test abs(y_min - true_min_y) < 0.05 * (objective_function(ub) - objective_function(lb))

###ND###
objective_function_ND = z -> 3 * norm(z) + 1 # this is minimized at x = (0, 0), y = 1
true_min_x = (0.0, 0.0)
true_min_y = objective_function_ND(true_min_x)
x = [(1.2, 3.0), (3.0, 3.5), (5.2, 5.7)]
y = objective_function_ND.(x)
min_y = minimum(y)
p = [1.2, 1.2]
theta = [2.0, 2.0]
lb = [-1.0, -1.0]
ub = [6.0, 6.0]

#Kriging
my_k_EIN = Kriging(x, y, lb, ub)
surrogate_optimize!(objective_function_ND, EI(), lb, ub, my_k_EIN, SobolSample())

# Check that EI is correctly minimizing instead of maximizing
y_min, index_min = findmin(my_k_EIN.y)
x_min = my_k_EIN.x[index_min]
@test norm(x_min .- true_min_x) < 0.05 * norm(ub .- lb)
@test abs(y_min - true_min_y) <
    0.05 * (objective_function_ND(ub) - objective_function_ND(lb))

###ND with SectionSampler###
# We will make sure the EI function finds the minimum when constrained to a specific slice of 3D space
objective_function_section = x -> x[1]^2 + x[2]^2 + x[3]^2 # this is minimized at x = (0, 0, 0), y = 0

# We will constrain x[2] to some value
x2_constraint = 2.0
true_min_x = (0.0, x2_constraint, 0.0)
true_min_y = objective_function_section(true_min_x)

sampler = SectionSample([NaN64, x2_constraint, NaN64], SobolSample())
lb = [-1.0, x2_constraint, -1.0]
ub = [6.0, x2_constraint, 6.0]
x = sample(5, lb, ub, sampler)
y = objective_function_section.(x)

#Kriging
my_k_EIN_section = Kriging(x, y, lb, ub)
# Constrain our sampling to the plane where x[2] = 1
surrogate_optimize!(objective_function_section, EI(), lb, ub, my_k_EIN_section, sampler)

# Check that EI is correctly minimizing instead of maximizing
y_min, index_min = findmin(my_k_EIN_section.y)
x_min = my_k_EIN_section.x[index_min]
@test norm(x_min .- true_min_x) < 0.05 * norm(ub .- lb)
@test abs(y_min - true_min_y) <
    0.05 * (objective_function_section(ub) - objective_function_section(lb))

## DYCORS ##

#1D#
objective_function = x -> 3 * x + 1
x = [2.1, 2.5, 4.0, 6.0]
y = objective_function.(x)
p = 1.9
lb = 2.0
ub = 6.0

my_k_DYCORS1 = Kriging(x, y, lb, ub, p = 1.9)
surrogate_optimize!(objective_function, DYCORS(), lb, ub, my_k_DYCORS1, RandomSample())

my_rad_DYCORS1 = RadialBasis(x, y, lb, ub, rad = linearRadial())
surrogate_optimize!(objective_function, DYCORS(), lb, ub, my_rad_DYCORS1, RandomSample())

#ND#
objective_function_ND = z -> 2 * norm(z) + 1
x = [(2.3, 2.2), (1.4, 1.5)]
y = objective_function_ND.(x)
p = [1.5, 1.5]
theta = [2.0, 2.0]
lb = [1.0, 1.0]
ub = [6.0, 6.0]

my_k_DYCORSN = Kriging(x, y, lb, ub)
surrogate_optimize!(
    objective_function_ND, DYCORS(), lb, ub, my_k_DYCORSN, RandomSample(),
    maxiters = 30
)

my_rad_DYCORSN = RadialBasis(x, y, lb, ub, rad = linearRadial())
surrogate_optimize!(
    objective_function_ND, DYCORS(), lb, ub, my_rad_DYCORSN, RandomSample(),
    maxiters = 30
)

my_wend_ND = Wendland(x, y, lb, ub)
surrogate_optimize!(
    objective_function_ND, DYCORS(), lb, ub, my_wend_ND, RandomSample(),
    maxiters = 30
)

### SOP ###
# 1D
objective_function = x -> 3 * x + 1
x = sample(20, 1.0, 6.0, SobolSample())
y = objective_function.(x)
p = 1.9
lb = 1.0
ub = 6.0
num_centers = 2
my_k_SOP1 = Kriging(x, y, lb, ub, p = 1.9)
surrogate_optimize!(
    objective_function, SOP(num_centers), lb, ub, my_k_SOP1, SobolSample(),
    maxiters = 60
)
#ND
objective_function_ND = z -> 2 * norm(z) + 1
x = [(2.3, 2.2), (1.4, 1.5)]
y = objective_function_ND.(x)
p = [1.5, 1.5]
theta = [2.0, 2.0]
lb = [1.0, 1.0]
ub = [6.0, 6.0]
my_k_SOPND = Kriging(x, y, lb, ub)
num_centers = 2
surrogate_optimize!(
    objective_function_ND, SOP(num_centers), lb, ub, my_k_SOPND,
    SobolSample(), maxiters = 20
)

f = x -> [x^2, x]
lb = 1.0
ub = 10.0
x = sample(100, lb, ub, SobolSample())
y = f.(x)
my_radial_basis_smb = RadialBasis(x, y, lb, ub, rad = linearRadial())
surrogate_optimize!(f, SMB(), lb, ub, my_radial_basis_smb, SobolSample())

f = x -> [x, sin(x)]
lb = 1.0
ub = 10.0
x = sample(500, lb, ub, RandomSample())
y = f.(x)
my_radial_basis_rtea = RadialBasis(x, y, lb, ub, rad = linearRadial())
Z = 0.8 #percentage
K = 2 #number of revaluations
p_cross = 0.5 #crossing vs copy
n_c = 1.0 # hyperparameter for children creation
sigma = 1.5 # mutation
surrogate_optimize!(
    f, RTEA(K, Z, p_cross, n_c, sigma), lb, ub,
    my_radial_basis_rtea, SobolSample(); maxiters = 10
)
