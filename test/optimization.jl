using Surrogates
using LinearAlgebra
using Flux
using Flux: @epochs
using PolyChaos

#######SRBF############
##### 1D #####

lb = 0.0
ub = 15.0
objective_function = x -> 2*x+1
x = [2.5,4.0,6.0]
y = [6.0,9.0,13.0]

# In 1D values of p closer to 2 make the det(R) closer and closer to 0,
#this does not happen in higher dimensions because p would be a vector and not
#all components are generally C^inf
p = 1.99
a = 2
b = 6

#Using Kriging
my_k_SRBF1 = Kriging(x,y,lb,ub)
surrogate_optimize(objective_function,SRBF(),a,b,my_k_SRBF1,UniformSample())

#Using RadialBasis
my_rad_SRBF1 = RadialBasis(x,y,a,b,rad = linearRadial())
surrogate_optimize(objective_function,SRBF(),a,b,my_rad_SRBF1,UniformSample())

my_wend_1d = Wendland(x,y,lb,ub)
surrogate_optimize(objective_function,SRBF(),a,b,my_wend_1d,UniformSample())

x = sample(20,lb,ub,SobolSample())
y = objective_function.(x)
my_poly1d = PolynomialChaosSurrogate(x,y,lb,ub)
surrogate_optimize(objective_function,SRBF(),a,b,my_poly1d,LowDiscrepancySample(2))

my_earth1d = EarthSurrogate(x,y,lb,ub)
surrogate_optimize(objective_function,SRBF(),a,b,my_earth1d,LowDiscrepancySample(2))

##### ND #####
objective_function_ND = z -> 3*norm(z)+1
lb = [1.0,1.0]
ub = [6.0,6.0]
x = sample(5,lb,ub,SobolSample())
y = objective_function_ND.(x)
p = [1.5,1.5]
theta = [1.0,1.0]

#Kriging

my_k_SRBFN = Kriging(x,y,lb,ub)
#Every optimization method now returns the y_min and its position
x_min, y_min = surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_k_SRBFN,UniformSample())

#Radials
lb = [1.0,1.0]
ub = [6.0,6.0]
x = sample(5,lb,ub,SobolSample())
objective_function_ND = z -> 3*norm(z)+1
y = objective_function_ND.(x)
my_rad_SRBFN = RadialBasis(x,y,lb,ub,rad = linearRadial())
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_rad_SRBFN,UniformSample())

# Lobachevsky
x = sample(5,lb,ub,UniformSample())
y = objective_function_ND.(x)
alpha = [2.0,2.0]
n = 4
my_loba_ND = LobachevskySurrogate(x,y,lb,ub)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_loba_ND,UniformSample())
#Linear
lb = [1.0,1.0]
ub = [6.0,6.0]
x = sample(500,lb,ub,SobolSample())
objective_function_ND = z -> 3*norm(z)+1
y = objective_function_ND.(x)
my_linear_ND = LinearSurrogate(x,y,lb,ub)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_linear_ND,SobolSample(),maxiters=15)

#=
#SVM
lb = [1.0,1.0]
ub = [6.0,6.0]
x = sample(5,lb,ub,SobolSample())
objective_function_ND = z -> 3*norm(z)+1
y = objective_function_ND.(x)
my_SVM_ND = SVMSurrogate(x,y,lb,ub)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_SVM_ND,SobolSample(),maxiters=15)
=#

#Neural
lb = [1.0,1.0]
ub = [6.0,6.0]
x = sample(5,lb,ub,SobolSample())
objective_function_ND = z -> 3*norm(z)+1
y = objective_function_ND.(x)
model = Chain(Dense(2,1), first)
loss(x, y) = Flux.mse(model(x), y)
opt = Descent(0.01)
n_echos = 1
my_neural_ND_neural = NeuralSurrogate(x,y,lb,ub)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_neural_ND_neural,SobolSample(),maxiters=15)

#Random Forest
using XGBoost
lb = [1.0,1.0]
ub = [6.0,6.0]
x = sample(5,lb,ub,SobolSample())
objective_function_ND = z -> 3*norm(z)+1
y = objective_function_ND.(x)
num_round = 2
my_forest_ND_SRBF = RandomForestSurrogate(x,y,lb,ub,num_round=2)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_forest_ND_SRBF,SobolSample(),maxiters=15)

#Inverse distance surrogate
lb = [1.0,1.0]
ub = [6.0,6.0]
x = sample(5,lb,ub,SobolSample())
objective_function_ND = z -> 3*norm(z)+1
my_p = 2.5
y = objective_function_ND.(x)
my_inverse_ND = InverseDistanceSurrogate(x,y,lb,ub,p=my_p)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_inverse_ND,SobolSample(),maxiters=15)

#SecondOrderPolynomialSurrogate
lb = [0.0,0.0]
ub = [10.0,10.0]
obj_ND = x -> log(x[1])*exp(x[2])
x = sample(15,lb,ub,UniformSample())
y = obj_ND.(x)
my_second_order_poly_ND = SecondOrderPolynomialSurrogate(x,y,lb,ub)
surrogate_optimize(obj_ND,SRBF(),lb,ub,my_second_order_poly_ND,SobolSample(),maxiters=15)

obj_ND = x -> log(x[1])*exp(x[2])
x = sample(40,lb,ub,UniformSample())
y = obj_ND.(x)
my_polyND = PolynomialChaosSurrogate(x,y,lb,ub)
surrogate_optimize(obj_ND,SRBF(),lb,ub,my_polyND,SobolSample(),maxiters=15)

####### LCBS #########
######1D######
objective_function = x -> 2*x+1
lb = 0.0
ub = 15.0
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]
p = 1.8
a = 2.0
b = 6.0
my_k_LCBS1 = Kriging(x,y,lb,ub)
surrogate_optimize(objective_function,LCBS(),a,b,my_k_LCBS1,UniformSample())


##### ND #####
objective_function_ND = z -> 3*norm(z)+1
x = [(1.2,3.0),(3.0,3.5),(5.2,5.7)]
y = objective_function_ND.(x)
p = [1.2,1.2]
theta = [2.0,2.0]
lb = [1.0,1.0]
ub = [6.0,6.0]

#Kriging
my_k_LCBSN = Kriging(x,y,lb,ub)
surrogate_optimize(objective_function_ND,LCBS(),lb,ub,my_k_LCBSN,UniformSample())


##### EI ######

###1D###
objective_function = x -> 2*x+1
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]
lb = 0.0
ub = 15.0
p = 2
a = 2
b = 6
my_k_EI1 = Kriging(x,y,lb,ub)
surrogate_optimize(objective_function,EI(),a,b,my_k_EI1,UniformSample(),maxiters=200,num_new_samples=155)


###ND###
objective_function_ND = z -> 3*norm(z)+1
x = [(1.2,3.0),(3.0,3.5),(5.2,5.7)]
y = objective_function_ND.(x)
p = [1.2,1.2]
theta = [2.0,2.0]
lb = [1.0,1.0]
ub = [6.0,6.0]

#Kriging
my_k_E1N = Kriging(x,y,lb,ub)
surrogate_optimize(objective_function_ND,EI(),lb,ub,my_k_E1N,UniformSample())


## DYCORS ##

#1D#
objective_function = x -> 3*x+1
x = [2.1,2.5,4.0,6.0]
y = objective_function.(x)
p = 1.9
lb = 2.0
ub = 6.0

my_k_DYCORS1 = Kriging(x,y,lb,ub,p=1.9)
surrogate_optimize(objective_function,DYCORS(),lb,ub,my_k_DYCORS1,UniformSample())

my_rad_DYCORS1 = RadialBasis(x,y,lb,ub,rad = linearRadial())
surrogate_optimize(objective_function,DYCORS(),lb,ub,my_rad_DYCORS1,UniformSample())


#ND#
objective_function_ND = z -> 2*norm(z)+1
x = [(2.3,2.2),(1.4,1.5)]
y = objective_function_ND.(x)
p = [1.5,1.5]
theta = [2.0,2.0]
lb = [1.0,1.0]
ub = [6.0,6.0]


my_k_DYCORSN = Kriging(x,y,lb,ub)
surrogate_optimize(objective_function_ND,DYCORS(),lb,ub,my_k_DYCORSN,UniformSample(),maxiters=30)

my_rad_DYCORSN = RadialBasis(x,y,lb,ub,rad = linearRadial())
surrogate_optimize(objective_function_ND,DYCORS(),lb,ub,my_rad_DYCORSN,UniformSample(),maxiters=30)

my_wend_ND = Wendland(x,y,lb,ub)
surrogate_optimize(objective_function_ND,DYCORS(),lb,ub,my_wend_ND,UniformSample(),maxiters=30)

### SOP ###
# 1D
objective_function = x -> 3*x+1
x = sample(20,1.0,6.0,SobolSample())
y = objective_function.(x)
p = 1.9
lb = 1.0
ub = 6.0
num_centers = 2
my_k_SOP1 = Kriging(x,y,lb,ub,p=1.9)
surrogate_optimize(objective_function,SOP(num_centers),lb,ub,my_k_SOP1,SobolSample(),maxiters=60)
#ND
objective_function_ND = z -> 2*norm(z)+1
x = [(2.3,2.2),(1.4,1.5)]
y = objective_function_ND.(x)
p = [1.5,1.5]
theta = [2.0,2.0]
lb = [1.0,1.0]
ub = [6.0,6.0]
my_k_SOPND = Kriging(x,y,lb,ub)
num_centers = 2
surrogate_optimize(objective_function_ND,SOP(num_centers),lb,ub,my_k_SOPND,SobolSample(),maxiters=20)



#multi optimization
#=
f  = x -> [x^2, x]
lb = 1.0
ub = 10.0
x  = sample(100, lb, ub, SobolSample())
y  = f.(x)
my_radial_basis_smb = RadialBasis(x, y, lb, ub, rad = linearRadial())
surrogate_optimize(f,SMB(),lb,ub,my_radial_basis_ego,SobolSample())



f  = x -> [x^2, x]
lb = 1.0
ub = 10.0
x  = sample(100, lb, ub, SobolSample())
y  = f.(x)
my_radial_basis_rtea = RadialBasis(x, y, lb, ub, rad = linearRadial())
Z = 0.8 #percentage
K = 2 #number of revaluations
p_cross = 0.5 #crossing vs copy
n_c = 1.0 # hyperparameter for children creation
sigma = 1.5 # mutation
surrogate_optimize(f,RTEA(Z,K,p_cross,n_c,sigma),lb,ub,my_radial_basis_rtea,SobolSample())
=#
