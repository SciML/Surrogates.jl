using Surrogates
using LinearAlgebra



#######SRBF############


##### 1D #####
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

my_k_SRBF1 = Kriging(x,y,p)
surrogate_optimize(objective_function,SRBF(),a,b,my_k_SRBF1,UniformSample())

#Using RadialBasis
my_rad_SRBF1 = RadialBasis(x,y,a,b,z->norm(z),1)
surrogate_optimize(objective_function,SRBF(),a,b,my_rad_SRBF1,UniformSample())


##### ND #####

objective_function_ND = z -> 3*norm(z)+1
x = [(1.4,1.4),(3.0,3.5),(5.2,5.7)]
y = objective_function_ND.(x)
p = [1.5,1.5]
theta = [1.0,1.0]
lb = [1.0,1.0]
ub = [6.0,6.0]

#Kriging

my_k_SRBFN = Kriging(x,y,p,theta)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_k_SRBFN,UniformSample())

#Radials

bounds = [[1.0,6.0],[1.0,6.0]]
my_rad_SRBFN = RadialBasis(x,y,bounds,z->norm(z),1)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_rad_SRBFN,UniformSample())



####### LCBS #########

######1D######
objective_function = x -> 2*x+1
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]
p = 1.8
a = 2
b = 6
my_k_LCBS1 = Kriging(x,y,p)
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
my_k_LCBSN = Kriging(x,y,p,theta)
surrogate_optimize(objective_function_ND,LCBS(),lb,ub,my_k_LCBSN,UniformSample())


##### EI ######

###1D###

objective_function = x -> 2*x+1
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]
p = 2
a = 2
b = 6
my_k_EI1 = Kriging(x,y,p)
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
my_k_E1N = Kriging(x,y,p,theta)
surrogate_optimize(objective_function_ND,EI(),lb,ub,my_k_E1N,UniformSample())




## DYCORS ##

#1D#

objective_function = x -> 3*x+1
x = [2.3,4.0,6.0]
y = objective_function.(x)
p = 2.0
lb = 2.0
ub = 6.0
my_k_DYCORS1 = Kriging(x,y,p)
my_rad_DYCORS1 = RadialBasis(x,y,lb,ub,z->norm(z),1)

surrogate_optimize(objective_function,DYCORS(),lb,ub,my_rad_DYCORS1,UniformSample())
surrogate_optimize(objective_function,DYCORS(),lb,ub,my_k_DYCORS1,UniformSample())



#ND#

objective_function_ND = z -> 2*norm(z)+1
x = [(1.2,3.0),(3.0,3.5),(5.2,5.7)]
y = objective_function_ND.(x)
p = [1.8,1.8]
theta = [2.0,2.0]
lb = [1.0,1.0]
ub = [6.0,6.0]
bounds = [[1.0,6.0],[1.0,6.0]]
my_k_DYCORSN = Kriging(x,y,p,theta)
my_rad_DYCORSN = RadialBasis(x,y,bounds,z->norm(z),1)
surrogate_optimize(objective_function_ND,DYCORS(),lb,ub,my_rad_DYCORSN,UniformSample())
surrogate_optimize(objective_function_ND,DYCORS(),lb,ub,my_k_DYCORSN,UniformSample())
