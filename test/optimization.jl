using Surrogates
using LinearAlgebra


#=
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

my_k = Kriging(x,y,p)
surrogate_optimize(objective_function,SRBF(),a,b,my_k,UniformSample())

#Using RadialBasis
my_rad = RadialBasis(x,y,a,b,z->norm(z),1)
surrogate_optimize(objective_function,SRBF(),a,b,my_rad,UniformSample())




##### ND #####

objective_function_ND = z -> 3*norm(z)+1
x = [(1.4,1.4),(3.0,3.5),(5.2,5.7)]
y = objective_function_ND.(x)
p = [1.5,1.5]
theta = [1.0,1.0]
lb = [1.0,1.0]
ub = [6.0,6.0]

#Kriging

my_k_ND = Kriging(x,y,p,theta)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_k_ND,UniformSample())

#Radials

bounds = [[1.0,6.0],[1.0,6.0]]
my_rad_ND = RadialBasis(x,y,bounds,z->norm(z),1)
surrogate_optimize(objective_function_ND,SRBF(),lb,ub,my_rad_ND,UniformSample())



####### LCBS #########

######1D######
objective_function = x -> 2*x+1
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]
p = 1.8
a = 2
b = 6
my_k = Kriging(x,y,p)
surrogate_optimize(objective_function,LCBS(),a,b,my_k,UniformSample())


##### ND #####
objective_function_ND = z -> 3*norm(z)+1
x = [(1.2,3.0),(3.0,3.5),(5.2,5.7)]
y = objective_function_ND.(x)
p = [1.2,1.2]
theta = [2.0,2.0]
lb = [1.0,1.0]
ub = [6.0,6.0]

#Kriging
my_k_ND = Kriging(x,y,p,theta)
surrogate_optimize(objective_function_ND,LCBS(),lb,ub,my_k_ND,UniformSample())


##### EI ######

###1D###

objective_function = x -> 2*x+1
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]
p = 2
a = 2
b = 6
my_k = Kriging(x,y,p)
surrogate_optimize(objective_function,EI(),a,b,my_k,UniformSample(),maxiters=200,num_new_samples=155)


###ND###

objective_function_ND = z -> 3*norm(z)+1
x = [(1.2,3.0),(3.0,3.5),(5.2,5.7)]
y = objective_function_ND.(x)
p = [1.2,1.2]
theta = [2.0,2.0]
lb = [1.0,1.0]
ub = [6.0,6.0]

#Kriging
my_k_ND = Kriging(x,y,p,theta)
surrogate_optimize(objective_function_ND,EI(),lb,ub,my_k_ND,UniformSample())

=#


## DYCORS ##

#1D#
objective_function = x -> 2*x+1
x = [2.5,4.0,6.0]
y = [6.0,9.0,13.0]
p = 1.99
a = 2
b = 6
my_k = Kriging(x,y,p)
surrogate_optimize(objective_function_ND,DYCORS(),lb,ub,my_k_ND,UniformSample())
