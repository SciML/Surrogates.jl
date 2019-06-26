using Surrogates
using LinearAlgebra


#######SRBF############

##### 1D #####
objective_function = x -> 2*x+1
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]

# In 1D values of p closer to 2 make the det(R) closer and closer to 0,
#this does not happen in higher dimensions because p would be a vector and not
#all components are generally C^inf
p = 1.0
a = 2
b = 6

#Using Kriging
my_k = Kriging(x,y,p)
SRBF(a,b,my_k,10,UniformSample(),10,objective_function)


#Using RadialBasis

my_rad = RadialBasis(x,y,a,b,z->norm(z),1)
SRBF(a,b,my_rad,10,UniformSample(),10,objective_function)



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
SRBF(lb,ub,my_k_ND,10,UniformSample(),10,objective_function_ND)

#Radials
bounds = [[1.0,6.0],[1.0,6.0]]
my_rad_ND = RadialBasis(x,y,bounds,z->norm(z),1)
SRBF(lb,ub,my_rad_ND,10,UniformSample(),10,objective_function_ND)


####### LCBS #########


######1D######
objective_function = x -> 2*x+1
x = [2.0,4.0,6.0]
y = [5.0,9.0,13.0]
p = 2
a = 2
b = 6
my_k = Kriging(x,y,p)
LCBS(a,b,my_k,10,SobolSample(),10,objective_function)



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
LCBS(lb,ub,my_k_ND,10,UniformSample(),10,objective_function_ND)
