using Surrogates
using Distributions
#1D MOE

n = 4
lb = 0.0
ub = 5.0
x1 = Surrogates.sample(n,1,Normal(2.5,0.1))
x2 = Surrogates.sample(n,1,Normal(1.0,0.4))
x = vcat(x1,x2)
f = x-> 2*x
y = f.(x)
#Standard definition
my_moe = MOE(x,y,lb,ub)
val = my_moe(3.0)
add_point!(my_moe,3.0,6.0)
add_point!(my_moe,[4.0,5.0],[8.0,10.0])

#Local surrogates redefinition
my_local_kind = [InverseDistanceStructure(p = 1.0),
                 SecondOrderPolynomialStructure()]
my_moe = MOE(x,y,lb,ub,k = 2,local_kind = my_local_kind)



#ND MOE

n = 10
lb = [0.0,0.0]
ub = [5.0,5.0]
x1 = Surrogates.sample(n,2,Normal(0.0,4.0))
x2 = Surrogates.sample(n,2,Normal(3.0,5.0))
x = vcat(x1,x2)
f = x -> x[1]*x[2]
y = f.(x)
my_moe_ND = MOE(x,y,lb,ub)
val = my_moe_ND((1.0,1.0))

add_point!(my_moe_ND, (1.0,1.0), 1.0)

#Local surr redefinition
my_locals = [InverseDistanceStructure(p = 2.0),
             SecondOrderPolynomialStructure()]
my_moe_redef = MOE(x,y,lb,ub,k = 2,local_kind = my_local_kind)
