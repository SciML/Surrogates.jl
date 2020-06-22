using Surrogates

#1D MOE
n = 4
lb = 0.0
ub = 5.0
x = sample(n,lb,ub,SobolSample())
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
