using Surrogates

#=
#1D MOE
n = 30
lb = 0.0
ub = 5.0
x = Surrogates.sample(n,lb,ub,RandomSample())
f = x-> 2*x
y = f.(x)
#Standard definition
my_moe = MOE(x,y,lb,ub)
val = my_moe(3.0)

#Local surrogates redefinition
my_local_kind = [InverseDistanceStructure(p = 1.0),
                 KrigingStructure(p=1.0, theta=1.0)]
my_moe = MOE(x,y,lb,ub,k = 2,local_kind = my_local_kind)

#ND MOE
n = 30
lb = [0.0,0.0]
ub = [5.0,5.0]
x = sample(n,lb,ub,LatinHypercubeSample())
f = x -> x[1]*x[2]
y = f.(x)
my_moe_ND = MOE(x,y,lb,ub)
val = my_moe_ND((1.0,1.0))

#Local surr redefinition
my_locals = [InverseDistanceStructure(p = 1.0),
             RandomForestStructure(num_round=1)]
my_moe_redef = MOE(x,y,lb,ub,k = 2,local_kind = my_locals)
=#
