using Surrogates

#1D MOE
n = 20
lb = 0.0
ub = 5.0
x = sample(n,lb,ub,SobolSample())
f = x-> 2*x
y = f.(x)
my_moe = MOE(x,y,lb,ub)
