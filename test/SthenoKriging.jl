using Surrogates
using Stheno
using Zygote
using Test

@testset "1D - 1D" begin

lb = 0.0
ub = 10.0
f = x -> log(x)*exp(x)
x = sample(5,lb,ub,SobolSample())
y = f.(x)
my_k = SthenoKriging(x,y)
add_point!(my_k,4.0,75.68)
add_point!(my_k,[5.0,6.0],[238.86,722.84])
pred = my_k(5.5)
