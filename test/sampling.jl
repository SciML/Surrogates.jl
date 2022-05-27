using Surrogates
using Distributions
using Test

#1D
lb = 0.0
ub = 5.0
n = 5
d = 1
Surrogates.sample(n,lb,ub,GridSample(0.1))
Surrogates.sample(n,lb,ub,UniformSample())
Surrogates.sample(n,lb,ub,SobolSample())
Surrogates.sample(n,lb,ub,LatinHypercubeSample())
Surrogates.sample(20,lb,ub,LowDiscrepancySample(10))
Surrogates.sample(5,d,Cauchy())
Surrogates.sample(5,d,Normal(0,4))
Surrogates.sample(5,lb,ub,KroneckerSample(sqrt(2),0))
Surrogates.sample(5,lb,ub,GoldenSample())
#ND
lb = [0.1,-0.5]
ub = [1.0,20.0]
n = 5
d = 2

#GridSample{T}
s = Surrogates.sample(n,lb,ub,GridSample([0.1,0.5]))
@test isa(s,Array{Tuple{typeof(s[1][1]),typeof(s[1][1])},1}) == true

#UniformSample()
s = Surrogates.sample(n,lb,ub,UniformSample())
@test isa(s,Array{Tuple{typeof(s[1][1]),typeof(s[1][1])},1}) == true

#SobolSample()
s = Surrogates.sample(n,lb,ub,SobolSample())
@test isa(s,Array{Tuple{typeof(s[1][1]),typeof(s[1][1])},1}) == true

#LHS
s = Surrogates.sample(n,lb,ub,LatinHypercubeSample())
@test isa(s,Array{Tuple{typeof(s[1][1]),typeof(s[1][1])},1}) == true

#LDS
s = Surrogates.sample(n,lb,ub,LowDiscrepancySample([10,3]))
@test isa(s,Array{Tuple{typeof(s[1][1]),typeof(s[1][1])},1}) == true

#Distribution 1
s = Surrogates.sample(n,d,Cauchy())
@test isa(s,Array{Tuple{typeof(s[1][1]),typeof(s[1][1])},1}) == true

#Distribution 2
s = Surrogates.sample(n,d,Normal(3,5))
@test isa(s,Array{Tuple{typeof(s[1][1]),typeof(s[1][1])},1}) == true

#Kronecker
Surrogates.sample(5,lb,ub,KroneckerSample([sqrt(2),3.1415],[0,0]))

#Golden
Surrogates.sample(5,lb,ub,GoldenSample())

# SectionSample on constrained space
constrained_val = 1.0
s = Surrogates.sample(n, lb, ub, SectionSample([NaN64, constrained_val], UniformSample()))
@test all(x -> x[end] == constrained_val, s)
@test isa(s,Array{Tuple{typeof(s[1][1]),typeof(s[1][1])},1}) == true
@test all(x -> lb[1] ≤ x[1] ≤ ub[1], s)
