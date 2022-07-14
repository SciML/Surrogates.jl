using Surrogates
using QuasiMonteCarlo
using QuasiMonteCarlo: KroneckerSample, SectionSample, GoldenSample
using Distributions: Cauchy, Normal
using Test

#1D
lb = 0.0
ub = 5.0
n = 5
d = 1

## Sampling methods from QuasiMonteCarlo.jl ##

# GridSample
s = Surrogates.sample(n, lb, ub, GridSample(0.1))
@test s isa Vector{Float64} && length(s) == n && all(x -> lb ≤ x ≤ ub, s)

# UniformSample
s = Surrogates.sample(n, lb, ub, UniformSample())
@test s isa Vector{Float64} && length(s) == n && all(x -> lb ≤ x ≤ ub, s)

# SobolSample
s = Surrogates.sample(n, lb, ub, SobolSample())
@test s isa Vector{Float64} && length(s) == n && all(x -> lb ≤ x ≤ ub, s)

# LatinHypercubeSample
s = Surrogates.sample(n, lb, ub, LatinHypercubeSample())
@test s isa Vector{Float64} && length(s) == n && all(x -> lb ≤ x ≤ ub, s)

# LowDiscrepancySample
s = Surrogates.sample(20, lb, ub, LowDiscrepancySample(10))
@test s isa Vector{Float64} && length(s) == 20 && all(x -> lb ≤ x ≤ ub, s)

# LatticeRuleSample (not originally in Surrogates.jl, now available through QuasiMonteCarlo.jl)
s = Surrogates.sample(20, lb, ub, LatticeRuleSample())
@test s isa Vector{Float64} && length(s) == 20 && all(x -> lb ≤ x ≤ ub, s)

# Distribution sampling (Cauchy)
s = Surrogates.sample(n, d, Cauchy())
@test s isa Vector{Float64} && length(s) == n

# Distributions sampling (Normal)
s = Surrogates.sample(n, d, Normal(0, 4))
@test s isa Vector{Float64} && length(s) == n

## Sampling methods specific to Surrogates.jl ##

# KroneckerSample
s = Surrogates.sample(n, lb, ub, KroneckerSample(sqrt(2), 0))
@test s isa Vector{Float64} && length(s) == n && all(x -> lb ≤ x ≤ ub, s)

# GoldenSample
s = Surrogates.sample(n, lb, ub, GoldenSample())
@test s isa Vector{Float64} && length(s) == n && all(x -> lb ≤ x ≤ ub, s)

# SectionSample
constrained_val = 1.0
s = Surrogates.sample(n, lb, ub, SectionSample([NaN64], UniformSample()))
@test s isa Vector{Float64} && length(s) == n && all(x -> lb ≤ x ≤ ub, s)

s = Surrogates.sample(n, lb, ub, SectionSample([constrained_val], UniformSample()))
@test s isa Vector{Float64} && length(s) == n && all(x -> lb ≤ x ≤ ub, s)
@test all(==(constrained_val), s)

# ND but 1D

lb = [0.0]
ub = [5.0]
s = Surrogates.sample(n, lb, ub, SobolSample())
@test s isa Vector{Float64} && length(s) == n && all(x -> lb[1] ≤ x ≤ ub[1], s)

# ND
# Now that we use QuasiMonteCarlo.jl, these tests are to make sure that we transform the output
# from a Matrix to a Vector of Tuples properly for ND problems.

lb = [0.1, -0.5]
ub = [1.0, 20.0]
n = 5
d = 2

#GridSample{T}
s = Surrogates.sample(n, lb, ub, GridSample([0.1, 0.5]))
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true

#UniformSample()
s = Surrogates.sample(n, lb, ub, UniformSample())
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true

#SobolSample()
s = Surrogates.sample(n, lb, ub, SobolSample())
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true

#LHS
s = Surrogates.sample(n, lb, ub, LatinHypercubeSample())
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true

#LDS
s = Surrogates.sample(n, lb, ub, LowDiscrepancySample([10, 3]))
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true

#Distribution 1
s = Surrogates.sample(n, d, Cauchy())
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true

#Distribution 2
s = Surrogates.sample(n, d, Normal(3, 5))
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true

#Kronecker
s = Surrogates.sample(n, lb, ub, KroneckerSample([sqrt(2), 3.1415], [0, 0]))
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true

#Golden
s = Surrogates.sample(n, lb, ub, GoldenSample())
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true

# SectionSample
constrained_val = 1.0
s = Surrogates.sample(n, lb, ub, SectionSample([NaN64, constrained_val], UniformSample()))
@test all(x -> x[end] == constrained_val, s)
@test isa(s, Array{Tuple{typeof(s[1][1]), typeof(s[1][1])}, 1}) == true
@test all(x -> lb[1] ≤ x[1] ≤ ub[1], s)
