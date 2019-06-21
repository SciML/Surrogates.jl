using Test
using Surrogates

@testset "Radials.jl" begin include("Radials.jl") end
@testset "Kriging.jl" begin include("Kriging.jl") end
@testset "Sampling" begin include("sampling.jl") end
@testset "Optimization" begin include("optimization.jl") end
