using Test
using Surrogates

@testset "Radials.jl" begin include("Radials.jl") end
@testset "Kriging.jl" begin include("Kriging.jl") end
@testset "Sampling" begin include("sampling.jl") end
@testset "Optimization" begin include("optimization.jl") end
@testset "LinearSurrogate" begin include("linearSurrogate.jl") end
@testset "Lobachesky" begin include("lobachesky.jl") end
@testset "RandomForestSurrogate" begin include("random_forest.jl") end
#@testset "SVMSurrogate" begin include("SVMSurrogate.jl") end
@testset "NeuralSurrogate" begin include("neuralSurrogate.jl") end
@testset "InverseDistanceSurrogate" begin include("inverseDistanceSurrogate.jl") end
@testset "SecondOrderPolynomialSurrogate" begin include("secondOrderPolynomialSurrogate.jl") end
@testset "AD_Compatibility" begin include("AD_compatibility.jl") end
@testset "SthenoKriging.jl" begin include("SthenoKriging.jl") end
@testset "Wendland" begin include("Wendland.jl") end
@testset "MOE" begin include("MOE.jl") end
@testset "VariableFidelity" begin include("VariableFidelity.jl") end
