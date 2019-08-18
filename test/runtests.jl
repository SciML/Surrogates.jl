using Test
using Surrogates


@testset "Radials.jl" begin include("radials.jl") end
@testset "Kriging.jl" begin include("kriging.jl") end
@testset "Sampling" begin include("sampling.jl") end
@testset "Optimization" begin include("optimization.jl") end
@testset "LinearSurrogate" begin include("linearSurrogate.jl") end
@testset "Lobachesky" begin include("lobachesky.jl") end
@testset "RandomForestSurrogate" begin include("random_forest.jl") end
@testset "SVMSurrogate" begin include("SVMSurrogate.jl") end
@testset "NeuralSurrogate" begin include("neuralSurrogate.jl") end
@testset "InverseDistanceSurrogate" begin include("inverseDistanceSurrogate.jl") end
@testset "SecondOrderPolynomialSurrogate" begin include("secondOrderPolynomialSurrogate.jl") end
