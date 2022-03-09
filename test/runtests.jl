
using Surrogates
using Test
using SafeTestsets

@time @safetestset  "Radials.jl" begin include("Radials.jl") end
@time @safetestset  "Kriging.jl" begin include("Kriging.jl") end
@time @safetestset  "Sampling" begin include("sampling.jl") end
@time @safetestset  "Optimization" begin include("optimization.jl") end
@time @safetestset  "LinearSurrogate" begin include("linearSurrogate.jl") end
@time @safetestset  "Lobachevsky" begin include("lobachevsky.jl") end
@time @safetestset  "RandomForestSurrogate" begin include("random_forest.jl") end
#@time @safetestset  "SVMSurrogate" begin include("SVMSurrogate.jl") end
@time @safetestset  "NeuralSurrogate" begin include("neuralSurrogate.jl") end
@time @safetestset  "InverseDistanceSurrogate" begin include("inverseDistanceSurrogate.jl") end
@time @safetestset  "SecondOrderPolynomialSurrogate" begin include("secondOrderPolynomialSurrogate.jl") end
@time @safetestset  "AD_Compatibility" begin include("AD_compatibility.jl") end
@time @safetestset "AD_Compatibility" begin include("AD_compatibility.jl") end
@time @safetestset  "Wendland" begin include("Wendland.jl") end
#@time @safetestset  "MOE" begin include("MOE.jl") end write em algorithm to get rid fo deps
@time @safetestset  "VariableFidelity" begin include("VariableFidelity.jl") end
@time @safetestset  "Earth" begin include("earth.jl") end
@time @safetestset  "Gradient Enhanced Kriging" begin include("GEK.jl") end
@time @safetestset  "Section Samplers" begin include("SectionSampleTests.jl") end
@time @safetestset  "SthenoKriging.jl" begin include("SthenoKriging.jl") end #throws error