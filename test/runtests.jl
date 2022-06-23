using Surrogates
using Test
using SafeTestsets
using Pkg

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.develop(PackageSpec(path = subpkg_path))
end
for pkg in ["SurrogatesAbstractGPs", "SurrogatesFlux", "SurrogatesPolyChaos",
    "SurrogatesRandomForest", "SurrogatesSVM"]
    @time begin
        dev_subpkg(pkg)
        Pkg.test(pkg)
    end
end

@time @safetestset "Radials.jl" begin include("Radials.jl") end
@time @safetestset "Kriging.jl" begin include("Kriging.jl") end
@time @safetestset "Sampling" begin include("sampling.jl") end
@time @safetestset "Optimization" begin include("optimization.jl") end
@time @safetestset "LinearSurrogate" begin include("linearSurrogate.jl") end
@time @safetestset "Lobachevsky" begin include("lobachevsky.jl") end
@time @safetestset "InverseDistanceSurrogate" begin include("inverseDistanceSurrogate.jl") end
@time @safetestset "SecondOrderPolynomialSurrogate" begin include("secondOrderPolynomialSurrogate.jl") end
# @time @safetestset  "AD_Compatibility" begin include("AD_compatibility.jl") end
@time @safetestset "Wendland" begin include("Wendland.jl") end
@time @safetestset "VariableFidelity" begin include("VariableFidelity.jl") end
@time @safetestset "Earth" begin include("earth.jl") end
@time @safetestset "Gradient Enhanced Kriging" begin include("GEK.jl") end
@time @safetestset "Section Samplers" begin include("SectionSampleTests.jl") end
