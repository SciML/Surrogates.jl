using Surrogates
using Test
using SafeTestsets
using Pkg

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @safetestset "Quality Assurance" begin
        include("qa/qa.jl")
    end
end

if GROUP == "All" || GROUP == "Core"
    @testset "Surrogates" begin
        @testset "Extensions" begin
            include("extensions.jl")
        end
        @testset "Algorithms" begin
            @time @safetestset "GEKPLS" begin
                include("GEKPLS.jl")
            end
            @time @safetestset "Radials" begin
                include("Radials.jl")
            end
            @time @safetestset "Kriging" begin
                include("Kriging.jl")
            end
            @time @safetestset "Sampling" begin
                include("sampling.jl")
            end
            @time @safetestset "Optimization" begin
                include("optimization.jl")
            end
            @time @safetestset "LinearSurrogate" begin
                include("linearSurrogate.jl")
            end
            @time @safetestset "Lobachevsky" begin
                include("lobachevsky.jl")
            end
            @time @safetestset "InverseDistanceSurrogate" begin
                include("inverseDistanceSurrogate.jl")
            end
            @time @safetestset "SecondOrderPolynomialSurrogate" begin
                include("secondOrderPolynomialSurrogate.jl")
            end
            @time @safetestset "Wendland" begin
                include("Wendland.jl")
            end
            @time @safetestset "VariableFidelity" begin
                include("VariableFidelity.jl")
            end
            @time @safetestset "Earth" begin
                include("earth.jl")
            end
            @time @safetestset "Gradient Enhanced Kriging" begin
                include("GEK.jl")
            end
            @time @safetestset "Section Samplers" begin
                include("SectionSampleTests.jl")
            end
        end
        @time @safetestset "AD" begin
            include("AD_compatibility.jl")
        end
        @time @safetestset "Interface Compatibility" begin
            include("interface_tests.jl")
        end
    end
end
