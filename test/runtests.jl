using Test
using Surrogates
println("Starting tests")

@testset "Radials_1D.jl" begin
    include("radials_1D.jl")
       end;

@testset "Kriging_1D.jl" begin
   include("kriging_1D.jl")
      end;
