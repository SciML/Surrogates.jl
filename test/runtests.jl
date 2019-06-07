using Test
using Surrogates
println("Starting tests")

@testset "Radials.jl" begin
    include("Radials.jl")
       end;

@testset "Kriging.jl" begin
   include("Kriging.jl")
      end;
