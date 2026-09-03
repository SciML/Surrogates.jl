# Interface compatibility tests for Surrogates.jl
# Tests BigFloat support for SciML array/number interface compliance

using Test
using Surrogates
using SurrogatesBase

mutable struct InterfaceContractDeterministic{T} <: SurrogatesBase.AbstractDeterministicSurrogate
    x::Vector{T}
    y::Vector{T}
end

mutable struct InterfaceContractStochastic{T} <: SurrogatesBase.AbstractStochasticSurrogate
    x::Vector{T}
    y::Vector{T}
end

(s::InterfaceContractDeterministic)(point) = point^2
(s::InterfaceContractStochastic)(point) = point^2

function SurrogatesBase.update!(
        s::Union{
            InterfaceContractDeterministic, InterfaceContractStochastic,
        }, x_new, y_new
    )
    if x_new isa AbstractVector && y_new isa AbstractVector
        append!(s.x, x_new)
        append!(s.y, y_new)
    else
        push!(s.x, x_new)
        push!(s.y, y_new)
    end
    return nothing
end

Surrogates.std_error_at_point(::InterfaceContractStochastic, point) = abs(point)
Surrogates.logpdf_surrogate(::InterfaceContractStochastic) = -0.5

@testset "Generic surrogate contracts" begin
    @test SRBF() isa Surrogates.SurrogateOptimizationAlgorithm
    @test MeanConstantLiar() isa Surrogates.ParallelStrategy

    deterministic = InterfaceContractDeterministic([0.0], [0.0])
    @test deterministic isa Surrogates.AbstractSurrogate
    @test deterministic(2.0) == 4.0
    @test update!(deterministic, 2.0, 4.0) === nothing
    @test deterministic.x == [0.0, 2.0]
    @test deterministic.y == [0.0, 4.0]

    stochastic = InterfaceContractStochastic([0.0], [0.0])
    @test stochastic isa Surrogates.AbstractSurrogate
    @test stochastic(2.0) == 4.0
    @test std_error_at_point(stochastic, 2.0) == 2.0
    @test logpdf_surrogate(stochastic) == -0.5
    @test update!(stochastic, [1.0, 2.0], [1.0, 4.0]) === nothing
    @test stochastic.x == [0.0, 1.0, 2.0]
    @test stochastic.y == [0.0, 1.0, 4.0]
end

@testset "Interface Compatibility" begin
    @testset "BigFloat Support - 1D Surrogates" begin
        # Test data with BigFloat
        x_bf = BigFloat[1.0, 2.0, 3.0, 4.0, 5.0]
        y_bf = BigFloat[0.5, 1.2, 2.1, 2.8, 3.6]
        lb_bf = BigFloat(0.0)
        ub_bf = BigFloat(6.0)
        test_point = BigFloat(2.5)

        @testset "RadialBasis 1D" begin
            rad = RadialBasis(x_bf, y_bf, lb_bf, ub_bf)
            result = rad(test_point)
            @test result isa BigFloat
        end

        @testset "InverseDistanceSurrogate 1D" begin
            ids = InverseDistanceSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = ids(test_point)
            @test result isa BigFloat
        end

        @testset "LobachevskySurrogate 1D" begin
            lob = LobachevskySurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = lob(test_point)
            @test result isa BigFloat
        end

        @testset "SecondOrderPolynomialSurrogate 1D" begin
            sop = SecondOrderPolynomialSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = sop(test_point)
            @test result isa BigFloat
        end

        @testset "Wendland 1D" begin
            wen = Wendland(x_bf, y_bf, lb_bf, ub_bf)
            result = wen(test_point)
            @test result isa BigFloat
        end

        @testset "Kriging 1D" begin
            k = Kriging(x_bf, y_bf, lb_bf, ub_bf)
            result = k(test_point)
            @test result isa BigFloat
        end
    end

    @testset "BigFloat Support - ND Surrogates" begin
        # Test data with BigFloat for N-dimensional
        # Six points, not five: a full quadratic in two dimensions has
        # 1 + 2d + d(d - 1) / 2 = 6 coefficients, so five samples leave
        # SecondOrderPolynomialSurrogate underdetermined.
        x_bf = [
            (BigFloat(1.0), BigFloat(2.0)), (BigFloat(2.0), BigFloat(3.0)),
            (BigFloat(3.0), BigFloat(1.0)), (BigFloat(4.0), BigFloat(4.0)),
            (BigFloat(5.0), BigFloat(2.0)), (BigFloat(2.0), BigFloat(5.0)),
        ]
        y_bf = BigFloat[0.5, 1.2, 2.1, 2.8, 3.5, 2.4]
        lb_bf = (BigFloat(0.0), BigFloat(0.0))
        ub_bf = (BigFloat(6.0), BigFloat(5.0))
        test_point = (BigFloat(2.5), BigFloat(2.5))

        @testset "RadialBasis ND" begin
            rad = RadialBasis(x_bf, y_bf, lb_bf, ub_bf)
            result = rad(test_point)
            @test result isa BigFloat
        end

        @testset "InverseDistanceSurrogate ND" begin
            ids = InverseDistanceSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = ids(test_point)
            @test result isa BigFloat
        end

        @testset "SecondOrderPolynomialSurrogate ND" begin
            sop = SecondOrderPolynomialSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = sop(test_point)
            @test result isa BigFloat
        end

        @testset "Kriging ND" begin
            k = Kriging(x_bf, y_bf, lb_bf, ub_bf)
            result = k(test_point)
            @test result isa BigFloat
        end
    end
end
