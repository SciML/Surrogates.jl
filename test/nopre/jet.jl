using Surrogates
using JET
using Test

@testset "JET static analysis" begin
    # Test 1D surrogates
    x1d = [1.0, 2.0, 3.0, 4.0, 5.0]
    y1d = [1.0, 4.0, 9.0, 16.0, 25.0]
    lb1d = 0.0
    ub1d = 6.0

    # Use target_modules to focus only on Surrogates code and avoid
    # false positives from external dependencies (e.g., LinearAlgebra on Julia 1.10)
    jet_kwargs = (; target_modules = (Surrogates,))

    @testset "LinearSurrogate" begin
        rep = JET.report_call(
            LinearSurrogate,
            (typeof(x1d), typeof(y1d), Float64, Float64); jet_kwargs...
        )
        @test isempty(JET.get_reports(rep))
    end

    @testset "RadialBasis" begin
        rep = JET.report_call(
            RadialBasis,
            (typeof(x1d), typeof(y1d), Float64, Float64); jet_kwargs...
        )
        @test isempty(JET.get_reports(rep))
    end

    @testset "Kriging" begin
        rep = JET.report_call(
            Kriging,
            (typeof(x1d), typeof(y1d), Float64, Float64); jet_kwargs...
        )
        @test isempty(JET.get_reports(rep))
    end

    @testset "InverseDistanceSurrogate" begin
        rep = JET.report_call(
            InverseDistanceSurrogate,
            (typeof(x1d), typeof(y1d), Float64, Float64); jet_kwargs...
        )
        @test isempty(JET.get_reports(rep))
    end

    @testset "SecondOrderPolynomialSurrogate" begin
        rep = JET.report_call(
            SecondOrderPolynomialSurrogate,
            (typeof(x1d), typeof(y1d), Float64, Float64); jet_kwargs...
        )
        @test isempty(JET.get_reports(rep))
    end

    @testset "LobachevskySurrogate" begin
        rep = JET.report_call(
            LobachevskySurrogate,
            (typeof(x1d), typeof(y1d), Float64, Float64); jet_kwargs...
        )
        @test isempty(JET.get_reports(rep))
    end
end
