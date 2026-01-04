# Interface compatibility tests for Surrogates.jl
# Tests BigFloat support for SciML array/number interface compliance

using Test
using Surrogates

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

        # Note: Kriging with BigFloat fails because eigvals() doesn't support
        # arbitrary precision types (requires LAPACK). This is a known limitation.
        @testset "Kriging 1D (known limitation)" begin
            @test_broken begin
                k = Kriging(x_bf, y_bf, lb_bf, ub_bf)
                result = k(test_point)
                result isa BigFloat
            end
        end
    end

    @testset "BigFloat Support - ND Surrogates" begin
        # Test data with BigFloat for N-dimensional
        x_bf = [
            (BigFloat(1.0), BigFloat(2.0)), (BigFloat(2.0), BigFloat(3.0)),
            (BigFloat(3.0), BigFloat(1.0)), (BigFloat(4.0), BigFloat(4.0)),
            (BigFloat(5.0), BigFloat(2.0)),
        ]
        y_bf = BigFloat[0.5, 1.2, 2.1, 2.8, 3.5]
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

        # Note: Kriging with BigFloat fails because eigvals() doesn't support
        # arbitrary precision types (requires LAPACK). This is a known limitation.
        @testset "Kriging ND (known limitation)" begin
            @test_broken begin
                k = Kriging(x_bf, y_bf, lb_bf, ub_bf)
                result = k(test_point)
                result isa BigFloat
            end
        end
    end
end
