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

        @testset "LinearSurrogate 1D" begin
            lin = LinearSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = lin(test_point)
            @test result isa BigFloat
            # No Float64 contamination in the fitted coefficients
            @test lin.coeff isa Vector{BigFloat}
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
            @test std_error_at_point(k, test_point) isa BigFloat
        end
    end

    @testset "BigFloat Support - ND Surrogates" begin
        # Test data with BigFloat for N-dimensional
        x_bf = [
            (BigFloat(1.0), BigFloat(2.0)), (BigFloat(2.0), BigFloat(3.0)),
            (BigFloat(3.0), BigFloat(1.0)), (BigFloat(4.0), BigFloat(4.0)),
            (BigFloat(5.0), BigFloat(2.0)), (BigFloat(2.5), BigFloat(4.5)),
        ]
        y_bf = BigFloat[0.5, 1.2, 2.1, 2.8, 3.5, 1.9]
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

        @testset "LinearSurrogate ND" begin
            lin = LinearSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = lin(test_point)
            @test result isa BigFloat
            # No Float64 contamination in the fitted coefficients
            @test lin.coeff isa Vector{BigFloat}
        end

        @testset "Kriging ND" begin
            k = Kriging(x_bf, y_bf, lb_bf, ub_bf)
            result = k(test_point)
            @test result isa BigFloat
            @test std_error_at_point(k, test_point) isa BigFloat
        end
    end

    @testset "Float32 Support" begin
        # Float32 is the other end of the genericity question from BigFloat:
        # silent promotion to Float64 is easy to introduce and invisible unless
        # the element type is asserted.
        x32 = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
        y32 = Float32[0.5, 1.2, 2.1, 2.8, 3.6]
        lb32 = 0.0f0
        ub32 = 6.0f0
        p32 = 2.5f0

        @testset "1D" begin
            for (name, build) in (
                    ("RadialBasis", (x, y, l, u) -> RadialBasis(x, y, l, u)),
                    (
                        "InverseDistanceSurrogate",
                        (x, y, l, u) -> InverseDistanceSurrogate(x, y, l, u),
                    ),
                    (
                        "LobachevskySurrogate",
                        (x, y, l, u) -> LobachevskySurrogate(x, y, l, u),
                    ),
                    (
                        "SecondOrderPolynomialSurrogate",
                        (x, y, l, u) -> SecondOrderPolynomialSurrogate(x, y, l, u),
                    ),
                    ("LinearSurrogate", (x, y, l, u) -> LinearSurrogate(x, y, l, u)),
                    ("Wendland", (x, y, l, u) -> Wendland(x, y, l, u)),
                    ("Kriging", (x, y, l, u) -> Kriging(x, y, l, u)),
                )
                @testset "$name" begin
                    s = build(copy(x32), copy(y32), lb32, ub32)
                    @test isfinite(s(p32))
                end
            end
        end

        @testset "ND" begin
            xnd = Tuple{Float32, Float32}[
                (1.0f0, 1.0f0), (2.0f0, 3.0f0), (3.0f0, 2.0f0),
                (4.0f0, 5.0f0), (5.0f0, 4.0f0), (2.5f0, 4.5f0),
            ]
            ynd = Float32[0.5, 1.2, 2.1, 2.8, 3.6, 3.0]
            lbnd = Float32[0.0, 0.0]
            ubnd = Float32[6.0, 6.0]
            pnd = (2.5f0, 2.5f0)
            for (name, build) in (
                    ("RadialBasis", (x, y, l, u) -> RadialBasis(x, y, l, u)),
                    (
                        "InverseDistanceSurrogate",
                        (x, y, l, u) -> InverseDistanceSurrogate(x, y, l, u),
                    ),
                    (
                        "SecondOrderPolynomialSurrogate",
                        (x, y, l, u) -> SecondOrderPolynomialSurrogate(x, y, l, u),
                    ),
                    ("LinearSurrogate", (x, y, l, u) -> LinearSurrogate(x, y, l, u)),
                    ("Wendland", (x, y, l, u) -> Wendland(x, y, l, u)),
                    ("Kriging", (x, y, l, u) -> Kriging(x, y, l, u)),
                )
                @testset "$name" begin
                    s = build(copy(xnd), copy(ynd), lbnd, ubnd)
                    @test isfinite(s(pnd))
                end
            end
        end
    end

end
