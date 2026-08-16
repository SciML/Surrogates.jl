using Surrogates
using Test

@testset "SecondOrderPolynomialSurrogate" begin
    @testset "underdetermined fits are rejected" begin
        # 1D needs 3 samples (intercept, x, x^2)
        @test_throws ArgumentError SecondOrderPolynomialSurrogate(
            [1.0, 2.0], [1.0, 4.0], 0.0, 5.0
        )
        # 2D needs 6 samples (1, x1, x2, x1*x2, x1^2, x2^2)
        x_few = [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 2.0), (2.0, 3.0)]
        y_few = ones(5)
        @test_throws ArgumentError SecondOrderPolynomialSurrogate(
            x_few, y_few, [0.0, 0.0], [5.0, 5.0]
        )

        # The count is necessary but not sufficient: a degenerate sample with
        # enough points still leaves the design rank deficient, and the
        # least-squares solve returns a minimum-norm fit without raising.
        # Documented behaviour, asserted so a future change is deliberate.
        x_collinear = [(t, 2t) for t in 1.0:6.0]
        y_collinear = [p[1]^2 for p in x_collinear]
        s = SecondOrderPolynomialSurrogate(
            x_collinear, y_collinear, [0.0, 0.0], [10.0, 20.0]
        )
        @test s((3.0, 6.0)) isa Number
    end

    @testset "1D quadratic recovery" begin
        # Quadratic data must be reproduced exactly, including after updates.
        q = x -> 1.0 + 2.0 * x + 3.0 * x^2
        lb = 0.0
        ub = 5.0
        x = sample(5, lb, ub, SobolSample())
        y = q.(x)
        my_poly = SecondOrderPolynomialSurrogate(x, y, lb, ub)
        @test my_poly.β ≈ [1.0, 2.0, 3.0]
        @test my_poly(2.7) ≈ q(2.7)
        update!(my_poly, 5.0, q(5.0))
        update!(my_poly, [6.0, 7.0], q.([6.0, 7.0]))
        @test my_poly.β ≈ [1.0, 2.0, 3.0]
        @test my_poly(6.5) ≈ q(6.5)

        # The minimum admissible sample count (3 points, 3 coefficients)
        # interpolates exactly.
        x3 = [0.0, 1.0, 3.0]
        my_poly_min = SecondOrderPolynomialSurrogate(x3, q.(x3), lb, ub)
        @test my_poly_min.(x3) ≈ q.(x3)
        @test my_poly_min.β ≈ [1.0, 2.0, 3.0]

        @test_throws ArgumentError my_poly(Float64[])
        @test_throws ArgumentError my_poly((2.0, 3.0, 4.0))
    end

    @testset "1D least-squares fit of non-polynomial data" begin
        lb = 0.0
        ub = 5.0
        obj_1D = x -> log(x) * exp(x)
        x = sample(5, lb, ub, SobolSample())
        y = obj_1D.(x)
        my_second_order_poly = SecondOrderPolynomialSurrogate(x, y, lb, ub)
        # Residuals of a least-squares fit are orthogonal to every column of
        # the design matrix; check intercept, x, and x^2 columns.
        r = y .- my_second_order_poly.(x)
        for col in (one.(x), x, x .^ 2)
            @test isapprox(sum(r .* col), 0.0; atol = 1.0e-8 * maximum(abs, y))
        end
        update!(my_second_order_poly, 5.0, 238.86)
        update!(my_second_order_poly, [6.0, 7.0], [722.84, 2133.94])
        @test length(my_second_order_poly.x) == 8

        @test_throws ArgumentError my_second_order_poly(Float64[])
        @test_throws ArgumentError my_second_order_poly((2.0, 3.0, 4.0))
    end

    @testset "ND quadratic recovery" begin
        # Full 2D quadratic with cross term must be reproduced exactly.
        q = x -> 0.3 + 0.7 * x[1] + 0.1 * x[2] + 0.8 * x[1] * x[2] +
            0.3 * x[1]^2 + 0.1 * x[2]^2
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        x = sample(10, lb, ub, SobolSample())
        y = q.(x)
        my_poly_ND = SecondOrderPolynomialSurrogate(x, y, lb, ub)
        @test my_poly_ND.β ≈ [0.3, 0.7, 0.1, 0.8, 0.3, 0.1]
        @test my_poly_ND((5.0, 7.0)) ≈ q((5.0, 7.0))
        update!(my_poly_ND, (5.0, 7.0), q((5.0, 7.0)))
        update!(my_poly_ND, [(1.5, 1.5), (3.4, 5.4)], q.([(1.5, 1.5), (3.4, 5.4)]))
        @test my_poly_ND.β ≈ [0.3, 0.7, 0.1, 0.8, 0.3, 0.1]
        @test my_poly_ND((2.0, 9.0)) ≈ q((2.0, 9.0))

        @test_throws ArgumentError my_poly_ND(Float64[])
        @test_throws ArgumentError my_poly_ND(2.0)
        @test_throws ArgumentError my_poly_ND((2.0, 3.0, 4.0))
    end

    @testset "multi-output" begin
        f = x -> [x^2, x]
        lb = 1.0
        ub = 10.0
        x = sample(5, lb, ub, SobolSample())
        push!(x, 2.0)
        y = f.(x)
        surrogate = SecondOrderPolynomialSurrogate(x, y, lb, ub)
        # should be exact
        @test surrogate.β ≈ [0 0; 0 1; 1 0]
        @test surrogate(2.0) ≈ [4, 2]
        @test surrogate(1.0) ≈ [1, 1]

        f = x -> [x[1], x[2]^2]
        lb = [1.0, 2.0]
        ub = [10.0, 8.5]
        x = sample(20, lb, ub, SobolSample())
        push!(x, (1.0, 2.0))
        y = f.(x)
        surrogate = SecondOrderPolynomialSurrogate(x, y, lb, ub)
        @test surrogate.β ≈ [0 0; 1 0; 0 0; 0 0; 0 0; 0 1]
        @test surrogate((1.0, 2.0)) ≈ [1, 4]
        x_new = (2.0, 2.0)
        y_new = f(x_new)
        @test surrogate(x_new) ≈ y_new
        update!(surrogate, x_new, y_new)
        @test surrogate(x_new) ≈ y_new
    end

    @testset "recovers a second-order polynomial (matrix form)" begin
        function second_order_target(x; a = 0.3, b = [0.7, 0.1], c = [0.3 0.4; 0.4 0.1])
            return a + b' * x + x' * c * x
        end
        second_order_target(x::Tuple; kwargs...) = second_order_target([x...]; kwargs...)
        lb = fill(-5.0, 2)
        ub = fill(5.0, 2)
        x = sample(30, lb, ub, SobolSample())
        y = second_order_target.(x)
        sec = SecondOrderPolynomialSurrogate(x, y, lb, ub)
        @test y ≈ sec.(x)
        # β must match the analytic coefficients: cross term 2 * c[1, 2]
        @test sec.β ≈ [0.3, 0.7, 0.1, 0.8, 0.3, 0.1]
    end
end
