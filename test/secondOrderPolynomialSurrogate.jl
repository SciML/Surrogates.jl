using Surrogates
using Test
using ForwardDiff
using Zygote
using LinearAlgebra

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

        # Enough points but degenerate: collinear samples cannot determine a
        # full quadratic, whatever the count.
        x_collinear = [(t, 2t) for t in 1.0:6.0]
        y_collinear = [p[1]^2 for p in x_collinear]
        @test_throws ArgumentError SecondOrderPolynomialSurrogate(
            x_collinear, y_collinear, [0.0, 0.0], [10.0, 20.0]
        )
        # More points, still collinear: still rejected.
        x_more = [(t, 2t) for t in 1.0:20.0]
        @test_throws ArgumentError SecondOrderPolynomialSurrogate(
            x_more, [p[1]^2 for p in x_more], [0.0, 0.0], [10.0, 40.0]
        )
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

    # A design that determines a full 2D quadratic: eight points, no three of
    # them collinear, reused by the type and query testsets below.
    SPREAD_2D = [
        (1, 1), (2, 3), (4, 2), (3, 5), (5, 2), (2, 6), (6, 4), (4, 6),
    ]

    @testset "input containers" begin
        q = p -> 0.3 + 0.7p[1] + 0.1p[2] + 0.8p[1] * p[2] + 0.3p[1]^2 + 0.1p[2]^2
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        expected = [0.3, 0.7, 0.1, 0.8, 0.3, 0.1]

        x_tup = [Float64.(p) for p in SPREAD_2D]
        x_vec = [collect(Float64.(p)) for p in SPREAD_2D]
        for x in (x_tup, x_vec)
            sec = SecondOrderPolynomialSurrogate(x, q.(x), lb, ub)
            @test sec.β ≈ expected
            @test sec((2.0, 3.0)) ≈ q((2.0, 3.0))
        end

        # One-dimensional inputs are plain numbers, not one-element containers.
        g = t -> 1.0 + 2.0t + 3.0t^2
        x1 = [0.0, 1.0, 2.0, 3.0, 4.0]
        sec1 = SecondOrderPolynomialSurrogate(x1, g.(x1), 0.0, 4.0)
        @test sec1.β ≈ [1.0, 2.0, 3.0]
        @test sec1(2.5) ≈ g(2.5)
    end

    @testset "element types" begin
        # Products of these are exact in every type below, so the recovered
        # coefficients can be compared without a tolerance argument.
        target = p -> p[1] * p[2]

        @testset "precision is carried through the fit" begin
            for T in (Float64, Float32, BigFloat)
                x = [T.(p) for p in SPREAD_2D]
                sec = SecondOrderPolynomialSurrogate(
                    x, target.(x), T.((0, 0)), T.((10, 10))
                )
                @test eltype(sec.β) === T
                @test sec(T.((2, 3))) isa T
                @test isapprox(sec(T.((2, 3))), T(6); atol = 100 * eps(T) * 6)
            end
        end

        @testset "integer and rational samples promote like the solve" begin
            # `\` promotes these anyway; what matters is that nothing throws
            # and the fit is still correct.
            for x in (SPREAD_2D, [(p[1] // 1, p[2] // 1) for p in SPREAD_2D])
                sec = SecondOrderPolynomialSurrogate(
                    x, target.(x), (0, 0), (10, 10)
                )
                @test eltype(sec.β) === Float64
                @test sec((2, 3)) ≈ 6.0
            end
        end

        @testset "a degenerate design is caught in every element type" begin
            # Collinear points span only {1, t, t^2} however many there are, so
            # the rank is 3 of the 6 columns a 2D quadratic needs. The rank was
            # previously read from an SVD, which is unavailable outside
            # Float32/Float64 and left these types unchecked.
            for pts in (
                    [(t, 2t) for t in 1.0:20.0],
                    [(Float32(t), 2.0f0 * Float32(t)) for t in 1.0:20.0],
                    [(BigFloat(t), 2 * BigFloat(t)) for t in 1.0:20.0],
                    [(t, 2t) for t in 1:20],
                    [(t // 1, 2t // 1) for t in 1:20],
                    # A constant coordinate is degenerate for the same reason.
                    [(t, 3.0) for t in 1.0:20.0],
                )
                y = [p[1] * p[1] for p in pts]
                err = try
                    SecondOrderPolynomialSurrogate(pts, y, (0, 0), (25, 50))
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("rank 3", err.msg)
            end
        end
    end

    @testset "query forms" begin
        q = p -> p[1] * p[2]
        x = [Float64.(p) for p in SPREAD_2D]
        sec = SecondOrderPolynomialSurrogate(x, q.(x), [0.0, 0.0], [10.0, 10.0])
        expected = q((2.0, 3.0))

        @test sec((2.0, 3.0)) ≈ expected
        @test sec([2.0, 3.0]) ≈ expected
        # Bounds written as row matrices produce 1 x d query points, e.g. from
        # `(lb .+ ub) ./ 2`.
        @test sec([2.0 3.0]) ≈ expected
        # A query need not share the element type of the fit.
        @test sec((2, 3)) ≈ expected

        g = t -> 1.0 + 2.0t + 3.0t^2
        x1 = [0.0, 1.0, 2.0, 3.0, 4.0]
        sec1 = SecondOrderPolynomialSurrogate(x1, g.(x1), 0.0, 4.0)
        # One-dimensional surrogates accept the scalar and its containers.
        @test sec1(2.5) ≈ g(2.5)
        @test sec1((2.5,)) ≈ g(2.5)
        @test sec1([2.5]) ≈ g(2.5)
    end

    @testset "coefficient order is the documented one" begin
        # The docstring promises intercept, coordinates, pairwise products in
        # lexicographic order, then squares -- so a target written as
        # a + b'p + p'Cp has cross coefficient 2 * C[1, 2], not C[1, 2].
        a = 0.3
        b = [0.7, 0.1, -0.2]
        C = [0.3 0.4 0.15; 0.4 0.1 -0.25; 0.15 -0.25 0.5]
        f = p -> a + b' * collect(p) + collect(p)' * C * collect(p)

        lb = fill(-5.0, 3)
        ub = fill(5.0, 3)
        x = sample(60, lb, ub, SobolSample())
        sec = SecondOrderPolynomialSurrogate(x, f.(x), lb, ub)
        @test sec.β ≈ [
            a,
            b[1], b[2], b[3],
            2C[1, 2], 2C[1, 3], 2C[2, 3],
            C[1, 1], C[2, 2], C[3, 3],
        ]
        @test length(sec.β) == 1 + 2 * 3 + 3 * 2 ÷ 2
    end

    @testset "least-squares residual is orthogonal to the design" begin
        # The defining property of the fit, and the thing that distinguishes it
        # from an interpolant: it does not pass through the samples, but the
        # residual has no component along any monomial in the basis.
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        x = sample(80, lb, ub, SobolSample())
        y = (p -> exp(p[1] / 3) * log1p(p[2])).(x)
        sec = SecondOrderPolynomialSurrogate(x, y, lb, ub)
        r = y .- sec.(x)
        @test !isapprox(maximum(abs, r), 0.0; atol = 1.0e-6)   # genuinely a regression
        basis = [
            ones(length(x)), [p[1] for p in x], [p[2] for p in x],
            [p[1] * p[2] for p in x], [p[1]^2 for p in x], [p[2]^2 for p in x],
        ]
        for col in basis
            @test isapprox(dot(r, col), 0.0; atol = 1.0e-8 * maximum(abs, y) * length(x))
        end
    end

    @testset "automatic differentiation" begin
        a = 0.3
        b = [0.7, 0.1]
        C = [0.3 0.4; 0.4 0.1]
        f = p -> a + b' * collect(p) + collect(p)' * C * collect(p)
        lb = [-5.0, -5.0]
        ub = [5.0, 5.0]
        x = sample(30, lb, ub, SobolSample())
        sec = SecondOrderPolynomialSurrogate(x, f.(x), lb, ub)
        p0 = [1.5, -2.0]

        @testset "gradient, ND" begin
            # Exact for a quadratic: grad = b + 2Cp, independent of the fit.
            g = ForwardDiff.gradient(sec, p0)
            @test isapprox(g, b + 2C * p0; atol = 1.0e-10)
            @test Zygote.gradient(sec, p0)[1] ≈ g
            # A tuple query differentiates to a tuple.
            zt = Zygote.gradient(sec, (1.5, -2.0))[1]
            @test zt isa Tuple
            @test all(isapprox.(zt, Tuple(g); atol = 1.0e-10))
        end

        @testset "Hessian is exactly 2C" begin
            H = ForwardDiff.hessian(sec, p0)
            @test isapprox(H, 2C; atol = 1.0e-10)
            @test H ≈ H'
            @test Zygote.hessian(sec, p0) ≈ H
        end

        @testset "third derivative of a quadratic vanishes" begin
            d3 = ForwardDiff.derivative(
                t -> ForwardDiff.derivative(
                    u -> ForwardDiff.derivative(v -> sec([v, 1.0]), u), t
                ), 1.0
            )
            @test isapprox(d3, 0.0; atol = 1.0e-8)
        end

        @testset "1D" begin
            g = t -> 1.0 + 2.0t + 3.0t^2
            x1 = sample(30, 0.0, 5.0, SobolSample())
            sec1 = SecondOrderPolynomialSurrogate(x1, g.(x1), 0.0, 5.0)
            @test isapprox(ForwardDiff.derivative(sec1, 2.0), 14.0; atol = 1.0e-10)
            @test Zygote.gradient(sec1, 2.0)[1] ≈ ForwardDiff.derivative(sec1, 2.0)
            second = ForwardDiff.derivative(t -> ForwardDiff.derivative(sec1, t), 2.0)
            @test isapprox(second, 6.0; atol = 1.0e-8)
        end

        @testset "multi-output Jacobian" begin
            fm = p -> [p[1], p[2]^2]
            secm = SecondOrderPolynomialSurrogate(x, fm.(x), lb, ub)
            J = ForwardDiff.jacobian(secm, p0)
            @test size(J) == (2, 2)
            @test isapprox(J, [1.0 0.0; 0.0 2p0[2]]; atol = 1.0e-10)
            @test Zygote.jacobian(secm, p0)[1] ≈ J
            @test Zygote.gradient(p -> sum(secm(p)), p0)[1] ≈ vec(sum(J; dims = 1))
        end

        @testset "the fit itself differentiates in the training data" begin
            # Sensitivity of a fitted value to the samples it was built from.
            # ForwardDiff traces the solve; Zygote cannot, because the design
            # matrix is built by mutation.
            pts = [Float64.(p) for p in SPREAD_2D]
            yv = [p[1] * p[2] for p in pts]
            q = [2.0, 3.0]
            h = 1.0e-6

            fy = yy -> SecondOrderPolynomialSurrogate(pts, yy, lb, ub)(q)
            gy = ForwardDiff.gradient(fy, yv)
            fdy = [
                (fy(yv + h * I[1:length(yv), k]) - fy(yv - h * I[1:length(yv), k])) / 2h
                    for k in eachindex(yv)
            ]
            @test isapprox(gy, fdy; atol = 1.0e-6)

            flat = collect(Iterators.flatten(pts))
            fx = v -> SecondOrderPolynomialSurrogate(
                [(v[2i - 1], v[2i]) for i in 1:length(pts)], yv, lb, ub
            )(q)
            gx = ForwardDiff.gradient(fx, flat)
            fdx = [
                (fx(flat + h * I[1:length(flat), k]) - fx(flat - h * I[1:length(flat), k])) / 2h
                    for k in eachindex(flat)
            ]
            @test isapprox(gx, fdx; atol = 1.0e-5)
        end
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
