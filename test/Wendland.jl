using Surrogates
using Test
using LinearAlgebra

@testset "Wendland" begin
    @testset "kernel" begin
        # One sample point makes the coefficient equal the response, so the
        # surrogate is exactly y * psi(eps * distance).
        # In 1D, ell = 1 ÷ 2 + 2 = 2, so psi(r) = (1 - r)^3 * (3r + 1).
        w = Wendland([0.0], [3.0], -5.0, 5.0, eps = 0.5)
        @test w(0.0) ≈ 3.0
        @test w(1.0) ≈ 3 * (1 - 0.5)^3 * (3 * 0.5 + 1)
        @test w(1.0) ≈ 0.9375
        @test w(-1.0) ≈ w(1.0)

        # Compact support: identically zero at and beyond 1 / eps.
        @test w(2.0) == 0.0
        @test w(2.5) == 0.0
        @test w(5.0) == 0.0
        # and continuous into it
        @test isapprox(w(2.0 - 1.0e-8), 0.0, atol = 1.0e-14)
    end

    @testset "eps is the reciprocal support radius" begin
        for eps in (0.25, 0.5, 2.0)
            w = Wendland([0.0], [1.0], -10.0, 10.0, eps = eps)
            radius = 1 / eps
            @test w(0.99 * radius) > 0
            @test w(radius) == 0.0
            @test w(1.01 * radius) == 0.0
        end
    end

    @testset "kernel exponent follows the input dimension" begin
        # ell = d ÷ 2 + 2, so the exponent is d ÷ 2 + 3.
        r = 0.5
        w1 = Wendland([0.0], [1.0], -5.0, 5.0, eps = 0.5)
        w2 = Wendland([(0.0, 0.0)], [1.0], [-5.0, -5.0], [5.0, 5.0], eps = 0.5)
        @test w1(1.0) ≈ (1 - r)^3 * (3r + 1)
        @test w2((1.0, 0.0)) ≈ (1 - r)^4 * (4r + 1)
    end

    @testset "1D input" begin
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        y = sin.(x)
        # Support radius 2 makes neighbouring kernels overlap, so the
        # interpolation system is not trivially diagonal.
        w = Wendland(x, y, 0.0, 2.0, eps = 0.5)

        @testset "interpolation" begin
            @test all(isapprox(w(x[i]), y[i], atol = 1.0e-4) for i in eachindex(x))
        end

        @testset "coefficients solve the interpolation system" begin
            n = length(x)
            W = [Surrogates._wendland(x[i] .- x[j], w.eps) for i in 1:n, j in 1:n]
            @test W ≈ permutedims(W)
            @test isapprox(W * w.coeff, y, atol = 1.0e-6)
        end

        @testset "call forms" begin
            @test w(0.75) isa Number
            @test w([0.75]) ≈ w(0.75)
            @test w((0.75,)) ≈ w(0.75)
            @test_throws ArgumentError w(Float64[])
            @test_throws ArgumentError w((2.0, 3.0, 4.0))
        end

        @testset "update!" begin
            mk() = Wendland(copy(x), copy(y), 0.0, 3.0, eps = 0.5)
            # A single sample may be bare or wrapped in a one-element vector;
            # both append one point rather than a nested container.
            w1 = mk()
            update!(w1, 2.5, sin(2.5))
            @test length(w1.x) == 6
            @test w1.x[end] == 2.5
            @test isapprox(w1(2.5), sin(2.5), atol = 1.0e-4)

            w2 = mk()
            update!(w2, [2.5], [sin(2.5)])
            @test length(w2.x) == 6
            @test w2.x[end] == 2.5
            @test isapprox(w2(2.5), sin(2.5), atol = 1.0e-4)

            w3 = mk()
            update!(w3, [2.5, 3.0], sin.([2.5, 3.0]))
            @test length(w3.x) == 7
            @test isapprox(w3(3.0), sin(3.0), atol = 1.0e-4)
        end

        @testset "update! does not mutate the caller's arrays" begin
            xs = copy(x)
            ys = copy(y)
            wu = Wendland(xs, ys, 0.0, 3.0, eps = 0.5)
            update!(wu, 2.5, sin(2.5))
            @test length(xs) == 5
            @test length(ys) == 5
            @test length(wu.x) == 6
        end

        @testset "element types are preserved" begin
            w32 = Wendland(Float32.(x), Float32.(y), 0.0f0, 2.0f0, eps = 0.5f0)
            @test w32(0.75f0) isa Float32
            xb, yb = BigFloat.(x), BigFloat.(y)
            wb = Wendland(xb, yb, BigFloat(0), BigFloat(2), eps = BigFloat(1) / 2)
            @test wb(BigFloat(3) / 4) isa BigFloat
            @test isapprox(wb(xb[3]), yb[3], atol = 1.0e-4)
        end
    end

    @testset "ND input" begin
        lb = [0.0, 0.0]
        ub = [4.0, 4.0]
        x = sample(12, lb, ub, SobolSample())
        f = p -> p[1] + p[2]
        y = f.(x)
        w = Wendland(x, y, lb, ub, eps = 0.3)

        @testset "interpolation" begin
            @test all(isapprox(w(x[i]), y[i], atol = 1.0e-4) for i in eachindex(x))
        end

        @testset "call forms agree" begin
            val = (1.0, 2.0)
            @test w(val) ≈ w([1.0, 2.0])
            # A query written as a row matrix must not broadcast against a
            # sample point into a d x d outer difference.
            @test w(val) ≈ w([1.0 2.0])
            @test_throws ArgumentError w(Float64[])
            @test_throws ArgumentError w(2.0)
            @test_throws ArgumentError w((2.0, 3.0, 4.0))
        end

        @testset "a node reached through a different container" begin
            @test w(x[3]) ≈ w(collect(x[3]))
            @test w(x[3]) ≈ w(permutedims(collect(x[3])))
        end

        @testset "update!" begin
            mk() = Wendland(copy(x), copy(y), lb, ub, eps = 0.3)
            w1 = mk()
            update!(w1, (3.0, 3.5), f((3.0, 3.5)))
            @test length(w1.x) == 13
            @test isapprox(w1((3.0, 3.5)), f((3.0, 3.5)), atol = 1.0e-4)

            w2 = mk()
            update!(w2, [(3.0, 3.5)], [f((3.0, 3.5))])
            @test length(w2.x) == 13
            @test isapprox(w2((3.0, 3.5)), f((3.0, 3.5)), atol = 1.0e-4)

            w3 = mk()
            update!(w3, [(0.5, 1.0), (1.5, 2.5)], [f((0.5, 1.0)), f((1.5, 2.5))])
            @test length(w3.x) == 14
            @test isapprox(w3((1.5, 2.5)), f((1.5, 2.5)), atol = 1.0e-4)
        end

        @testset "the kernel sees only distances" begin
            # Rotating the samples and the query leaves the prediction alone.
            θ = 0.7
            R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
            rot(p) = Tuple(R * collect(p))
            w_rot = Wendland(rot.(x), y, lb, ub, eps = 0.3)
            @test w_rot(rot((1.0, 2.0))) ≈ w((1.0, 2.0))

            shift(p) = (p[1] + 10.0, p[2] - 3.0)
            w_sh = Wendland(shift.(x), y, lb, ub, eps = 0.3)
            @test w_sh(shift((1.0, 2.0))) ≈ w((1.0, 2.0))
        end
    end

    @testset "input types" begin
        xt = [(1.0, 2.0), (3.0, 4.0), (5.0, 1.0)]
        xv = [collect(p) for p in xt]
        yn = [1.0, 2.0, 3.0]
        lb = [0.0, 0.0]
        ub = [6.0, 6.0]
        st = Wendland(xt, yn, lb, ub, eps = 0.2)

        @testset "integer samples" begin
            # The kernel values are not integers, so the interpolation matrix
            # has to be built at a floating-point element type.
            s = Wendland([1, 2, 3], [1, 4, 9], 0, 4, eps = 0.5)
            @test s(2) ≈ 4.0
            @test s(1.5) isa AbstractFloat

            # Integer coordinates whose supports actually overlap, so the
            # off-diagonal entries are nonzero.
            si = Wendland([(1, 2), (2, 2), (3, 2)], yn, [0, 0], [6, 6], eps = 0.3)
            @test isapprox(si((2.0, 2.0)), 2.0, atol = 1.0e-6)
        end

        @testset "rational samples" begin
            s = Wendland(
                Rational{Int}[1, 2, 3], Rational{Int}[1, 4, 9],
                0 // 1, 4 // 1, eps = 1 // 2
            )
            @test s(3 // 2) isa AbstractFloat
            @test isapprox(s(2 // 1), 4.0, atol = 1.0e-6)
        end

        @testset "ND samples as vectors rather than tuples" begin
            sv = Wendland(xv, yn, lb, ub, eps = 0.2)
            @test sv([2.0, 3.0]) ≈ st((2.0, 3.0))
            @test isapprox(sv(xv[2]), yn[2], atol = 1.0e-6)
        end

        @testset "bounds as tuples" begin
            s = Wendland(xt, yn, (0.0, 0.0), (6.0, 6.0), eps = 0.2)
            @test s((2.0, 3.0)) ≈ st((2.0, 3.0))
        end

        @testset "query as a column matrix" begin
            @test st(reshape([2.0, 3.0], 2, 1)) ≈ st((2.0, 3.0))
        end

        @testset "three-dimensional input" begin
            x3 = [(1.0, 2.0, 3.0), (2.0, 1.0, 0.5), (0.0, 0.0, 1.0)]
            s = Wendland(x3, yn, [0.0, 0, 0], [6.0, 6, 6], eps = 0.2)
            @test all(isapprox(s(x3[i]), yn[i], atol = 1.0e-6) for i in eachindex(x3))
            # ell = 3 ÷ 2 + 2 = 3, matching the two-dimensional exponent of 4
            r = 0.5
            w3 = Wendland([(0.0, 0.0, 0.0)], [1.0], [-5.0, -5, -5], [5.0, 5, 5], eps = 0.5)
            @test w3((1.0, 0.0, 0.0)) ≈ (1 - r)^4 * (4r + 1)
        end
    end

    @testset "duplicate sample points" begin
        # `surrogate_optimize!` re-proposes points it already has, which makes
        # the interpolation matrix singular. The system stays consistent, so
        # the solve still reproduces the responses.
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        y = sin.(x)
        w = Wendland(x, y, 0.0, 3.0, eps = 0.5)
        update!(w, 1.0, sin(1.0))
        @test length(w.x) == 6
        @test all(isapprox(w(w.x[i]), w.y[i], atol = 1.0e-6) for i in eachindex(w.x))
        @test isapprox(w(1.0), sin(1.0), atol = 1.0e-6)
    end

    @testset "non-converged solve warns" begin
        x = sample(40, 0.0, 10.0, SobolSample())
        y = sin.(x)
        @test_logs (:warn, r"did not converge") match_mode = :any Wendland(
            x, y, 0.0, 10.0, eps = 0.01, maxiters = 1
        )
        # A well-conditioned solve stays quiet.
        @test_logs Wendland([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], 0.0, 2.0, eps = 0.5)
    end

    @testset "vector responses are not supported" begin
        x = [0.0, 1.0, 2.0]
        @test_throws MethodError Wendland(x, [[t, t^2] for t in x], 0.0, 2.0, eps = 0.5)
    end
end
