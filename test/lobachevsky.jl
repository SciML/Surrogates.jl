using Surrogates
using Test
using LinearAlgebra
using QuadGK
using Cubature

@testset "LobachevskySurrogate" begin
    @testset "hyperparameter validation" begin
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        y = sin.(x)
        xn = [(1.0, 2.0), (3.0, 1.0), (2.0, 3.0)]
        yn = [1.0, 2.0, 3.0]
        lbn, ubn = [0.0, 0.0], [4.0, 4.0]

        # alpha = 0 makes every kernel value identical, leaving a rank-one
        # interpolation system rather than an error.
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 2.0, alpha = 0.0)
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 2.0, alpha = -1.0)
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 2.0, alpha = 5.0)
        # The same bounds apply per dimension
        @test_throws ArgumentError LobachevskySurrogate(xn, yn, lbn, ubn, alpha = [1.0, 0.0])
        @test_throws ArgumentError LobachevskySurrogate(xn, yn, lbn, ubn, alpha = [5.0, 1.0])

        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 2.0, n = 3)
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 2.0, n = 0)
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 2.0, n = -2)
        # factorial(n) overflows Int64 past 20
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 2.0, n = 22)
        @test LobachevskySurrogate(x, y, 0.0, 2.0, n = 20) isa LobachevskySurrogate
        @test LobachevskySurrogate(x, y, 0.0, 2.0, alpha = 4.0) isa LobachevskySurrogate

        # One scale per input dimension; a scalar would fail later with a
        # BoundsError from indexing it per dimension.
        @test_throws ArgumentError LobachevskySurrogate(xn, yn, lbn, ubn, alpha = 2.0)
        @test_throws ArgumentError LobachevskySurrogate(
            xn, yn, lbn, ubn, alpha = [1.0, 1.0, 1.0])
        @test LobachevskySurrogate(xn, yn, lbn, ubn, alpha = (1.0, 1.0)) isa
              LobachevskySurrogate
        # The default supplies one scale per dimension
        @test length(LobachevskySurrogate(xn, yn, lbn, ubn).alpha) == 2

        for order in (2, 4, 8, 20)
            @test LobachevskySurrogate(x, y, 0.0, 2.0, n = order) isa LobachevskySurrogate
        end
    end

    @testset "kernel" begin
        # The kernel depends on the two points only through their difference,
        # and is even in it, so the interpolation matrix is symmetric.
        for (p, q) in ((0.3, 1.7), (1.0, 2.5), (-1.0, 0.4))
            @test Surrogates.phi_nj1D(p, q, 1.0, 4) ≈ Surrogates.phi_nj1D(q, p, 1.0, 4)
        end
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        D = [Surrogates.phi_nj1D(x[i], x[j], 1.0, 4) for i in eachindex(x), j in eachindex(x)]
        # Symmetric to rounding only, which is why the solve wraps it in
        # `Symmetric` before factorizing.
        @test D ≈ permutedims(D)
        @test maximum(abs.(D .- permutedims(D))) < 1.0e-14
        @test isposdef(Symmetric(D))
        # Compact support: the kernel vanishes past the order-dependent radius.
        @test Surrogates.phi_nj1D(0.0, 100.0, 1.0, 4) == 0.0
    end

    @testset "1D input" begin
        obj = t -> 3t + log(t)
        a, b = 1.0, 4.0
        x = sample(60, a, b, SobolSample())
        y = obj.(x)
        loba = LobachevskySurrogate(x, y, a, b, alpha = 2.0, n = 6)

        @testset "interpolation" begin
            @test all(isapprox(loba(x[i]), y[i], atol = 1.0e-6) for i in eachindex(x))
        end

        @testset "approximates the objective" begin
            @test all(isapprox(loba(t), obj(t), atol = 1.0e-3)
            for t in range(1.2, 3.8, length = 25))
        end

        @testset "call forms" begin
            @test loba(2.5) isa Number
            @test loba([2.5]) ≈ loba(2.5)
            @test loba((2.5,)) ≈ loba(2.5)
            @test_throws ArgumentError loba(Float64[])
            @test_throws ArgumentError loba((2.0, 3.0, 4.0))
        end

        @testset "update!" begin
            mk() = LobachevskySurrogate(copy(x), copy(y), a, 6.0, alpha = 2.0, n = 6)
            # A single sample may be bare or wrapped in a one-element vector.
            l1 = mk()
            update!(l1, 4.5, obj(4.5))
            @test length(l1.x) == length(x) + 1
            @test isapprox(l1(4.5), obj(4.5), atol = 1.0e-6)

            l2 = mk()
            update!(l2, [4.5], [obj(4.5)])
            @test length(l2.x) == length(x) + 1
            @test isapprox(l2(4.5), obj(4.5), atol = 1.0e-6)

            l3 = mk()
            update!(l3, [4.5, 5.0], obj.([4.5, 5.0]))
            @test length(l3.x) == length(x) + 2
            @test isapprox(l3(5.0), obj(5.0), atol = 1.0e-6)
        end

        @testset "update! does not mutate the caller's arrays" begin
            xs, ys = copy(x), copy(y)
            l = LobachevskySurrogate(xs, ys, a, 6.0, alpha = 2.0, n = 6)
            update!(l, 4.5, obj(4.5))
            @test length(xs) == length(x)
            @test length(ys) == length(y)
            @test length(l.x) == length(x) + 1
        end
    end

    @testset "ND input" begin
        obj = p -> p[1] + log(p[2] + 1)
        lb, ub = [0.0, 0.0], [4.0, 4.0]
        x = sample(60, lb, ub, SobolSample())
        y = obj.(x)
        loba = LobachevskySurrogate(x, y, lb, ub, alpha = [2.0, 2.0], n = 6)

        @testset "interpolation" begin
            @test all(isapprox(loba(x[i]), y[i], atol = 1.0e-5) for i in eachindex(x))
        end

        @testset "call forms agree" begin
            val = (1.0, 2.0)
            @test loba(val) ≈ loba([1.0, 2.0])
            @test loba(val) ≈ loba([1.0 2.0])
            @test loba(val) ≈ loba(reshape([1.0, 2.0], 2, 1))
            @test_throws ArgumentError loba(Float64[])
            @test_throws ArgumentError loba(1.0)
            @test_throws ArgumentError loba((2.0, 3.0, 4.0))
        end

        @testset "update!" begin
            mk() = LobachevskySurrogate(copy(x), copy(y), lb, ub, alpha = [2.0, 2.0], n = 6)
            l1 = mk()
            update!(l1, (1.5, 2.5), obj((1.5, 2.5)))
            @test length(l1.x) == length(x) + 1
            @test isapprox(l1((1.5, 2.5)), obj((1.5, 2.5)), atol = 1.0e-5)

            l2 = mk()
            update!(l2, [(1.5, 2.5)], [obj((1.5, 2.5))])
            @test length(l2.x) == length(x) + 1

            l3 = mk()
            update!(l3, [(1.5, 2.5), (3.5, 0.5)], [obj((1.5, 2.5)), obj((3.5, 0.5))])
            @test length(l3.x) == length(x) + 2
        end

        @testset "three-dimensional input" begin
            obj3 = p -> p[1] + p[2] + p[3]^2
            lb3, ub3 = [0.0, 0.0, 0.0], [4.0, 4.0, 4.0]
            x3 = sample(50, lb3, ub3, SobolSample())
            y3 = obj3.(x3)
            l3 = LobachevskySurrogate(x3, y3, lb3, ub3, alpha = [2.0, 2.0, 2.0], n = 6)
            @test all(isapprox(l3(x3[i]), y3[i], atol = 1.0e-4) for i in eachindex(x3))
        end
    end

    @testset "closed-form integral" begin
        @testset "1D" begin
            obj = t -> 3t + log(t)
            a, b = 1.0, 4.0
            x = sample(60, a, b, SobolSample())
            loba = LobachevskySurrogate(x, obj.(x), a, b, alpha = 2.0, n = 6)
            # The closed form must reproduce numerical integration of the very
            # same surrogate; that isolates the formula from the fit quality.
            @test isapprox(lobachevsky_integral(loba, a, b),
                quadgk(loba, a, b)[1], rtol = 1.0e-8)
            # and the surrogate's integral must approximate the objective's
            @test isapprox(lobachevsky_integral(loba, a, b),
                quadgk(obj, a, b)[1], atol = 1.0e-3)
            # A sub-interval integrates too, and splitting is additive
            @test isapprox(lobachevsky_integral(loba, a, 2.5) +
                           lobachevsky_integral(loba, 2.5, b),
                lobachevsky_integral(loba, a, b), rtol = 1.0e-10)
            # Integer bounds
            @test lobachevsky_integral(loba, 1, 4) ≈ lobachevsky_integral(loba, a, b)
        end

        @testset "ND" begin
            obj = p -> p[1] + log(p[2] + 1)
            lb, ub = [0.0, 0.0], [4.0, 4.0]
            x = sample(60, lb, ub, SobolSample())
            loba = LobachevskySurrogate(x, obj.(x), lb, ub, alpha = [2.0, 2.0], n = 6)
            @test isapprox(lobachevsky_integral(loba, lb, ub),
                hcubature(loba, lb, ub, abstol = 1.0e-8)[1], rtol = 1.0e-5)
            @test isapprox(lobachevsky_integral(loba, lb, ub),
                hcubature(obj, lb, ub, abstol = 1.0e-8)[1], rtol = 1.0e-2)
            # Bounds may be tuples
            @test lobachevsky_integral(loba, (lb[1], lb[2]), (ub[1], ub[2])) ≈
                  lobachevsky_integral(loba, lb, ub)
        end
    end

    @testset "integrate_dimension" begin
        obj = p -> p[1] + log(p[2] + 1)
        lb, ub = [0.0, 0.0], [4.0, 4.0]
        x = sample(50, lb, ub, SobolSample())
        loba = LobachevskySurrogate(x, obj.(x), lb, ub, alpha = [2.0, 2.0], n = 6)

        alpha_before = copy(loba.alpha)
        lb_arg, ub_arg = copy(lb), copy(ub)
        reduced = lobachevsky_integrate_dimension(loba, lb_arg, ub_arg, 2)

        @testset "nothing is mutated" begin
            @test loba.alpha == alpha_before
            @test lb_arg == lb
            @test ub_arg == ub
            @test length(loba.x[1]) == 2
        end

        @testset "the reduced surrogate is usable and one-dimensional" begin
            @test reduced.alpha == alpha_before[1]
            @test reduced.lb == lb[1]
            @test reduced.ub == ub[1]
            @test length(reduced.x) == length(loba.x)
            @test reduced(1.0) isa Number
        end

        @testset "it equals the marginal" begin
            # Integrating out dimension 2 must reproduce a direct quadrature of
            # the full surrogate over that coordinate.
            for x1 in (0.5, 1.0, 2.0, 3.0)
                @test isapprox(reduced(x1),
                    quadgk(t -> loba((x1, t)), lb[2], ub[2])[1], rtol = 1.0e-6)
            end
        end

        @testset "integrating the marginal gives the full integral" begin
            @test isapprox(lobachevsky_integral(reduced, reduced.lb, reduced.ub),
                lobachevsky_integral(loba, lb, ub), rtol = 1.0e-8)
        end

        @testset "the reduced surrogate is self-consistent" begin
            # Its stored responses are its own values at the reduced nodes, so
            # refitting reproduces its coefficients.
            refit = LobachevskySurrogate(reduced.x, reduced.y, reduced.lb, reduced.ub,
                alpha = reduced.alpha, n = reduced.n)
            @test isapprox(refit.coeff, reduced.coeff, rtol = 1.0e-5)
        end

        @testset "either dimension can be integrated out" begin
            r1 = lobachevsky_integrate_dimension(loba, lb, ub, 1)
            @test isapprox(r1(2.0),
                quadgk(t -> loba((t, 2.0)), lb[1], ub[1])[1], rtol = 1.0e-6)
            # Bounds may be tuples
            rt = lobachevsky_integrate_dimension(loba, (lb[1], lb[2]), (ub[1], ub[2]), 2)
            @test rt(1.0) ≈ reduced(1.0)
        end

        @testset "integer responses" begin
            # The marginal's values are not integers, so the reduced surrogate
            # must be built with them rather than assigned into an integer `y`.
            li = LobachevskySurrogate([(1.0, 2.0), (3.0, 1.0), (2.0, 3.0), (0.5, 0.5)],
                [1, 2, 3, 4], lb, ub, alpha = [1.0, 1.0])
            ri = lobachevsky_integrate_dimension(li, lb, ub, 2)
            @test eltype(ri.y) <: AbstractFloat
            @test isapprox(ri(2.0),
                quadgk(t -> li((2.0, t)), lb[2], ub[2])[1], rtol = 1.0e-6)
        end

        @testset "the dimension is checked" begin
            @test_throws ArgumentError lobachevsky_integrate_dimension(loba, lb, ub, 0)
            @test_throws ArgumentError lobachevsky_integrate_dimension(loba, lb, ub, 3)
        end

        @testset "3D reduces to 2D" begin
            obj3 = p -> p[1] + p[2] + p[3]^2
            lb3, ub3 = [0.0, 0.0, 0.0], [4.0, 4.0, 4.0]
            x3 = sample(40, lb3, ub3, SobolSample())
            l3 = LobachevskySurrogate(x3, obj3.(x3), lb3, ub3, alpha = [2.0, 2.0, 2.0], n = 6)
            r3 = lobachevsky_integrate_dimension(l3, lb3, ub3, 3)
            @test length(r3.alpha) == 2
            @test length(r3.lb) == 2
            @test length(r3.x[1]) == 2
            @test isapprox(r3((1.0, 2.0)),
                quadgk(t -> l3((1.0, 2.0, t)), lb3[3], ub3[3])[1], rtol = 1.0e-6)
        end
    end

    @testset "sparse matches dense" begin
        obj = t -> 3t + log(t)
        x = sample(60, 1.0, 4.0, SobolSample())
        y = obj.(x)
        dense = LobachevskySurrogate(x, y, 1.0, 4.0, alpha = 2.0, n = 6)
        sp = LobachevskySurrogate(x, y, 1.0, 4.0, alpha = 2.0, n = 6, sparse = true)
        @test isapprox(sp(2.5), dense(2.5), rtol = 1.0e-6)

        objn = p -> p[1] + log(p[2] + 1)
        lb, ub = [0.0, 0.0], [4.0, 4.0]
        xn = sample(50, lb, ub, SobolSample())
        yn = objn.(xn)
        dn = LobachevskySurrogate(xn, yn, lb, ub, alpha = [2.0, 2.0], n = 6)
        sn = LobachevskySurrogate(xn, yn, lb, ub, alpha = [2.0, 2.0], n = 6, sparse = true)
        @test isapprox(sn((1.0, 2.0)), dn((1.0, 2.0)), rtol = 1.0e-6)
    end

    @testset "input types" begin
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        y = sin.(x)

        @testset "integer samples" begin
            # The kernel values are not integers, so the interpolation matrix
            # has to be built at a floating-point element type.
            s = LobachevskySurrogate([0, 1, 2, 3], [0, 1, 4, 9], 0, 4, alpha = 1.0)
            @test s(1.5) isa AbstractFloat
            @test isapprox(s(2), 4.0, atol = 1.0e-6)
        end

        @testset "element types are preserved" begin
            s32 = LobachevskySurrogate(Float32.(x), Float32.(y), 0.0f0, 2.0f0, alpha = 1.0f0)
            @test s32(0.75f0) isa Float32
            @test lobachevsky_integral(s32, 0.0f0, 2.0f0) isa Float32
            # The multivariate integral accumulates into a running product,
            # which must start at the element type rather than at Float64.
            xt32 = [(1.0f0, 2.0f0), (3.0f0, 1.0f0), (2.0f0, 3.0f0)]
            n32 = LobachevskySurrogate(xt32, Float32[1, 2, 3], Float32[0, 0],
                Float32[4, 4], alpha = Float32[1, 1])
            @test n32((2.0f0, 2.0f0)) isa Float32
            @test lobachevsky_integral(n32, Float32[0, 0], Float32[4, 4]) isa Float32

            xb, yb = BigFloat.(x), BigFloat.(y)
            sb = LobachevskySurrogate(xb, yb, BigFloat(0), BigFloat(2), alpha = BigFloat(1))
            @test sb(BigFloat(3) / 4) isa BigFloat
            @test isapprox(sb(xb[3]), yb[3], atol = 1.0e-8)
        end

        @testset "ND samples as vectors rather than tuples" begin
            xt = [(1.0, 2.0), (3.0, 1.0), (2.0, 3.0)]
            xv = [collect(p) for p in xt]
            yn = [1.0, 2.0, 3.0]
            lb, ub = [0.0, 0.0], [4.0, 4.0]
            sv = LobachevskySurrogate(xv, yn, lb, ub, alpha = [1.0, 1.0])
            st = LobachevskySurrogate(xt, yn, lb, ub, alpha = [1.0, 1.0])
            @test sv([2.0, 2.0]) ≈ st((2.0, 2.0))
        end

        @testset "bounds as tuples" begin
            xt = [(1.0, 2.0), (3.0, 1.0), (2.0, 3.0)]
            yn = [1.0, 2.0, 3.0]
            s = LobachevskySurrogate(xt, yn, (0.0, 0.0), (4.0, 4.0), alpha = [1.0, 1.0])
            @test s((2.0, 2.0)) isa Number
        end

        @testset "rational samples" begin
            s = LobachevskySurrogate(Rational{Int}[0, 1, 2], Rational{Int}[0, 1, 4],
                0 // 1, 3 // 1, alpha = 1 // 1)
            @test s(3 // 2) isa AbstractFloat
        end

        @testset "integer ND samples and integer responses" begin
            xt = [(1, 2), (3, 1), (2, 3), (0, 0)]
            s = LobachevskySurrogate(xt, [1, 2, 3, 4], [0, 0], [4, 4], alpha = [1.0, 1.0])
            @test s((2.0, 2.0)) isa AbstractFloat
            s1 = LobachevskySurrogate(x, [0, 1, 2, 3, 4], 0.0, 2.0, alpha = 1.0)
            @test s1(0.75) isa AbstractFloat
        end

        @testset "samples as a range" begin
            s = LobachevskySurrogate(range(0.0, 2.0, length = 5), y, 0.0, 2.0, alpha = 1.0)
            @test s(0.75) ≈ LobachevskySurrogate(x, y, 0.0, 2.0, alpha = 1.0)(0.75)
        end

        @testset "queries of other numeric types" begin
            s = LobachevskySurrogate(x, y, 0.0, 2.0, alpha = 1.0)
            @test s(1) ≈ s(1.0)
            @test s(0.75f0) ≈ s(0.75)
            xt = [(1.0, 2.0), (3.0, 1.0), (2.0, 3.0)]
            sn = LobachevskySurrogate(xt, [1.0, 2.0, 3.0], [0.0, 0.0], [4.0, 4.0],
                alpha = [1.0, 1.0])
            @test sn((2, 2.0)) ≈ sn((2.0, 2.0))
        end

        @testset "1D bounds as integers" begin
            s = LobachevskySurrogate(x, y, 0, 2, alpha = 1.0)
            @test s(0.75) ≈ LobachevskySurrogate(x, y, 0.0, 2.0, alpha = 1.0)(0.75)
        end

        @testset "single and duplicate samples" begin
            @test LobachevskySurrogate([1.0], [3.0], 0.0, 2.0, alpha = 1.0)(1.5) isa Number
            # A repeated node makes the system singular but consistent.
            d = LobachevskySurrogate([0.0, 1.0, 1.0], [0.0, 1.0, 1.0], 0.0, 2.0,
                alpha = 1.0)
            @test all(isfinite, d.coeff)
            @test isapprox(d(1.0), 1.0, atol = 1.0e-6)
        end

        @testset "ND update! with a vector sample" begin
            xt = [(1.0, 2.0), (3.0, 1.0), (2.0, 3.0)]
            xv = [collect(p) for p in xt]
            l = LobachevskySurrogate(xv, [1.0, 2.0, 3.0], [0.0, 0.0], [4.0, 4.0],
                alpha = [1.0, 1.0])
            update!(l, [2.0, 2.0], 5.0)
            @test length(l.x) == 4
            @test isapprox(l([2.0, 2.0]), 5.0, atol = 1.0e-6)
        end

        @testset "vector responses" begin
            @test_throws DimensionMismatch LobachevskySurrogate(
                x, [[t, t^2] for t in x[1:(end - 1)]], 0.0, 2.0, alpha = 1.0)
        end
    end

    # The interpolant is linear in the responses and the kernel matrix does not
    # involve them, so every output must agree with the surrogate fitted to
    # that output alone.
    @testset "vector-valued responses" begin
        f = t -> [sin(t), cos(t), 2t]
        x = sort(sample(20, 0.0, 4.0, SobolSample()))
        y = f.(x)
        lb, ub = 0.0, 4.0

        @testset "1D" begin
            s = LobachevskySurrogate(x, y, lb, ub, alpha = 2.0, n = 6)
            @test size(s.coeff) == (length(x), 3)
            @test s(1.3) isa AbstractVector
            @test length(s(1.3)) == 3
            # Interpolation still holds at every node, output by output.
            @test all(isapprox(s(x[i]), y[i], atol = 1.0e-8) for i in eachindex(x))
            for k in 1:3
                sk = LobachevskySurrogate(
                    x, [yi[k] for yi in y], lb, ub, alpha = 2.0, n = 6)
                @test isapprox(s(1.3)[k], sk(1.3), atol = 1.0e-10)
                @test isapprox(
                    lobachevsky_integral(s, lb, ub)[k],
                    lobachevsky_integral(sk, lb, ub), atol = 1.0e-10
                )
            end
        end

        @testset "1D update!" begin
            s = LobachevskySurrogate(x, y, lb, ub, alpha = 2.0, n = 6)
            update!(s, 4.5, f(4.5))
            @test size(s.coeff) == (length(x) + 1, 3)
            @test isapprox(s(4.5), f(4.5), atol = 1.0e-8)
            update!(s, [5.0, 5.5], f.([5.0, 5.5]))
            @test length(s.y) == length(x) + 3
            @test isapprox(s(5.5), f(5.5), atol = 1.0e-8)
        end

        @testset "ND" begin
            g = p -> [p[1] * p[2], sin(p[1]) + p[2]^2]
            lbn, ubn = [0.0, 0.0], [2.0, 2.0]
            xn = sample(40, lbn, ubn, SobolSample())
            yn = g.(xn)
            s = LobachevskySurrogate(xn, yn, lbn, ubn, alpha = [2.0, 2.0], n = 6)
            @test size(s.coeff) == (length(xn), 2)
            @test all(isapprox(s(xn[i]), yn[i], atol = 1.0e-8) for i in eachindex(xn))
            for k in 1:2
                sk = LobachevskySurrogate(
                    xn, [yi[k] for yi in yn], lbn, ubn, alpha = [2.0, 2.0], n = 6)
                @test isapprox(s((1.0, 1.5))[k], sk((1.0, 1.5)), atol = 1.0e-10)
                @test isapprox(
                    lobachevsky_integral(s, lbn, ubn)[k],
                    lobachevsky_integral(sk, lbn, ubn), atol = 1.0e-10
                )
            end

            marginal = lobachevsky_integrate_dimension(s, lbn, ubn, 2)
            @test size(marginal.coeff) == (length(xn), 2)
            @test marginal.y[1] isa AbstractVector
            for k in 1:2
                quad = quadgk(t -> s((1.0, t))[k], lbn[2], ubn[2])[1]
                @test isapprox(marginal(1.0)[k], quad, atol = 1.0e-6)
            end
        end
    end
end
