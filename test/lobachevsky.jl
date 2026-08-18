using Surrogates
using LinearAlgebra
using Test
using QuadGK
using Cubature

@testset "LobachevskySurrogate" begin
    @testset "hyperparameter validation" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 5.0, alpha = 5.0)
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 5.0, alpha = -1.0)
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 5.0, n = 3)
        @test_throws ArgumentError LobachevskySurrogate(x, y, 0.0, 5.0, n = 22)

        x_nd = [(1.0, 1.0), (2.0, 2.0), (3.0, 1.5)]
        y_nd = [1.0, 2.0, 3.0]
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        @test_throws ArgumentError LobachevskySurrogate(
            x_nd, y_nd, lb, ub, alpha = [5.0, 2.0]
        )
        @test_throws ArgumentError LobachevskySurrogate(x_nd, y_nd, lb, ub, n = 5)
    end

    @testset "1D fit and closed-form integral" begin
        obj = x -> 3 * x + log(x)
        a = 1.0
        b = 4.0
        x = sample(400, a, b, SobolSample())
        y = obj.(x)
        my_loba = LobachevskySurrogate(x, y, a, b, alpha = 2.0, n = 6)
        @test isapprox(my_loba(3.83), obj(3.83), rtol = 1.0e-3)

        int_1D = lobachevsky_integral(my_loba, a, b)
        # Against the true integral: this bounds interpolation error.
        @test abs(int_1D - quadgk(obj, a, b)[1]) < 1.0e-3
        # Against quadrature of the surrogate itself: this isolates the closed
        # form, which is exact. The tolerance is set by quadgk's own accuracy
        # on the piecewise-polynomial integrand, not by the closed form.
        @test isapprox(int_1D, quadgk(my_loba, a, b)[1], rtol = 1.0e-6)
        update!(my_loba, 3.7, 12.1)
        update!(my_loba, [1.23, 3.45], [5.2, 109.67])

        @test_throws ArgumentError my_loba(Float64[])
        @test_throws ArgumentError my_loba((2.0, 3.0, 4.0))
    end

    @testset "ND fit and closed-form integral" begin
        obj = x -> x[1] + log(x[2])
        lb = [0.0, 0.0]
        ub = [8.0, 8.0]
        x = sample(400, lb, ub, SobolSample())
        y = obj.(x)
        my_loba_ND = LobachevskySurrogate(x, y, lb, ub, alpha = [2.4, 2.4], n = 8)
        my_loba_kwargs = LobachevskySurrogate(x, y, lb, ub)
        pred = my_loba_ND((1.0, 2.0))
        @test isfinite(pred)

        int_ND = lobachevsky_integral(my_loba_ND, lb, ub)
        int_val_true = hcubature(obj, lb, ub)[1]
        @test abs(int_ND - int_val_true) < 1.0e-2 * abs(int_val_true)
        # Closed form vs. quadrature of the surrogate itself (exactness of the
        # tensor-product CDF formula, independent of interpolation error).
        int_surr = hcubature(v -> my_loba_ND((v[1], v[2])), lb, ub, reltol = 1.0e-8)[1]
        @test isapprox(int_ND, int_surr, rtol = 1.0e-5)
        update!(my_loba_ND, (7.2, 7.1), obj((7.2, 7.1)))
        update!(my_loba_ND, [(6.9, 7.5), (6.8, 7.4)], [obj((6.9, 7.5)), obj((6.8, 7.4))])

        @test_throws ArgumentError my_loba_ND(Float64[])
        @test_throws ArgumentError my_loba_ND(1.0)
        @test_throws ArgumentError my_loba_ND((2.0, 3.0, 4.0))
    end

    @testset "integrate_dimension does not mutate its inputs" begin
        obj = x -> x[1] + log(x[2])
        lb = [1.0, 1.0]
        ub = [8.0, 8.0]
        x = sample(60, lb, ub, SobolSample())
        y = obj.(x)
        my_loba_ND = LobachevskySurrogate(x, y, lb, ub, alpha = [2.4, 2.4], n = 4)

        alpha_before = copy(my_loba_ND.alpha)
        lb_before = copy(lb)
        ub_before = copy(ub)
        reduced = lobachevsky_integrate_dimension(my_loba_ND, lb, ub, 2)

        # The original surrogate and the caller's bounds are untouched
        @test my_loba_ND.alpha == alpha_before
        @test lb == lb_before
        @test ub == ub_before
        # The original surrogate is still usable in 2D
        @test isfinite(my_loba_ND((2.0, 3.0)))

        # Tuple bounds must work too (the implementation collects before
        # deleting, so it must not require a mutable vector)
        reduced_tuple = lobachevsky_integrate_dimension(
            my_loba_ND, (1.0, 1.0), (8.0, 8.0), 2
        )
        @test isapprox(reduced_tuple(2.0), reduced(2.0), rtol = 1.0e-10)

        # The reduced surrogate is 1D with scalar bounds and alpha
        @test reduced.alpha isa Number
        @test reduced.lb == 1.0
        @test reduced.ub == 8.0
        @test isfinite(reduced(2.0))

        # Marginal correctness: the reduced surrogate at x1 must equal the
        # integral of the full surrogate over the integrated dimension.
        marg, _ = quadgk(t -> my_loba_ND((2.0, t)), 1.0, 8.0)
        @test isapprox(reduced(2.0), marg, rtol = 1.0e-6)
    end

    @testset "integrate_dimension 3D -> 2D" begin
        obj = x -> x[1] + log(x[2]) + x[3]^2
        lb = [1.0, 1.0, 1.0]
        ub = [4.0, 4.0, 4.0]
        x = sample(40, lb, ub, SobolSample())
        y = obj.(x)
        my_loba_ND = LobachevskySurrogate(x, y, lb, ub)
        reduced = lobachevsky_integrate_dimension(my_loba_ND, lb, ub, 2)
        @test length(reduced.alpha) == 2
        @test reduced.lb == [1.0, 1.0]
        @test reduced.ub == [4.0, 4.0]

        # Marginal correctness, as in the 2D -> 1D case: the reduced surrogate
        # must equal the integral of the full surrogate over dimension 2.
        marg, _ = quadgk(t -> my_loba_ND((2.0, t, 3.0)), 1.0, 4.0)
        @test isapprox(reduced((2.0, 3.0)), marg, rtol = 1.0e-6)
    end

    @testset "integrate_dimension result is self-consistent" begin
        # The reduced surrogate stores the marginal values at its own nodes, so
        # refitting it must reproduce the marginal rather than the original
        # full-dimensional responses.
        obj = x -> x[1] + log(x[2])
        lb = [1.0, 1.0]
        ub = [8.0, 8.0]
        x = sample(40, lb, ub, SobolSample())
        y = obj.(x)
        s = LobachevskySurrogate(x, y, lb, ub, alpha = [2.4, 2.4], n = 4)
        reduced = lobachevsky_integrate_dimension(s, lb, ub, 2)

        @test all(isapprox(reduced(reduced.x[i]), reduced.y[i]) for i in eachindex(reduced.x))
        refit = LobachevskySurrogate(
            reduced.x, reduced.y, reduced.lb, reduced.ub,
            alpha = reduced.alpha, n = reduced.n
        )
        @test isapprox(refit(3.0), reduced(3.0), rtol = 1.0e-8)
    end

    @testset "sparse construction" begin
        obj = x -> 3 * x + log(x)
        a = 1.0
        b = 4.0
        x = sample(100, a, b, SobolSample())
        y = obj.(x)
        my_loba_sparse = LobachevskySurrogate(x, y, a, b, alpha = 2.0, n = 6, sparse = true)
        @test isapprox(my_loba_sparse(2.5), obj(2.5), rtol = 1.0e-2)

        obj_nd = x -> x[1] + log(x[2])
        lb = [1.0, 1.0]
        ub = [8.0, 8.0]
        x = sample(100, lb, ub, SobolSample())
        y = obj_nd.(x)
        my_loba_ND_sparse = LobachevskySurrogate(
            x, y, lb, ub, alpha = [2.4, 2.4], n = 8, sparse = true
        )
        @test isfinite(my_loba_ND_sparse((2.0, 3.0)))
    end
end
