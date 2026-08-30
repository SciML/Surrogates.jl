using LinearAlgebra
using Surrogates
using Test
using Statistics

# Ordinary-kriging variance from the Lagrangian (augmented) system, an
# independent derivation of the quantity `std_error_at_point` reports:
#
#   [R 𝟙; 𝟙ᵀ 0] [w; -λ] = [r; 1],   s²(x) = σ² (1 - rᵀw + λ)
#
# `R` is taken from the surrogate's own factorization so the nugget matches;
# only the closed form under test is independent.
function reference_std_error(k, val)
    R = Matrix(k.R_fact)
    n = size(R, 1)
    d = length(k.x[1])
    r = [
        exp(-sum(k.theta[l] * abs(val[l] - k.x[i][l])^k.p[l] for l in 1:d))
            for i in 1:n
    ]
    o = ones(n)
    sol = [R o; o' 0.0] \ [r; 1.0]
    w, λ = sol[1:n], -sol[n + 1]
    return sqrt(max(k.sigma * (1 - dot(r, w) + λ), 0.0))
end

@testset "1D" begin
    lb = 0.0
    ub = 10.0
    f = x -> log(x) * exp(x)
    x = sample(5, lb, ub, SobolSample())
    y = f.(x)
    my_p = 1.9

    @testset "hyperparameter validation" begin
        @test_throws ArgumentError Kriging(x, y, lb, ub, p = -1.0)
        @test_throws ArgumentError Kriging(x, y, lb, ub, p = 3.0)
        @test_throws ArgumentError Kriging(x, y, lb, ub, theta = -2.0)
        # p = 0 makes every off-diagonal correlation exp(-θ), so R is singular
        # for more than two samples.
        @test_throws ArgumentError Kriging(x, y, lb, ub, p = 0.0)
    end

    my_k = Kriging(x, y, lb, ub, p = my_p, optimize_theta = false)
    @test my_k.theta ≈ 0.5 * std(x)^(-my_p)

    @testset "input dimension validation" begin
        @test_throws ArgumentError my_k(rand(3))
        @test_throws ArgumentError my_k(Float64[])
    end

    update!(my_k, 4.0, 75.68)
    update!(my_k, [5.0, 6.0], [238.86, 722.84])
    pred = my_k(5.5)

    @test 238.86 ≤ pred ≤ 722.84
    @test my_k(5.0) ≈ 238.86
    @test std_error_at_point(my_k, 5.0) < 1.0e-3 * my_k(5.0)

    @testset "without update!" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        k = Kriging(x, y, lb, ub, p = 1.3)
        @test k(1.0) == 4.0
        @test std_error_at_point(k, 1.0) < 10^(-6)
    end

    @testset "update! with a single sample" begin
        k = Kriging([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lb, ub, p = 1.3)
        update!(k, 4.0, 9.0)
        @test k(4.0) ≈ 9.0
        @test std_error_at_point(k, 4.0) < 10^(-6)
    end

    @testset "update! with several samples" begin
        k = Kriging([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lb, ub, p = 1.3)
        update!(k, [4.0, 5.0, 6.0], [9.0, 13.0, 15.0])
        @test k(4.0) ≈ 9.0
        @test std_error_at_point(k, 4.0) < 10^(-6)
    end

    @testset "hyperparameter initialization" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        k = Kriging(x, y, lb, ub, optimize_theta = false)
        p_expected = 2.0
        @test k.p == p_expected
        @test k.theta == 0.5 / std(x)^p_expected
    end
end

@testset "ND" begin
    lb = [0.0, 0.0, 1.0]
    ub = [5.0, 7.5, 10.0]
    x = sample(5, lb, ub, SobolSample())
    f = x -> x[1] + x[2] * x[3]
    y = f.(x)
    my_theta = [2.0, 2.0, 2.0]
    my_p = [1.9, 1.9, 1.9]
    my_k = Kriging(x, y, lb, ub, p = my_p, theta = my_theta)
    update!(my_k, (4.0, 3.2, 9.5), 34.4)
    update!(my_k, [(1.0, 4.65, 6.4), (2.3, 5.4, 6.7)], [30.76, 38.48])
    @test my_k((3.5, 5.5, 6.5)) isa Number

    unit_p = [1.0, 1.0, 1.0]
    base_x = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    base_y = [1.0, 2.0, 3.0]

    @testset "without update!" begin
        k = Kriging(base_x, base_y, lb, ub, p = unit_p, theta = my_theta)
        @test k((1.0, 2.0, 3.0)) ≈ 1.0
        @test std_error_at_point(k, (1.0, 2.0, 3.0)) < 10^(-6)
    end

    @testset "update! with a single sample" begin
        k = Kriging(base_x, base_y, lb, ub, p = unit_p, theta = my_theta)
        update!(k, (10.0, 11.0, 12.0), 4.0)
        @test k((10.0, 11.0, 12.0)) ≈ 4.0
        @test std_error_at_point(k, (10.0, 11.0, 12.0)) < 10^(-6)
    end

    @testset "update! with several samples" begin
        k = Kriging(base_x, base_y, lb, ub, p = unit_p, theta = my_theta)
        update!(k, [(10.0, 11.0, 12.0), (13.0, 14.0, 15.0)], [4.0, 5.0])
        @test k((10.0, 11.0, 12.0)) ≈ 4.0
        @test std_error_at_point(k, (10.0, 11.0, 12.0)) < 10^(-6)
    end

    kwarg_krig_ND = Kriging(base_x, base_y, lb, ub, optimize_theta = false)

    @testset "hyperparameter validation" begin
        @test_throws ArgumentError Kriging(base_x, base_y, lb, ub, p = 3 * my_p)
        @test_throws ArgumentError Kriging(base_x, base_y, lb, ub, p = -my_p)
        @test_throws ArgumentError Kriging(base_x, base_y, lb, ub, theta = -my_theta)
        @test_throws ArgumentError Kriging(base_x, base_y, lb, ub, p = 0 * my_p)
    end

    @testset "input dimension validation" begin
        @test_throws ArgumentError kwarg_krig_ND(1.0)
        @test_throws ArgumentError kwarg_krig_ND([1.0])
        @test_throws ArgumentError kwarg_krig_ND([2.0, 3.0])
        @test_throws ArgumentError kwarg_krig_ND(ones(5))
    end

    @testset "hyperparameter initialization" begin
        p_expected = 2.0
        @test all(==(p_expected), kwarg_krig_ND.p)
        @test all(
            kwarg_krig_ND.theta .≈
                [0.5 / std(x_i[ℓ] for x_i in base_x)^p_expected for ℓ in 1:3]
        )
    end

    @testset "queries in any container" begin
        k = Kriging(base_x, base_y, lb, ub, p = unit_p, theta = my_theta)
        val = (2.0, 3.0, 4.5)
        @test k(collect(val)) ≈ k(val)
        # Bounds written as row matrices make `(lb .+ ub) ./ 2` a 1 x d matrix.
        @test k(reshape(collect(val), 1, 3)) ≈ k(val)
        @test std_error_at_point(k, collect(val)) ≈ std_error_at_point(k, val)
    end
end

@testset "the mean squared error is the ordinary-kriging variance" begin
    # The trend-estimation term is `(1 - 𝟙ᵀR⁻¹r)²`, not `(1 - rᵀR⁻¹r)²`
    # (Jones 2001, eq. 5). Squaring the prediction residual instead was wrong
    # everywhere except at a sample and infinitely far from every sample.
    lb, ub = 0.0, 10.0
    f = x -> log(x + 1) * exp(x / 4)
    x = sample(12, lb, ub, SobolSample())
    k = Kriging(x, f.(x), lb, ub, p = 1.9)
    for val in range(0.05, 9.95, length = 23)
        @test std_error_at_point(k, val) ≈ reference_std_error(k, val) atol = 1.0e-8
    end

    lb3, ub3 = [0.0, 0.0, 1.0], [5.0, 7.5, 10.0]
    x3 = sample(25, lb3, ub3, SobolSample())
    g = p -> p[1] + p[2] * p[3]
    k3 = Kriging(x3, g.(x3), lb3, ub3, p = [1.9, 1.9, 1.9], theta = [2.0, 2.0, 2.0])
    for val in sample(15, lb3, ub3, HaltonSample())
        @test std_error_at_point(k3, val) ≈ reference_std_error(k3, val) atol = 1.0e-8
    end

    # The variance vanishes at a sample and is positive between samples.
    @test std_error_at_point(k, x[4]) < 1.0e-8
    @test std_error_at_point(k, (x[4] + x[5]) / 2) > 1.0e-8
end

@testset "duplicate samples" begin
    lb, ub = 0.0, 10.0
    @test_throws ArgumentError Kriging(
        [1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0], lb, ub
    )
    lb3, ub3 = [0.0, 0.0], [5.0, 5.0]
    dup_x = [(1.0, 2.0), (3.0, 4.0), (1.0, 2.0)]
    @test_throws ArgumentError Kriging(dup_x, [1.0, 2.0, 3.0], lb3, ub3)

    # `update!` warns and leaves the surrogate alone: the observation is already
    # there, and optimizers re-propose points near convergence.
    k = Kriging([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lb, ub)
    before = k(2.5)
    @test_logs (:warn,) update!(k, 2.0, 5.0)
    @test length(k.x) == 3
    @test k(2.5) == before
    # A repetition inside a batch is caught too.
    @test_logs (:warn,) update!(k, [4.0, 4.0], [7.0, 7.0])
    @test length(k.x) == 3
end

@testset "update! leaves the caller's containers alone" begin
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    k = Kriging(x, y, 0.0, 10.0)
    update!(k, 4.0, 9.0)
    @test x == [1.0, 2.0, 3.0]
    @test y == [4.0, 5.0, 6.0]
    @test length(k.x) == 4
end

@testset "the default correlation scale is re-derived by update!" begin
    # With fitting switched off, the data-derived heuristic still tracks the
    # sample spread as points are added.
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    k = Kriging(x, y, 0.0, 10.0; optimize_theta = false)
    update!(k, 9.0, 20.0)
    @test k.theta ≈ 0.5 / std(k.x)^2.0
    @test k.theta != 0.5 / std(x)^2.0

    # An explicit theta is a modelling choice and survives update!.
    k_fixed = Kriging(x, y, 0.0, 10.0; theta = 0.37)
    update!(k_fixed, 9.0, 20.0)
    @test k_fixed.theta == 0.37
end

@testset "near-duplicate samples stay solvable" begin
    # The nugget is sized from the maximum allowed condition number, so samples
    # may crowd together without the correlation matrix going singular.
    x = [1.0, 2.0, 2.0 + 1.0e-9, 3.0]
    y = [4.0, 5.0, 5.0, 6.0]
    k = Kriging(x, y, 0.0, 10.0)
    @test isfinite(k(2.5))
    @test std_error_at_point(k, 2.5) ≥ 0.0
end

@testset "element types" begin
    @testset "BigFloat" begin
        # The nugget criterion needs LAPACK eigenvalues, so for a non-BLAS
        # element type it is skipped and the factorization regularizes instead.
        x = BigFloat[1.0, 2.0, 3.0, 4.0, 5.0]
        y = BigFloat[0.5, 1.2, 2.1, 2.8, 3.6]
        k = Kriging(x, y, BigFloat(0.0), BigFloat(6.0))
        @test k(BigFloat(2.5)) isa BigFloat
        @test k(BigFloat(3.0)) ≈ y[3]
        @test std_error_at_point(k, BigFloat(2.5)) isa BigFloat

        xn = [
            (BigFloat(1.0), BigFloat(2.0)), (BigFloat(3.0), BigFloat(1.0)),
            (BigFloat(2.0), BigFloat(4.0)), (BigFloat(4.0), BigFloat(3.0)),
        ]
        kn = Kriging(
            xn, BigFloat[1.0, 2.0, 3.0, 4.0],
            BigFloat[0.0, 0.0], BigFloat[5.0, 5.0]
        )
        @test kn((BigFloat(2.0), BigFloat(2.0))) isa BigFloat
    end

    @testset "Float32 is not promoted to Float64" begin
        # `p`, the default `theta`, and the condition-number bound used to be
        # Float64 literals, each of which promoted the whole solve.
        x = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
        y = Float32[0.5, 1.2, 2.1, 2.8, 3.6]
        k = Kriging(x, y, 0.0f0, 6.0f0)
        @test k.p isa Float32
        @test k.theta isa Float32
        @test eltype(k.R_fact) === Float32
        @test k(2.5f0) isa Float32
        @test std_error_at_point(k, 2.5f0) isa Float32
        @test k(3.0f0) ≈ y[3]

        xn = [(1.0f0, 2.0f0), (3.0f0, 1.0f0), (2.0f0, 4.0f0), (4.0f0, 3.0f0)]
        kn = Kriging(xn, Float32[1, 2, 3, 4], Float32[0, 0], Float32[5, 5])
        @test eltype(kn.p) === Float32
        @test eltype(kn.theta) === Float32
        @test eltype(kn.R_fact) === Float32
        @test kn((2.0f0, 2.0f0)) isa Float32
        @test std_error_at_point(kn, (2.0f0, 2.0f0)) isa Float32
        @test kn((3.0f0, 1.0f0)) ≈ 2.0f0
    end

    @testset "integer samples" begin
        k = Kriging([1, 2, 3, 4], [1.0, 4.0, 9.0, 16.0], 0, 5)
        @test isfinite(k(2.5))
        @test k(3) ≈ 9.0
    end
end

@testset "inverse_of_R is deprecated but still R inverse" begin
    # An explicit, moderate scale: the assertion is about the deprecation shim,
    # and a fitted scale leaves R conditioned near the allowed maximum, where
    # forming the inverse at all loses most of its digits — which is the reason
    # the property is deprecated.
    k = Kriging([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 0.0, 10.0; theta = 1.0)
    Rinv = @test_deprecated k.inverse_of_R
    @test Rinv * Matrix(k.R_fact) ≈ I
end

@testset "theta is fitted by maximum likelihood" begin
    branin = p -> begin
        b, c, t = 5.1 / (4pi^2), 5 / pi, 1 / (8pi)
        (p[2] - b * p[1]^2 + c * p[1] - 6)^2 + 10 * (1 - t) * cos(p[1]) + 10
    end
    lb, ub = [-5.0, 0.0], [10.0, 15.0]
    x = sample(50, lb, ub, SobolSample())
    y = branin.(x)
    xt = sample(300, lb, ub, HaltonSample())
    yt = branin.(xt)
    err(k) = sqrt(sum((k(p) - yt[i])^2 for (i, p) in enumerate(xt)) / length(xt))

    heuristic = Kriging(x, y, lb, ub; optimize_theta = false)
    fitted = Kriging(x, y, lb, ub)

    @test fitted.optimize_theta
    @test !heuristic.optimize_theta
    # The likelihood is what selects theta, so it must improve; the prediction
    # error is a consequence, not the criterion.
    @test Surrogates._kriging_loglik(x, y, fitted.p, fitted.theta) >
        Surrogates._kriging_loglik(x, y, heuristic.p, heuristic.theta)
    @test err(fitted) < err(heuristic)
    # The fitted scale is anisotropic where the heuristic is nearly isotropic.
    @test fitted.theta[1] / fitted.theta[2] > 10

    @testset "an explicitly supplied theta is used as given" begin
        k = Kriging(x, y, lb, ub; theta = [0.3, 0.7])
        @test k.theta == [0.3, 0.7]
        @test !k.optimize_theta
        # ... and opting in explicitly still fits it.
        kf = Kriging(x, y, lb, ub; theta = [0.3, 0.7], optimize_theta = true)
        @test kf.theta != [0.3, 0.7]
        @test Surrogates._kriging_loglik(x, y, kf.p, kf.theta) >
            Surrogates._kriging_loglik(x, y, kf.p, [0.3, 0.7])
    end

    @testset "update! refits" begin
        k = Kriging(x, y, lb, ub)
        before = copy(k.theta)
        new_p = (1.0, 2.0)
        update!(k, new_p, branin(new_p))
        @test length(k.x) == 51
        @test k.theta != before
        @test k(new_p) ≈ branin(new_p) rtol = 1.0e-3
    end

    @testset "the search is bounded and never returns a worse scale" begin
        # Every start failing must leave the starting scale in place rather than
        # a value the likelihood never endorsed.
        theta0 = [0.05, 0.05]
        fit = Surrogates._fit_kriging_theta(x, y, [2.0, 2.0], theta0; n_start = 2)
        @test all(fit .> 0)
        @test all(fit .<= theta0 .* 10.0^Surrogates._KRIGING_THETA_DECADES)
        @test all(fit .>= theta0 .* 10.0^(-Surrogates._KRIGING_THETA_DECADES))
        @test Surrogates._kriging_loglik(x, y, [2.0, 2.0], fit) >=
            Surrogates._kriging_loglik(x, y, [2.0, 2.0], theta0)
    end

    @testset "extreme scales score badly, they do not raise" begin
        # The search routinely visits scales that drive R to the all-ones limit
        # (where the symmetric eigensolver can fail to converge) or to the
        # identity. Both must return a number the search can compare, never an
        # exception, and neither may beat the fitted scale.
        best = Surrogates._kriging_loglik(x, y, fitted.p, fitted.theta)
        @test isfinite(best)
        for theta in ([1.0e-30, 1.0e-30], [1.0e30, 1.0e30], [1.0e-30, 1.0e30])
            v = Surrogates._kriging_loglik(x, y, [2.0, 2.0], theta)
            @test v isa Real
            @test v < best
        end
        # A correlation matrix with non-finite entries is rejected outright.
        @test Surrogates._kriging_loglik(x, y, [2.0, 2.0], [Inf, Inf]) == -Inf
    end

    @testset "one dimension" begin
        f1 = t -> t^2 + 3sin(t)
        x1 = sample(30, 0.0, 10.0, SobolSample())
        y1 = f1.(x1)
        kh = Kriging(x1, y1, 0.0, 10.0; optimize_theta = false)
        kf = Kriging(x1, y1, 0.0, 10.0)
        @test kf.theta isa Number
        @test Surrogates._kriging_loglik(x1, y1, kf.p, kf.theta) >
            Surrogates._kriging_loglik(x1, y1, kh.p, kh.theta)
    end
end
