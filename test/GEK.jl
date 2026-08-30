using Surrogates
using LinearAlgebra
using ForwardDiff
using Test

# Ordinary-kriging variance from the Lagrangian (augmented) system with the GEK
# trend basis, an independent derivation of what `std_error_at_point` reports.
function reference_gek_std_error(k, val)
    n = length(k.x)
    d = length(k.x[1])
    N = n * (1 + d)
    R = Matrix(k.R_fact)
    r = Surrogates._gek_r(k, val)
    f = Surrogates._gek_trend(n, d, Float64)
    sol = [R f; f' 0.0] \ [r; 1.0]
    w, λ = sol[1:N], -sol[N + 1]
    return sqrt(max(k.sigma * (1 - dot(r, w) + λ), 0.0))
end

@testset "the covariance blocks are the derivatives of the kernel" begin
    # Every block of the GEK matrix is a partial derivative of
    # k(x, x') = exp(-Σ θ_l Δ_l²); check each against a finite difference of
    # the kernel itself. The derivative-derivative block used to be missing its
    # `2θ_l δ_lm` term, which left the matrix indefinite.
    theta = [0.7, 0.3]
    kern = (a, b) -> exp(-sum(theta[l] * (a[l] - b[l])^2 for l in 1:2))
    a, b = [1.3, 0.4], [2.1, 1.9]
    x = [(a[1], a[2]), (b[1], b[2])]
    R = Surrogates._gek_covariance(x, theta)
    n, d = 2, 2

    @test R ≈ R'
    @test isposdef(Symmetric(R))
    @test R[1, 1] ≈ 1.0
    @test R[1, 2] ≈ kern(a, b)

    for l in 1:d
        # Cov(f(x₁), ∂ₗf(x₂)) = ∂k/∂x₂ˡ
        dk = ForwardDiff.gradient(v -> kern(a, v), b)[l]
        @test R[1, Surrogates._gek_deriv_index(n, d, 2, l)] ≈ dk
        # Cov(∂ₗf(x₁), f(x₂)) = ∂k/∂x₁ˡ
        dk1 = ForwardDiff.gradient(v -> kern(v, b), a)[l]
        @test R[Surrogates._gek_deriv_index(n, d, 1, l), 2] ≈ dk1
        for m in 1:d
            # Cov(∂ₗf(x₁), ∂ₘf(x₂)) = ∂²k/∂x₁ˡ∂x₂ᵐ
            d2k = ForwardDiff.gradient(
                u -> ForwardDiff.gradient(v -> kern(u, v), b)[m], a
            )[l]
            idx = (
                Surrogates._gek_deriv_index(n, d, 1, l),
                Surrogates._gek_deriv_index(n, d, 2, m),
            )
            @test R[idx...] ≈ d2k
        end
    end
end

@testset "1D" begin
    lb = 0.0
    ub = 5.0
    f = t -> t^3 - 6t^2 + 4t + 12
    df = t -> 3t^2 - 12t + 4
    x = collect(range(0.5, 4.5, length = 6))
    y = vcat(f.(x), df.(x))
    my_gek = GEK(x, y, lb, ub, theta = 0.3)

    @testset "reproduces the values and the slopes it was given" begin
        # The defining property of gradient-enhanced kriging. It held for
        # neither before: the predictor dropped `theta` from the correlation
        # function and summed only the value half of the weight vector, so the
        # derivative observations never reached the prediction.
        for xi in x
            @test my_gek(xi) ≈ f(xi) atol = 1.0e-2
            @test ForwardDiff.derivative(my_gek, xi) ≈ df(xi) atol = 1.0e-2
        end
        @test maximum(std_error_at_point(my_gek, xi) for xi in x) < 1.0e-2
    end

    @testset "is accurate between the samples" begin
        probe = range(0.6, 4.4, length = 40)
        rms = sqrt(sum((my_gek(v) - f(v))^2 for v in probe) / length(probe))
        @test rms < 1.0e-2
    end

    @testset "the mean squared error is the kriging variance" begin
        # A shorter correlation length than the surrogate above uses, so the
        # covariance matrix is well conditioned and the standard errors are
        # genuinely non-zero: the reference solves a dense augmented system where
        # `std_error_at_point` solves against a Cholesky factor, and comparing
        # two round-off-dominated zeros would test nothing.
        g = GEK(x, y, lb, ub; theta = 2.0)
        for v in range(0.6, 4.4, length = 17)
            @test std_error_at_point(g, v) ≈
                reference_gek_std_error(g, v) atol = 1.0e-8
        end
        @test maximum(std_error_at_point(g, v) for v in range(0.6, 4.4, length = 17)) >
            1.0e-3
    end

    @testset "input dimension validation" begin
        @test_throws ArgumentError my_gek(Float64[])
        @test_throws ArgumentError my_gek((2.0, 3.0, 4.0))
    end

    @testset "update! carries the gradient" begin
        g = GEK(x, y, lb, ub, theta = 0.3)
        update!(g, 2.5, f(2.5), df(2.5))
        @test length(g.x) == 7
        @test length(g.y) == 14
        @test g(2.5) ≈ f(2.5) atol = 1.0e-2
        # A new sample without its gradient cannot be placed in the covariance
        # system; the three-argument form used to splice it in and silently
        # leave the observation vector and the block layout out of step.
        @test_throws ArgumentError update!(g, 3.5, f(3.5))
    end

    @testset "update! leaves the caller's containers alone" begin
        xc = copy(x)
        yc = copy(y)
        g = GEK(xc, yc, lb, ub, theta = 0.3)
        update!(g, 2.5, f(2.5), df(2.5))
        @test xc == x
        @test yc == y
    end
end

@testset "ND" begin
    lb = [0.0, 0.0]
    ub = [3.0, 3.0]
    F = p -> p[1]^2 + p[2]^2 + 0.5 * p[1] * p[2]
    G = p -> [2p[1] + 0.5p[2], 2p[2] + 0.5p[1]]
    x = [(0.5, 0.5), (2.5, 0.7), (1.2, 2.4), (2.8, 2.6), (1.7, 1.3), (0.4, 2.1)]
    y = vcat([F(p) for p in x], reduce(vcat, [G(p) for p in x]))
    my_gek_ND = GEK(x, y, lb, ub, theta = [0.2, 0.2])

    @testset "reproduces the values and the gradients it was given" begin
        # The value-value block was `exp(-θ(xᵢ - xⱼ))` with no `abs(·)^p`, so it
        # was not even symmetric and had entries above one. The model missed its
        # own samples by ~7.6e15.
        for p in x
            @test my_gek_ND(p) ≈ F(p) atol = 1.0e-8
            @test ForwardDiff.gradient(v -> my_gek_ND(v), collect(p)) ≈
                G(p) atol = 1.0e-8
        end
        @test maximum(std_error_at_point(my_gek_ND, p) for p in x) < 1.0e-6
    end

    @testset "the mean squared error is the kriging variance" begin
        g = GEK(x, y, lb, ub; theta = [1.0, 1.0])
        probe = [(1.0, 1.0), (2.0, 1.5), (0.8, 2.2), (2.2, 2.2)]
        for p in probe
            @test std_error_at_point(g, p) ≈
                reference_gek_std_error(g, p) atol = 1.0e-8
        end
        @test maximum(std_error_at_point(g, p) for p in probe) > 1.0e-3
    end

    @testset "input dimension validation" begin
        @test_throws ArgumentError my_gek_ND(Float64[])
        @test_throws ArgumentError my_gek_ND(2.0)
        @test_throws ArgumentError my_gek_ND((2.0, 3.0, 4.0))
    end

    @testset "queries in any container" begin
        val = (1.0, 1.5)
        @test my_gek_ND(collect(val)) ≈ my_gek_ND(val)
        # The ND tutorial plots with `my_GEK([x y])`, a 1 x d matrix.
        @test my_gek_ND([val[1] val[2]]) ≈ my_gek_ND(val)
    end

    @testset "update! carries the gradient" begin
        g = GEK(x, y, lb, ub, theta = [0.2, 0.2])
        new_p = (2.0, 2.0)
        update!(g, new_p, F(new_p), G(new_p))
        @test length(g.x) == 7
        @test length(g.y) == 21
        @test g(new_p) ≈ F(new_p) atol = 1.0e-8
        # The observation vector keeps all values ahead of all gradients.
        @test g.y[1:7] ≈ [F(p) for p in g.x]
        @test_throws ArgumentError update!(g, (0.9, 0.9), F((0.9, 0.9)))
    end
end

@testset "hyperparameter and sample validation" begin
    lb, ub = 0.0, 5.0
    f = t -> t^2
    df = t -> 2t
    x = collect(range(0.5, 4.5, length = 5))
    y = vcat(f.(x), df.(x))

    # The derivative blocks are second derivatives of the correlation function,
    # which the power-exponential kernel only has for p = 2. The old default was
    # p = 1, whose kernel has a cusp at the origin.
    @test_throws ArgumentError GEK(x, y, lb, ub, p = 1.0)
    @test_throws ArgumentError GEK(x, y, lb, ub, p = 0.03)
    @test_throws ArgumentError GEK(x, y, lb, ub, theta = -1.0)
    @test GEK(x, y, lb, ub; optimize_theta = false).p == 2.0

    xd = [1.0, 2.0, 2.0, 3.0]
    yd = vcat(f.(xd), df.(xd))
    @test_throws ArgumentError GEK(xd, yd, lb, ub)

    # A malformed observation vector used to build a mismatched block layout.
    @test_throws ArgumentError GEK(x, f.(x), lb, ub)

    g = GEK(x, y, lb, ub)
    @test_logs (:warn,) update!(g, x[2], f(x[2]), df(x[2]))
    @test length(g.x) == 5

    lb2, ub2 = [0.0, 0.0], [3.0, 3.0]
    x2 = [(0.5, 0.5), (2.5, 0.7), (1.2, 2.4)]
    y2 = vcat([p[1] + p[2] for p in x2], reduce(vcat, [[1.0, 1.0] for _ in x2]))
    @test_throws ArgumentError GEK(x2, y2, lb2, ub2, p = [1.0, 1.0])
    @test_throws ArgumentError GEK(x2, y2, lb2, ub2, theta = [-1.0, 1.0])
end

@testset "Float32 is not promoted to Float64" begin
    x = collect(Float32.(range(0.5, 4.5, length = 6)))
    f = t -> t^3 - 6t^2 + 4t + 12
    df = t -> 3t^2 - 12t + 4
    g = GEK(x, vcat(f.(x), df.(x)), 0.0f0, 5.0f0, theta = 0.3f0)
    @test g.p isa Float32
    @test eltype(g.R_fact) === Float32
    @test g(2.5f0) isa Float32
    @test std_error_at_point(g, 2.5f0) isa Float32

    xs = Tuple{Float32, Float32}[
        (0.5, 0.5), (2.5, 0.7), (1.2, 2.4), (2.8, 2.6), (1.7, 1.3), (0.4, 2.1),
    ]
    F = p -> p[1]^2 + p[2]^2 + 0.5f0 * p[1] * p[2]
    G = p -> [2p[1] + 0.5f0 * p[2], 2p[2] + 0.5f0 * p[1]]
    y = vcat([F(p) for p in xs], reduce(vcat, [G(p) for p in xs]))
    gn = GEK(xs, y, Float32[0, 0], Float32[3, 3], theta = Float32[0.2, 0.2])
    @test eltype(gn.p) === Float32
    @test eltype(gn.R_fact) === Float32
    @test gn((1.0f0, 1.5f0)) isa Float32
    @test maximum(abs(gn(p) - F(p)) for p in xs) < 1.0e-3
end

@testset "inverse_of_R is deprecated but still R inverse" begin
    lb, ub = 0.0, 5.0
    x = collect(range(0.5, 4.5, length = 4))
    y = vcat(x .^ 2, 2 .* x)
    g = GEK(x, y, lb, ub, theta = 0.3)
    Rinv = @test_deprecated g.inverse_of_R
    @test Rinv * Matrix(g.R_fact) ≈ I
end

@testset "theta is fitted by maximum likelihood" begin
    lb, ub = [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]
    F = p -> sum(abs2, p)
    Gr = p -> 2 .* collect(p)
    x = sample(20, lb, ub, SobolSample())
    y = vcat([F(p) for p in x], reduce(vcat, [Gr(p) for p in x]))
    xt = sample(150, lb, ub, HaltonSample())
    yt = [F(p) for p in xt]
    err(k) = sqrt(sum((k(p) - yt[i])^2 for (i, p) in enumerate(xt)) / length(xt))

    fixed = GEK(x, y, lb, ub; theta = [1.0, 1.0, 1.0])
    fitted = GEK(x, y, lb, ub)

    @test fitted.optimize_theta
    @test !fixed.optimize_theta
    @test Surrogates._gek_loglik(x, y, fitted.theta) >
        Surrogates._gek_loglik(x, y, fixed.theta)
    # The former fixed default of 1.0 was far from the data's own scale.
    @test err(fitted) < err(fixed) / 10

    @testset "the starting scale now comes from the data" begin
        k = GEK(x, y, lb, ub; optimize_theta = false)
        @test k.theta ≈ Surrogates._kriging_default_theta(x, lb, ub, k.p)
        @test all(k.theta .!= 1.0)
    end

    @testset "extreme scales score badly, they do not raise" begin
        best = Surrogates._gek_loglik(x, y, fitted.theta)
        @test isfinite(best)
        for theta in ([1.0e-30, 1.0e-30, 1.0e-30], [1.0e30, 1.0e30, 1.0e30])
            v = Surrogates._gek_loglik(x, y, theta)
            @test v isa Real
            @test v < best
        end
        @test Surrogates._gek_loglik(x, y, [Inf, Inf, Inf]) == -Inf
    end

    @testset "update! refits" begin
        k = GEK(x, y, lb, ub)
        before = copy(k.theta)
        new_p = (1.0, 2.0, 3.0)
        update!(k, new_p, F(new_p), Gr(new_p))
        @test length(k.x) == 21
        @test k.theta != before
        @test k(new_p) ≈ F(new_p) rtol = 1.0e-3
    end

    @testset "one dimension" begin
        f1 = t -> t^3 - 6t^2 + 4t + 12
        df1 = t -> 3t^2 - 12t + 4
        x1 = collect(range(0.5, 4.5, length = 8))
        y1 = vcat(f1.(x1), df1.(x1))
        kf = GEK(x1, y1, 0.0, 5.0)
        @test kf.theta isa Number
        @test Surrogates._gek_loglik(x1, y1, kf.theta) >
            Surrogates._gek_loglik(x1, y1, 1.0)
    end
end
