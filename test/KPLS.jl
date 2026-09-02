using Surrogates
using Test
using LinearAlgebra

# Sphere function (sum of squares): easy analytical test
function sphere_function(x)
    return sum(x .^ 2)
end

# 3D sphere
let
    n = 100
    lb = [-5.0, -5.0, -5.0]
    ub = [5.0, 5.0, 5.0]
    x = sample(n, lb, ub, SobolSample())
    y = sphere_function.(x)
    n_test = 50
    x_test = sample(n_test, lb, ub, GoldenSample())
    y_true = sphere_function.(x_test)

    @testset "KPLS: 3D sphere, n_comp=2" begin
        n_comp = 2
        theta = [0.01 for _ in 1:n_comp]
        k = KPLS(x, y, n_comp, lb, ub, theta)
        y_pred = k.(x_test)
        rmse = sqrt(sum((y_pred .- y_true) .^ 2) / n_test)
        @test rmse < 6.0e-3
    end

    @testset "KPLS: 3D sphere, n_comp=1" begin
        n_comp = 1
        theta = [0.01 for _ in 1:n_comp]
        k = KPLS(x, y, n_comp, lb, ub, theta)
        y_pred = k.(x_test)
        rmse = sqrt(sum((y_pred .- y_true) .^ 2) / n_test)
        @test rmse < 6.0e-3
    end
end

# Water flow function (8D): tests high-dimensional reduction
function water_flow(x)
    r_w = x[1]; r = x[2]; T_u = x[3]; H_u = x[4]
    T_l = x[5]; H_l = x[6]; L = x[7]; K_w = x[8]
    log_val = log(r / r_w)
    return (2 * pi * T_u * (H_u - H_l)) /
        (log_val * (1 + (2 * L * T_u / (log_val * r_w^2 * K_w)) + T_u / T_l))
end

let
    n = 500
    lb = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
    ub = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]
    x = sample(n, lb, ub, SobolSample())
    y = water_flow.(x)
    n_test = 100
    x_test = sample(n_test, lb, ub, GoldenSample())
    y_true = water_flow.(x_test)

    @testset "KPLS: 8D water flow, n_comp=2" begin
        n_comp = 2
        theta = [0.01 for _ in 1:n_comp]
        k = KPLS(x, y, n_comp, lb, ub, theta)
        y_pred = k.(x_test)
        rmse = sqrt(sum((y_pred .- y_true) .^ 2) / n_test)
        @test rmse < 0.15
    end
end

# Test update!
@testset "KPLS: update! adds points" begin
    lb = [-5.0, -5.0, -5.0]
    ub = [5.0, 5.0, 5.0]
    n_comp = 2
    theta = [0.01, 0.01]

    # Small initial dataset
    x_init = [
        (1.0, 2.0, 3.0), (4.0, 4.0, 4.0), (-1.0, -2.0, -3.0),
        (2.0, -1.0, 1.0), (-3.0, 3.0, -1.0),
    ]
    y_init = sphere_function.(x_init)
    k = KPLS(x_init, y_init, n_comp, lb, ub, theta)

    n_test = 20
    x_test = sample(n_test, lb, ub, GoldenSample())
    y_true = sphere_function.(x_test)
    y_pred1 = k.(x_test)
    rmse1 = sqrt(sum((y_pred1 .- y_true) .^ 2) / n_test)

    # Add more points
    n_new = 80
    x_new = sample(n_new, lb, ub, SobolSample())
    y_new = sphere_function.(x_new)
    for i in 1:n_new
        update!(k, x_new[i], y_new[i])
    end
    y_pred2 = k.(x_test)
    rmse2 = sqrt(sum((y_pred2 .- y_true) .^ 2) / n_test)

    # More data should generally improve accuracy
    @test rmse2 < rmse1
    @test rmse2 < 8.0e-3
end

@testset "KPLS: constructor validation" begin
    lb, ub = [-1.0, -1.0], [1.0, 1.0]
    x = sample(12, lb, ub, SobolSample())
    y = sphere_function.(x)

    # One correlation scale per PLS component. Without this check the mismatch
    # surfaced as a `DimensionMismatch` from `squar_exp`'s `reshape`.
    @test_throws ArgumentError KPLS(x, y, 2, lb, ub, [1.0])
    @test_throws ArgumentError KPLS(x, y, 1, lb, ub, [1.0, 1.0])

    # PLS has at most `d` components; asking for more silently produced columns
    # of zeros and a meaningless correlation scale for each missing component.
    @test_throws ArgumentError KPLS(x, y, 3, lb, ub, [1.0, 1.0, 1.0])
    @test_throws ArgumentError KPLS(x, y, 0, lb, ub, Float64[])

    # Out-of-bounds training points used to print and return `nothing`.
    x_bad = vcat(x, [(5.0, 5.0)])
    y_bad = vcat(y, 50.0)
    @test_throws ArgumentError KPLS(x_bad, y_bad, 1, lb, ub, [1.0])
end

@testset "KPLS: update! leaves the caller's containers alone" begin
    lb, ub = [-1.0, -1.0], [1.0, 1.0]
    x = sample(12, lb, ub, SobolSample())
    y = sphere_function.(x)
    k = KPLS(x, y, 1, lb, ub, [1.0])

    update!(k, (0.31, 0.47), sphere_function((0.31, 0.47)))
    @test length(x) == 12
    @test length(y) == 12
    @test length(k.x) == 13
    @test size(k.x_matrix, 1) == 13

    # A duplicate is a no-op, and an out-of-bounds point is an error.
    @test_logs (:warn,) update!(k, (0.31, 0.47), sphere_function((0.31, 0.47)))
    @test length(k.x) == 13
    @test_throws ArgumentError update!(k, (5.0, 5.0), 50.0)
end

@testset "KPLS: prediction matches an independent ordinary-kriging solve" begin
    # The KPLS kernel is a Gaussian kernel on the standardized inputs with
    # squared PLS rotation coefficients folded in. Rebuilding it from the fitted
    # fields and solving the ordinary-kriging system directly must reproduce the
    # surrogate; this checks the whole prediction path, not just the fit.
    lb, ub = [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]
    g(x) = x[1]^2 + 0.5x[2] + sin(x[3])
    x = sample(25, lb, ub, SobolSample())
    y = g.(x)
    k = KPLS(x, y, 2, lb, ub, [1.0, 1.0])

    Xs, W, th = k.X_after_std, k.pls_mean, k.theta
    ker(a, b) = exp(
        -sum(
            th[l] * sum(W[q, l]^2 * (a[q] - b[q])^2 for q in axes(W, 1))
                for l in eachindex(th)
        )
    )
    nt = size(Xs, 1)
    R = [ker(Xs[i, :], Xs[j, :]) for i in 1:nt, j in 1:nt] + (1.0e6 * eps()) * I
    ys = vec((k.y_matrix .- k.y_mean) ./ k.y_std)
    ones_v = ones(nt)
    mu = dot(ones_v, R \ ys) / dot(ones_v, R \ ones_v)
    b = R \ (ys .- mu)

    x_test = sample(30, lb, ub, GoldenSample())
    for p in x_test
        ps = vec((collect(p)' .- k.X_offset) ./ k.X_scale)
        r = [ker(ps, Xs[i, :]) for i in 1:nt]
        @test k.y_mean + k.y_std * (mu + dot(r, b)) ≈ k(p) atol = 1.0e-10
    end
end
