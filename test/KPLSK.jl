using Surrogates
using Test

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

    @testset "KPLSK: 3D sphere, n_comp=2" begin
        n_comp = 2
        theta = [0.01 for _ in 1:n_comp]
        k = KPLSK(x, y, n_comp, lb, ub, theta)
        y_pred = k.(x_test)
        rmse = sqrt(sum((y_pred .- y_true) .^ 2) / n_test)
        @test rmse < 6e-3
    end

    @testset "KPLSK: 3D sphere, n_comp=1" begin
        n_comp = 1
        theta = [0.01 for _ in 1:n_comp]
        k = KPLSK(x, y, n_comp, lb, ub, theta)
        y_pred = k.(x_test)
        rmse = sqrt(sum((y_pred .- y_true) .^ 2) / n_test)
        @test rmse < 6e-3
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

    @testset "KPLSK: 8D water flow, n_comp=2" begin
        n_comp = 2
        theta = [0.01 for _ in 1:n_comp]
        k = KPLSK(x, y, n_comp, lb, ub, theta)
        y_pred = k.(x_test)
        rmse = sqrt(sum((y_pred .- y_true) .^ 2) / n_test)
        @test rmse < 0.15
    end
end

# Test update!
@testset "KPLSK: update! adds points" begin
    lb = [-5.0, -5.0, -5.0]
    ub = [5.0, 5.0, 5.0]
    n_comp = 2
    theta = [0.01, 0.01]

    # Small initial dataset
    x_init = [(1.0, 2.0, 3.0), (4.0, 4.0, 4.0), (-1.0, -2.0, -3.0),
              (2.0, -1.0, 1.0), (-3.0, 3.0, -1.0)]
    y_init = sphere_function.(x_init)
    k = KPLSK(x_init, y_init, n_comp, lb, ub, theta)

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
    @test rmse2 < 8e-3
end
