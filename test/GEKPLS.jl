using Surrogates
using Zygote
using Test

# Accuracy assertions here are one-sided bounds with the measured value in a
# comment. They used to be two-sided `isapprox(rmse, pinned, atol)` bands, which
# fail when the model gets *better* — three of them were failing for exactly
# that reason, while their own comments still recorded the older, lower numbers.

# #water flow function tests
function water_flow(x)
    r_w = x[1]
    r = x[2]
    T_u = x[3]
    H_u = x[4]
    T_l = x[5]
    H_l = x[6]
    L = x[7]
    K_w = x[8]
    log_val = log(r / r_w)
    return (2 * pi * T_u * (H_u - H_l)) /
        (log_val * (1 + (2 * L * T_u / (log_val * r_w^2 * K_w)) + T_u / T_l))
end

n = 1000
lb = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
ub = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]
x = sample(n, lb, ub, SobolSample())
grads = gradient.(water_flow, x)
y = water_flow.(x)
n_test = 100
x_test = sample(n_test, lb, ub, GoldenSample())
y_true = water_flow.(x_test)

@testset "Test 1: Water Flow Function Test (dimensions = 8; n_comp = 2; extra_points = 2)" begin
    n_comp = 2
    delta_x = 0.0001
    extra_points = 2
    initial_theta = [0.01 for i in 1:n_comp]
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    y_pred = g.(x_test)
    rmse = sqrt(sum(((y_pred - y_true) .^ 2) / n_test))
    @test rmse < 0.03  # 0.0221
end

@testset "Test 2: Water Flow Function Test (dimensions = 8; n_comp = 3; extra_points = 2)" begin
    n_comp = 3
    delta_x = 0.0001
    extra_points = 2
    initial_theta = [0.01 for i in 1:n_comp]
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    y_pred = g.(x_test)
    rmse = sqrt(sum(((y_pred - y_true) .^ 2) / n_test))
    @test rmse < 0.03  # 0.0219
end

@testset "Test 3: Water Flow Function Test (dimensions = 8; n_comp = 3; extra_points = 3)" begin
    n_comp = 3
    delta_x = 0.0001
    extra_points = 3
    initial_theta = [0.01 for i in 1:n_comp]
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    y_pred = g.(x_test)
    rmse = sqrt(sum(((y_pred - y_true) .^ 2) / n_test))
    @test rmse < 0.03  # 0.0219
end

# ## welded beam tests
function welded_beam(x)
    h = x[1]
    l = x[2]
    t = x[3]
    a = 6000 / (sqrt(2) * h * l)
    b = (6000 * (14 + 0.5 * l) * sqrt(0.25 * (l^2 + (h + t)^2))) /
        (2 * (0.707 * h * l * (l^2 / 12 + 0.25 * (h + t)^2)))
    return (sqrt(a^2 + b^2 + l * a * b)) / (sqrt(0.25 * (l^2 + (h + t)^2)))
end

n = 1000
lb = [0.125, 5.0, 5.0]
ub = [1.0, 10.0, 10.0]
x = sample(n, lb, ub, SobolSample())
grads = gradient.(welded_beam, x)
y = welded_beam.(x)
n_test = 100
x_test = sample(n_test, lb, ub, GoldenSample())
y_true = welded_beam.(x_test)

@testset "Test 4: Welded Beam Function Test (dimensions = 3; n_comp = 3; extra_points = 2)" begin
    n_comp = 3
    delta_x = 0.0001
    extra_points = 2
    initial_theta = [0.01 for i in 1:n_comp]
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    y_pred = g.(x_test)
    rmse = sqrt(sum(((y_pred - y_true) .^ 2) / n_test))
    @test rmse < 26.0  # 23.08
end

rmse_two_extra_points = 0.0

@testset "Test 5: Welded Beam Function Test (dimensions = 3; n_comp = 2; extra_points = 2)" begin
    n_comp = 2
    delta_x = 0.0001
    extra_points = 2
    initial_theta = [0.01 for i in 1:n_comp]
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    y_pred = g.(x_test)
    rmse = sqrt(sum(((y_pred - y_true) .^ 2) / n_test))
    @test rmse < 26.0  # 23.64
    global rmse_two_extra_points = rmse
end

## increasing extra points increases accuracy
@testset "Test 6: Welded Beam Function Test (dimensions = 3; n_comp = 2; extra_points = 4)" begin
    n_comp = 2
    delta_x = 0.0001
    extra_points = 4
    initial_theta = [0.01 for i in 1:n_comp]
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    y_pred = g.(x_test)
    rmse = sqrt(sum(((y_pred - y_true) .^ 2) / n_test))
    @test rmse < 26.0  # 22.81
    @test rmse < rmse_two_extra_points
end

## sphere function tests
function sphere_function(x)
    return sum(x .^ 2)
end

## 3D
n = 100
lb = [-5.0, -5.0, -5.0]
ub = [5.0, 5.0, 5.0]
x = sample(n, lb, ub, SobolSample())
grads = gradient.(sphere_function, x)
y = sphere_function.(x)
n_test = 100
x_test = sample(n_test, lb, ub, GoldenSample())
y_true = sphere_function.(x_test)

@testset "Test 7: Sphere Function Test (dimensions = 3; n_comp = 2; extra_points = 2)" begin
    n_comp = 2
    delta_x = 0.0001
    extra_points = 2
    initial_theta = [0.01 for i in 1:n_comp]
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    y_pred = g.(x_test)
    rmse = sqrt(sum(((y_pred - y_true) .^ 2) / n_test))
    @test rmse < 0.001  # 0.000167
end

## 2D
n = 50
lb = [-10.0, -10.0]
ub = [10.0, 10.0]
x = sample(n, lb, ub, SobolSample())
grads = gradient.(sphere_function, x)
y = sphere_function.(x)
n_test = 10
x_test = sample(n_test, lb, ub, GoldenSample())
y_true = sphere_function.(x_test)

@testset "Test 8: Sphere Function Test (dimensions = 2; n_comp = 2; extra_points = 2" begin
    n_comp = 2
    delta_x = 0.0001
    extra_points = 2
    initial_theta = [0.01 for i in 1:n_comp]
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    y_pred = g.(x_test)
    rmse = sqrt(sum(((y_pred - y_true) .^ 2) / n_test))
    @test rmse < 0.001  # 3.9e-5
end

@testset "Test 9: Add Point Test (dimensions = 3; n_comp = 2; extra_points = 2)" begin
    #first we create a surrogate model with just 3 input points
    initial_x_vec = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    initial_y = sphere_function.(initial_x_vec)
    initial_grads = gradient.(sphere_function, initial_x_vec)
    lb = [-5.0, -5.0, -5.0]
    ub = [10.0, 10.0, 10.0]
    n_comp = 2
    delta_x = 0.0001
    extra_points = 2
    initial_theta = [0.01 for i in 1:n_comp]
    g = GEKPLS(
        initial_x_vec, initial_y, initial_grads, n_comp, delta_x, lb, ub,
        extra_points,
        initial_theta
    )
    n_test = 100
    x_test = sample(n_test, lb, ub, GoldenSample())
    y_true = sphere_function.(x_test)
    y_pred1 = g.(x_test)
    rmse1 = sqrt(sum(((y_pred1 - y_true) .^ 2) / n_test)) #rmse1 = 31.91

    #then we update the model with more points to see if performance improves
    n = 100
    x = sample(n, lb, ub, SobolSample())
    grads = gradient.(sphere_function, x)
    y = sphere_function.(x)
    for i in 1:size(x, 1)
        update!(g, x[i], y[i], grads[i][1])
    end
    y_pred2 = g.(x_test)
    rmse2 = sqrt(sum(((y_pred2 - y_true) .^ 2) / n_test)) #rmse2 = 0.0015
    @test (rmse2 < rmse1)
end

@testset "Test 10: Check optimization (dimensions = 3; n_comp = 2; extra_points = 2)" begin
    lb = [-5.0, -5.0, -5.0]
    ub = [10.0, 10.0, 10.0]
    n_comp = 2
    delta_x = 0.0001
    extra_points = 2
    initial_theta = [0.01 for i in 1:n_comp]
    n = 100
    x = sample(n, lb, ub, SobolSample())
    grads = gradient.(sphere_function, x)
    y = sphere_function.(x)
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    x_point,
        minima = surrogate_optimize!(
        sphere_function, SRBF(), lb, ub, g,
        RandomSample(); maxiters = 20,
        num_new_samples = 20, needs_gradient = true
    )
    @test isapprox(minima, 0.0, atol = 0.0001)
end

@testset "Test 11: Check gradient (dimensions = 3; n_comp = 2; extra_points = 3)" begin
    lb = [-5.0, -5.0, -5.0]
    ub = [10.0, 10.0, 10.0]
    n_comp = 2
    delta_x = 0.0001
    extra_points = 3
    initial_theta = [0.01 for i in 1:n_comp]
    n = 100
    x = sample(n, lb, ub, SobolSample())
    grads = gradient.(sphere_function, x)
    y = sphere_function.(x)
    g = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
    grad_surr = gradient(g, (1.0, 1.0, 1.0))
    #test at a single point
    grad_true = gradient(sphere_function, (1.0, 1.0, 1.0))
    bool_tup = isapprox.((grad_surr[1] .- grad_true[1]), (0.0, 0.0, 0.0), (atol = 0.001))
    @test (true, true, true) == bool_tup
    #test for a bunch of points
    grads_surr_vec = gradient.(g, x)
    sum_of_rmse = 0.0
    for i in eachindex(grads_surr_vec)
        sum_of_rmse += sqrt((sum((grads_surr_vec[i][1] .- grads[i][1]) .^ 2) / 3.0))
    end
    @test sum_of_rmse < 0.02  # 0.0115
end

@testset "Test 12: Construction and update! contracts" begin
    lb = [-5.0, -5.0, -5.0]
    ub = [5.0, 5.0, 5.0]
    x = sample(30, lb, ub, SobolSample())
    grads = gradient.(sphere_function, x)
    y = sphere_function.(x)

    @testset "training points outside the bounds are rejected" begin
        # This used to print a diagnostic and return `nothing`, so the caller
        # got a `Nothing` where a surrogate was expected.
        out_x = vcat(collect(x), [(9.0, 0.0, 0.0)])
        out_y = sphere_function.(out_x)
        out_grads = gradient.(sphere_function, out_x)
        @test_throws ArgumentError GEKPLS(
            out_x, out_y, out_grads, 2, 1.0e-4, lb, ub, 2, [0.01, 0.01]
        )
    end

    @testset "update! returns nothing and honours its contract" begin
        g = GEKPLS(x, y, grads, 2, 1.0e-4, lb, ub, 2, [0.01, 0.01])
        new_p = (1.0, 2.0, 3.0)
        @test update!(
            g, new_p, sphere_function(new_p),
            gradient(sphere_function, new_p)[1]
        ) === nothing
        @test length(g.x) == 31

        # A duplicate warns and leaves the model alone.
        before = g((0.5, 0.5, 0.5))
        @test_logs (:warn,) update!(
            g, new_p, sphere_function(new_p),
            gradient(sphere_function, new_p)[1]
        )
        @test length(g.x) == 31
        @test g((0.5, 0.5, 0.5)) == before

        # Out of bounds is an error, not a printed diagnostic.
        @test_throws ArgumentError update!(g, (9.0, 0.0, 0.0), 81.0, [18.0, 0.0, 0.0])
    end

    @testset "update! leaves the caller's containers alone" begin
        xc = collect(x)
        yc = collect(y)
        g = GEKPLS(xc, yc, grads, 2, 1.0e-4, lb, ub, 2, [0.01, 0.01])
        new_p = (1.0, 2.0, 3.0)
        update!(g, new_p, sphere_function(new_p), gradient(sphere_function, new_p)[1])
        @test length(xc) == 30
        @test length(yc) == 30
        @test length(g.x) == 31
    end

    @testset "the keyword front-end matches the positional form" begin
        # `(x, y, lb, ub; kwargs...)` is the shape every other surrogate uses.
        a = GEKPLS(x, y, grads, 2, 1.0e-4, lb, ub, 2, [0.01, 0.01])
        b = GEKPLS(
            x, y, grads, lb, ub; n_comp = 2, delta_x = 1.0e-4,
            extra_points = 2, theta = [0.01, 0.01]
        )
        @test a((1.0, 1.0, 1.0)) ≈ b((1.0, 1.0, 1.0))
    end

    @testset "an oversized nugget costs accuracy" begin
        # The jitter used to be fixed at 1e6 * eps(); it is now the smallest
        # power of ten that lets the Cholesky factorization succeed.
        x_test = sample(50, lb, ub, GoldenSample())
        y_true = sphere_function.(x_test)
        err(nug) = sqrt(
            sum(
                (
                    GEKPLS(
                        x, y, grads, 2, 1.0e-4, lb, ub, 2, [0.01, 0.01];
                        nugget = nug
                    ).(x_test) - y_true
                ) .^ 2
            ) / 50
        )
        @test err(10.0 * eps()) < err(1.0e6 * eps())
    end
end

@testset "Test 13: theta can be fitted by maximizing the reduced likelihood" begin
    lb = [0.125, 5.0, 5.0]
    ub = [1.0, 10.0, 10.0]
    x = sample(60, lb, ub, SobolSample())
    y = welded_beam.(x)
    grads = gradient.(welded_beam, x)
    x_test = sample(60, lb, ub, GoldenSample())
    y_true = welded_beam.(x_test)
    err(g) = sqrt(sum((g.(x_test) - y_true) .^ 2) / length(x_test))

    given = GEKPLS(x, y, grads, 2, 1.0e-4, lb, ub, 2, [0.01, 0.01])
    fitted = GEKPLS(
        x, y, grads, 2, 1.0e-4, lb, ub, 2, [0.01, 0.01]; optimize_theta = true
    )

    # Fitting is off by default here, unlike Kriging and GEK: every likelihood
    # evaluation factorizes an nt x nt matrix with nt = n(1 + extra_points).
    @test !given.optimize_theta
    @test fitted.optimize_theta
    @test given.theta == [0.01, 0.01]
    @test fitted.theta != [0.01, 0.01]
    # The reduced likelihood is the criterion, so it must improve.
    @test fitted.reduced_likelihood_function_value >
        given.reduced_likelihood_function_value
    @test err(fitted) < err(given)

    @testset "update! refits when fitting is on" begin
        g = GEKPLS(
            x, y, grads, 2, 1.0e-4, lb, ub, 2, [0.01, 0.01]; optimize_theta = true
        )
        before = copy(g.theta)
        new_p = (0.5, 7.0, 7.0)
        update!(g, new_p, welded_beam(new_p), gradient(welded_beam, new_p)[1])
        @test length(g.x) == 61
        @test g.theta != before
    end

    @testset "the keyword front-end passes it through" begin
        g = GEKPLS(
            x, y, grads, lb, ub; n_comp = 2, extra_points = 2,
            theta = [0.01, 0.01], optimize_theta = true, n_start = 2
        )
        @test g.optimize_theta
        @test g.theta != [0.01, 0.01]
    end
end
