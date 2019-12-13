using Surrogates
using Stheno
using Zygote
using Test

@testset "1D - 1D" begin

    lb = 0.0
    ub = 10.0
    f = x -> log(x)*exp(x)
    x = sample(5,lb,ub,SobolSample())
    y = f.(x)
    my_k = SthenoKriging(x,y)
    x_new = 4.0
    y_new = f.(x_new)
    add_point!(my_k, x_new, y_new)
    xs_new = [5.0, 6.0]
    ys_new = f.(xs_new)
    add_point!(my_k, xs_new, ys_new)
    y_pred = f(x_new)
    pred = my_k(x_new)
    @test pred ≈ y_pred
    pred_std = std_error_at_point(my_k, x_new)
    @test pred_std ≈ zero(y_pred) atol=1e-6

    @testset "AD" begin
        Zygote.gradient(x -> my_k(x), 1.0)
    end

    @testset "Optimization" begin
        objective_function = x -> 2*x+1
        x = [2.0,4.0,6.0]
        y = [5.0,9.0,13.0]
        p = 2
        a = 2
        b = 6
        my_k_EI1 = SthenoKriging(x,y)
        surrogate_optimize(objective_function,EI(),a,b,my_k_EI1,UniformSample(),maxiters=200,num_new_samples=155)
    end
end

@testset "ND - 1D" begin

    lb = [0.0; 0.0]
    ub = [5.0; 10.0]
    f = x -> log(x[1])*exp(x[2])
    x = sample(5,lb,ub,SobolSample())
    y = f.(x)
    my_k = SthenoKriging(x,y)
    x_new = (2.0, 1.0)
    y_new = f(x_new)
    add_point!(my_k, x_new, y_new)
    xs_new = [(1.0, 2.0), (0.5, 2.5)]
    ys_new = f.(xs_new)
    add_point!(my_k, xs_new, ys_new)
    y_pred = f(x_new)
    pred = my_k(x_new)
    @test pred ≈ y_pred
    pred_std = std_error_at_point(my_k, x_new)
    @test pred_std ≈ zero(y_pred) atol=1e-6

    @testset "AD" begin
        g1 = Zygote.gradient(x -> my_k(x), (1.0, 1.0))[1]
        g2 = Zygote.gradient(x -> my_k(x), [1.0, 1.0])[1]
        @test [g1...] ≈ g2
    end
    @testset "Optimization" begin
        objective_function_ND = z -> 3*hypot(z...)+1
        x = [(1.2,3.0),(3.0,3.5),(5.2,5.7)]
        y = objective_function_ND.(x)
        theta = [2.0,2.0]
        lb = [1.0,1.0]
        ub = [6.0,6.0]

        my_k_E1N = SthenoKriging(x,y)
        surrogate_optimize(objective_function_ND,EI(),lb,ub,my_k_E1N,UniformSample())
    end
end

@testset "1D - ND" begin

    lb = 0.0
    ub = 10.0
    f = x -> [x, x^2]
    x = sample(5,lb,ub,SobolSample())
    y = f.(x)

    my_k = SthenoKriging(x,y,Stheno.GP(Stheno.EQ(), Stheno.GPC()))
    x_new = 4.0
    y_new = f.(x_new)
    add_point!(my_k, x_new, y_new)
    xs_new = [5.0, 6.0]
    ys_new = f.(xs_new)
    add_point!(my_k, xs_new, ys_new)
    y_pred = f(x_new)
    pred = my_k(x_new)
    @test pred ≈ y_pred
    pred_std = std_error_at_point(my_k, x_new)
    @test pred_std ≈ zero(y_pred) atol=1e-6

    # separate Stheno models
    gpc = Stheno.GPC()
    gp1 = Stheno.GP(Stheno.EQ(), gpc)
    gp2 = Stheno.GP(Stheno.Exp(), gpc)
    my_k = SthenoKriging(x, y, (gp1, gp2))
    x_new = 0.2
    y_new = f.(x_new)
    add_point!(my_k, x_new, y_new)
    xs_new = [5.5, 6.5]
    ys_new = f.(xs_new)
    add_point!(my_k, xs_new, ys_new)
    y_pred = f(x_new)
    pred = my_k(x_new)
    @test pred ≈ y_pred
    pred_std = std_error_at_point(my_k, x_new)
    @test pred_std ≈ zero(y_pred) atol=1e-6

    @testset "AD" begin
        Zygote.gradient(x -> sum(my_k(x)), 1.0)
    end
end

@testset "ND - ND" begin

    lb = [0.0; 0.0]
    ub = [5.0; 10.0]
    f = x -> [x[1]*x[2], x[2]^2]
    x = sample(5,lb,ub,SobolSample())
    y = f.(x)

    my_k = SthenoKriging(x,y,Stheno.GP(Stheno.EQ(), Stheno.GPC()))
    x_new = (1.0, 1.0)
    y_new = f(x_new)
    add_point!(my_k, x_new, y_new)
    xs_new = [(1.0, 2.0), (0.5, 2.5)]
    ys_new = f.(xs_new)
    add_point!(my_k, xs_new, ys_new)
    y_pred = f(x_new)
    pred = my_k(x_new)
    @test pred ≈ y_pred
    pred_std = std_error_at_point(my_k, x_new)
    @test pred_std ≈ zero(y_pred) atol=1e-6

    # separate Stheno models
    gpc = Stheno.GPC()
    gp1 = Stheno.GP(Stheno.EQ(), gpc)
    gp2 = Stheno.GP(Stheno.Exp(), gpc)
    my_k = SthenoKriging(x, y, (gp1, gp2))
    x_new = (0.2, 0.1)
    y_new = f(x_new)
    add_point!(my_k, x_new, y_new)
    xs_new = [(5.5, 0.5), (6.5, 0.5)]
    ys_new = f.(xs_new)
    add_point!(my_k, xs_new, ys_new)
    y_pred = f(x_new)
    pred = my_k(x_new)
    @test pred ≈ y_pred
    pred_std = std_error_at_point(my_k, x_new)
    @test pred_std ≈ zero(y_pred) atol=1e-6

    @testset "AD" begin
        g1 = Zygote.gradient(x -> sum(my_k(x)), (1.1, 1.1))[1]
        g2 = Zygote.gradient(x -> sum(my_k(x)), [1.1, 1.1])[1]
        @test [g1...] ≈ g2
    end
end
