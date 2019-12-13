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
