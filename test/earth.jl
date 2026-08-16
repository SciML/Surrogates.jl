using Surrogates
using Test

@testset "EarthSurrogate" begin
    @testset "intercept is fitted, not double counted" begin
        # A large mean offset makes an intercept bug unmissable: adding the
        # response mean on top of coefficients fit to uncentered data roughly
        # doubles the prediction.
        lb = 0.0
        ub = 5.0
        f = x -> 100.0 + x
        x = sample(20, lb, ub, SobolSample())
        y = f.(x)
        e = EarthSurrogate(x, y, lb, ub)
        @test isapprox(e(2.5), f(2.5), rtol = 1.0e-2)

        # The intercept is a fitted least-squares parameter, so the residuals
        # must be orthogonal to the constant column (they would not be if the
        # intercept were pinned to mean(y)).
        r = y .- e.(x)
        @test isapprox(sum(r), 0.0; atol = 1.0e-8 * maximum(abs, y))
    end

    @testset "1D" begin
        lb = 0.0
        ub = 5.0
        n = 20
        x = sample(n, lb, ub, SobolSample())
        f = x -> 2 * x + x^2
        y = f.(x)
        my_ear1d = EarthSurrogate(x, y, lb, ub)

        # The hinge basis must actually fit the data, not just return the
        # intercept: check in-sample RMSE against the response spread.
        rmse = sqrt(sum(abs2, my_ear1d.(x) .- y) / n)
        @test rmse < 0.05 * (maximum(y) - minimum(y))
        @test isapprox(my_ear1d(3.0), f(3.0), rtol = 0.05)

        update!(my_ear1d, 6.0, f(6.0))
        rmse = sqrt(sum(abs2, my_ear1d.(my_ear1d.x) .- my_ear1d.y) / length(my_ear1d.y))
        @test rmse < 0.05 * (maximum(my_ear1d.y) - minimum(my_ear1d.y))

        @test_throws ArgumentError my_ear1d(Float64[])
        @test_throws ArgumentError my_ear1d((2.0, 3.0, 4.0))
    end

    @testset "ND" begin
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        n = 30
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1] * x[2] + x[1]
        y = f.(x)
        my_earnd = EarthSurrogate(x, y, lb, ub)

        rmse = sqrt(sum(abs2, my_earnd.(x) .- y) / n)
        @test rmse < 0.05 * (maximum(y) - minimum(y))
        @test isfinite(my_earnd((2.0, 2.0)))

        update!(my_earnd, (2.0, 2.0), f((2.0, 2.0)))
        @test isfinite(my_earnd((2.0, 2.0)))

        @test_throws ArgumentError my_earnd(Float64[])
        @test_throws ArgumentError my_earnd(2.0)
        @test_throws ArgumentError my_earnd((2.0, 3.0, 4.0))
    end
end
