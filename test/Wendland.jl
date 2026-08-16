using Surrogates
using Test

@testset "Wendland" begin
    @testset "1D interpolation" begin
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        y = sin.(x)
        # eps = 0.5 gives support radius 2, so neighboring kernels overlap and
        # the interpolation system is nontrivial.
        my_wend = Wendland(x, y, 0.0, 2.0, eps = 0.5)
        @test all(isapprox(my_wend(x[i]), y[i], atol = 1.0e-4) for i in eachindex(x))

        update!(my_wend, 2.5, sin(2.5))
        @test isapprox(my_wend(2.5), sin(2.5), atol = 1.0e-4)

        @test_throws ArgumentError my_wend(Float64[])
        @test_throws ArgumentError my_wend((2.0, 3.0, 4.0))
    end

    @testset "compact support" begin
        # A single training point has zero influence beyond distance 1/eps.
        w = Wendland([0.0], [3.0], -5.0, 5.0, eps = 0.5)
        @test w(2.5) == 0.0
        @test w(0.0) ≈ 3.0
    end

    @testset "ND interpolation" begin
        lb = [0.0, 0.0]
        ub = [4.0, 4.0]
        x = sample(12, lb, ub, SobolSample())
        f = x -> x[1] + x[2]
        y = f.(x)
        my_wend_ND = Wendland(x, y, lb, ub, eps = 0.3)
        @test all(isapprox(my_wend_ND(x[i]), y[i], atol = 1.0e-4) for i in eachindex(x))

        update!(my_wend_ND, (3.0, 3.5), f((3.0, 3.5)))
        update!(my_wend_ND, [(0.5, 1.0), (1.5, 2.5)], [f((0.5, 1.0)), f((1.5, 2.5))])
        @test isapprox(my_wend_ND((3.0, 3.5)), f((3.0, 3.5)), atol = 1.0e-4)

        @test_throws ArgumentError my_wend_ND(Float64[])
        @test_throws ArgumentError my_wend_ND(2.0)
        @test_throws ArgumentError my_wend_ND((2.0, 3.0, 4.0))
    end

    @testset "update! accepts every argument shape" begin
        # A single 1D sample passed as a one-element vector used to throw a
        # MethodError, because the length-based branch classified it as a point
        # rather than as a collection and pushed the vector itself.
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        y = sin.(x)
        mk() = Wendland(x, y, 0.0, 3.0, eps = 0.5)

        w = mk()
        update!(w, 2.5, sin(2.5))
        @test length(w.x) == 6
        @test isapprox(w(2.5), sin(2.5), atol = 1.0e-4)

        w = mk()
        update!(w, [2.5], [sin(2.5)])
        @test length(w.x) == 6
        @test isapprox(w(2.5), sin(2.5), atol = 1.0e-4)

        w = mk()
        update!(w, [2.5, 3.0], sin.([2.5, 3.0]))
        @test length(w.x) == 7
        @test isapprox(w(3.0), sin(3.0), atol = 1.0e-4)

        lb = [0.0, 0.0]
        ub = [4.0, 4.0]
        xn = sample(12, lb, ub, SobolSample())
        f = v -> v[1] + v[2]
        yn = f.(xn)
        mkn() = Wendland(xn, yn, lb, ub, eps = 0.3)

        w = mkn()
        update!(w, (3.0, 3.5), f((3.0, 3.5)))
        @test length(w.x) == 13
        @test isapprox(w((3.0, 3.5)), f((3.0, 3.5)), atol = 1.0e-4)

        w = mkn()
        update!(w, [(3.0, 3.5)], [f((3.0, 3.5))])
        @test length(w.x) == 13
        @test isapprox(w((3.0, 3.5)), f((3.0, 3.5)), atol = 1.0e-4)
    end

    @testset "non-converged solve warns" begin
        x = sample(40, 0.0, 10.0, SobolSample())
        y = sin.(x)
        @test_logs (:warn, r"did not converge") match_mode = :any Wendland(
            x, y, 0.0, 10.0, eps = 0.01, maxiters = 1
        )
    end
end
