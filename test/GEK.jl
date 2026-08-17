using Surrogates
using Test
using LinearAlgebra
using Statistics

@testset "GEK" begin
    @testset "1D" begin
        lb = 0.0
        ub = 5.0
        n = 10
        x = sample(n, lb, ub, SobolSample())
        f = t -> t^2
        der = t -> 2 * t
        y = vcat(f.(x), der.(x))

        @testset "reproduces values and gradients at the samples" begin
            # The defining property: with exact gradients supplied, GEK must
            # interpolate both the values and the derivatives.
            g = GEK(x, y, lb, ub)
            # Not exact to machine precision: the 1-D value+derivative system is
            # ill conditioned enough that the nugget engages, capping cond(R) at
            # 1e8 and slightly relaxing interpolation. The ND case below is
            # exact to ~1e-15 because its system is better conditioned.
            scale = maximum(abs, f.(x))
            @test all(isapprox(g(x[i]), f(x[i]), atol = 1.0e-3 * scale) for i in eachindex(x))
            for i in eachindex(x)
                fd = (g(x[i] + 1.0e-6) - g(x[i] - 1.0e-6)) / 2.0e-6
                @test isapprox(fd, der(x[i]), rtol = 1.0e-4)
            end
        end

        @testset "accurate between the samples" begin
            g = GEK(x, y, lb, ub)
            for t in (0.37, 1.62, 2.71, 4.33)
                @test isapprox(g(t), f(t), rtol = 1.0e-3)
            end
        end

        @testset "std_error_at_point" begin
            g = GEK(x, y, lb, ub)
            @test all(std_error_at_point(g, x[i]) < 1.0e-2 for i in eachindex(x))
            for t in range(lb, ub, length = 40)
                s = std_error_at_point(g, t)
                @test isfinite(s)
                @test s ≥ 0.0
            end
        end

        @testset "hyperparameter validation" begin
            @test_throws ArgumentError GEK(x, y, lb, ub, p = 1.0)
            @test_throws ArgumentError GEK(x, y, lb, ub, theta = -1.0)
            @test_throws ArgumentError GEK([1.0, 1.0, 2.0], zeros(6), lb, ub)
            # A y vector that does not match n * (1 + d) is rejected
            @test_throws ArgumentError GEK(x, f.(x), lb, ub)
        end

        @testset "update! requires a gradient" begin
            g = GEK(copy(x), copy(y), lb, ub)
            @test_throws ArgumentError update!(g, 2.5, f(2.5))
            update!(g, 2.5, f(2.5), der(2.5))
            @test length(g.x) == n + 1
            @test length(g.y) == 2 * (n + 1)
            @test isapprox(g(2.5), f(2.5), rtol = 1.0e-3)
            # A defaulted theta is re-derived from the enlarged sample, so an
            # updated surrogate matches a fresh fit on the same data.
            @test g.theta == GEK(g.x, g.y, lb, ub).theta
            fresh = GEK(g.x, g.y, lb, ub)
            @test isapprox(g(1.9), fresh(1.9), rtol = 1.0e-10)
            @test_logs (:warn, r"already exists") update!(g, 2.5, f(2.5), der(2.5))
        end

        @testset "an explicit theta survives update!" begin
            g = GEK(copy(x), copy(y), lb, ub, theta = 0.05)
            update!(g, 2.5, f(2.5), der(2.5))
            @test g.theta == 0.05
            @test isapprox(g(1.9), GEK(g.x, g.y, lb, ub, theta = 0.05)(1.9), rtol = 1.0e-10)
        end

        @testset "dimension checks" begin
            g = GEK(x, y, lb, ub)
            @test_throws ArgumentError g(Float64[])
            @test_throws ArgumentError g((2.0, 3.0, 4.0))
        end
    end

    @testset "ND" begin
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        n = 10
        d = 2
        x = sample(n, lb, ub, SobolSample())
        f = v -> v[1]^2 + v[2]^2
        grad = v -> [2 * v[1], 2 * v[2]]
        # Gradients in point-major order: [∂1(x1), ∂2(x1), ∂1(x2), ∂2(x2), …]
        y = vcat(f.(x), reduce(vcat, grad.(x)))

        @testset "reproduces values and gradients at the samples" begin
            g = GEK(x, y, lb, ub)
            @test all(isapprox(g(x[i]), f(x[i]), rtol = 1.0e-8) for i in eachindex(x))
            h = 1.0e-6
            for i in eachindex(x)
                p = x[i]
                d1 = (g((p[1] + h, p[2])) - g((p[1] - h, p[2]))) / 2h
                d2 = (g((p[1], p[2] + h)) - g((p[1], p[2] - h))) / 2h
                @test isapprox(d1, 2 * p[1], rtol = 1.0e-4)
                @test isapprox(d2, 2 * p[2], rtol = 1.0e-4)
            end
        end

        @testset "accurate between the samples" begin
            # Exercises the default theta, which scales with the sample spread.
            g = GEK(x, y, lb, ub)
            for p in ((0.7, 1.3), (2.2, 3.8), (4.1, 0.6))
                @test isapprox(g(p), f(p), rtol = 5.0e-2)
            end
        end

        @testset "std_error_at_point" begin
            g = GEK(x, y, lb, ub)
            @test all(std_error_at_point(g, x[i]) < 1.0e-2 for i in eachindex(x))
            for p in sample(20, lb, ub, HaltonSample())
                s = std_error_at_point(g, p)
                @test isfinite(s)
                @test s ≥ 0.0
            end
        end

        @testset "update! requires a gradient" begin
            g = GEK(copy(x), copy(y), lb, ub)
            @test_throws ArgumentError update!(g, (2.0, 2.0), f((2.0, 2.0)))
            update!(g, (2.0, 2.0), f((2.0, 2.0)), grad((2.0, 2.0)))
            @test length(g.x) == n + 1
            @test length(g.y) == (n + 1) * (1 + d)
            @test isapprox(g((2.0, 2.0)), f((2.0, 2.0)), rtol = 1.0e-6)
        end

        @testset "hyperparameter validation" begin
            @test_throws ArgumentError GEK(x, y, lb, ub, p = [1.0, 1.0])
            @test_throws ArgumentError GEK(x, y, lb, ub, theta = [-1.0, 1.0])
        end

        @testset "dimension checks" begin
            g = GEK(x, y, lb, ub)
            @test_throws ArgumentError g(Float64[])
            @test_throws ArgumentError g(2.0)
            @test_throws ArgumentError g((2.0, 3.0, 4.0))
        end
    end

    @testset "default theta scales with the data" begin
        # A fixed theta gives a correlation length unrelated to the sample
        # spacing, so on a wide domain the samples are effectively uncorrelated
        # and the surrogate reverts to its mean between them. The default
        # follows Kriging's rule instead.
        f = v -> v[1]^2 + v[2]^2
        grad = v -> [2 * v[1], 2 * v[2]]

        @testset "1D matches the documented formula" begin
            lb = 0.0
            ub = 5.0
            x = sample(10, lb, ub, SobolSample())
            y = vcat((t -> t^2).(x), (t -> 2t).(x))
            g = GEK(x, y, lb, ub)
            @test g.theta ≈ 0.5 / max(1.0e-6 * abs(ub - lb), std(x))^2
        end

        @testset "ND matches the documented formula" begin
            lb = [0.0, 0.0]
            ub = [5.0, 5.0]
            x = sample(10, lb, ub, SobolSample())
            y = vcat(f.(x), reduce(vcat, grad.(x)))
            g = GEK(x, y, lb, ub)
            @test all(
                g.theta .≈ [
                    0.5 / max(1.0e-6 * norm(ub .- lb), std(x_i[i] for x_i in x))^2
                        for i in 1:2
                ]
            )
        end

        @testset "adapts to a wide domain" begin
            # On [0, 50]^2 a fixed theta = 1 is catastrophic; the scaled default
            # stays accurate. This is the case the default exists for.
            lb = [0.0, 0.0]
            ub = [50.0, 50.0]
            x = sample(20, lb, ub, SobolSample())
            y = vcat(f.(x), reduce(vcat, grad.(x)))
            probes = ((7.0, 13.0), (22.0, 38.0), (41.0, 6.0))

            g = GEK(x, y, lb, ub)
            rel_default = maximum(abs(g(p) - f(p)) / f(p) for p in probes)
            g_fixed = GEK(x, y, lb, ub, theta = [1.0, 1.0])
            rel_fixed = maximum(abs(g_fixed(p) - f(p)) / f(p) for p in probes)

            @test rel_default < 1.0e-2
            @test rel_default < rel_fixed / 100
        end
    end
end
