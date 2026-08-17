using LinearAlgebra
using Surrogates
using Test
using Statistics

@testset "Kriging" begin
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
            # p = 0 makes every off-diagonal correlation exp(-theta), so the
            # correlation matrix is singular for more than two samples.
            @test_throws ArgumentError Kriging(x, y, lb, ub, p = 0.0)
            # p = 2 is the smooth boundary of the valid range and must be allowed
            @test Kriging(x, y, lb, ub, p = 2.0) isa Kriging
        end

        @testset "duplicate samples are rejected" begin
            @test_throws ArgumentError Kriging(
                [1.0, 1.0, 2.0], [3.0, 3.0, 4.0], lb, ub
            )
            # `update!` with a duplicate is a documented no-op: the model
            # already holds that observation.
            k = Kriging([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lb, ub)
            @test_logs (:warn, r"already exists") update!(k, 2.0, 5.0)
            @test length(k.x) == 3
        end

        @testset "hyperparameter initialization" begin
            my_k = Kriging(x, y, lb, ub, p = my_p)
            @test my_k.theta ≈ 0.5 * std(x)^(-my_p)

            kwar_krig = Kriging(x, y, lb, ub)
            p_expected = 2.0
            @test kwar_krig.p == p_expected
            @test kwar_krig.theta == 0.5 / std(x)^p_expected
        end

        @testset "dimension checks" begin
            my_k = Kriging(x, y, lb, ub, p = my_p)
            @test_throws ArgumentError my_k(rand(3))
            @test_throws ArgumentError my_k(Float64[])
        end

        @testset "update! and prediction" begin
            my_k = Kriging(copy(x), copy(y), lb, ub, p = my_p)
            update!(my_k, 4.0, 75.68)
            update!(my_k, [5.0, 6.0], [238.86, 722.84])
            pred = my_k(5.5)
            @test 238.86 ≤ pred ≤ 722.84
            @test my_k(5.0) ≈ 238.86
            @test std_error_at_point(my_k, 5.0) < 1.0e-3 * my_k(5.0)
        end

        @testset "interpolates without update!" begin
            my_k = Kriging([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lb, ub, p = 1.3)
            @test my_k(1.0) == 4.0
            @test std_error_at_point(my_k, 1.0) < 1.0e-6
        end

        @testset "update! with a single point" begin
            my_k = Kriging([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lb, ub, p = 1.3)
            update!(my_k, 4.0, 9.0)
            @test my_k(4.0) ≈ 9.0
            @test std_error_at_point(my_k, 4.0) < 1.0e-6
        end

        @testset "update! with several points" begin
            my_k = Kriging([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], lb, ub, p = 1.3)
            update!(my_k, [4.0, 5.0, 6.0], [9.0, 13.0, 15.0])
            @test my_k(4.0) ≈ 9.0
            @test std_error_at_point(my_k, 4.0) < 1.0e-6
        end

        @testset "update! re-derives a defaulted theta" begin
            # The default theta is a function of the sample spread, so it is
            # re-derived on update; that also makes an updated surrogate agree
            # with a fresh fit on the same data.
            xs = [1.0, 2.0, 3.0]
            ys = [4.0, 5.0, 6.0]
            k = Kriging(copy(xs), copy(ys), lb, ub)
            theta0 = k.theta
            update!(k, [7.0, 9.0], [10.0, 12.0])
            @test k.theta != theta0
            fresh = Kriging(k.x, k.y, lb, ub)
            @test k.theta == fresh.theta
            @test isapprox(k(5.0), fresh(5.0), rtol = 1.0e-10)
        end

        @testset "an explicit theta survives update!" begin
            xs = [1.0, 2.0, 3.0]
            ys = [4.0, 5.0, 6.0]
            k = Kriging(copy(xs), copy(ys), lb, ub, theta = 0.25)
            update!(k, [7.0, 9.0], [10.0, 12.0])
            @test k.theta == 0.25
            fresh = Kriging(k.x, k.y, lb, ub, theta = 0.25)
            @test isapprox(k(5.0), fresh(5.0), rtol = 1.0e-10)
        end

        @testset "std_error_at_point grows away from the samples" begin
            # Zero at a sample, and larger the further a query sits from the
            # nearest sample. This is the defining shape of the kriging variance
            # and nothing else in the suite pins it.
            xs = [0.0, 1.0, 2.0, 3.0, 8.0]
            k = Kriging(xs, sin.(xs), 0.0, 10.0)
            @test all(std_error_at_point(k, t) < 1.0e-6 for t in xs)
            # Midpoint of the wide gap is further from data than the midpoint of
            # a narrow one.
            @test std_error_at_point(k, 5.5) > std_error_at_point(k, 1.5)
            @test std_error_at_point(k, 1.5) > std_error_at_point(k, 1.05)
            # The estimate is a standard error: never negative, always finite.
            for t in range(0.0, 10.0, length = 50)
                s = std_error_at_point(k, t)
                @test isfinite(s)
                @test s ≥ 0.0
            end
        end

        @testset "near-duplicate samples exercise the nugget" begin
            # Two samples 1e-9 apart make the correlation matrix numerically
            # singular; the nugget is what keeps the solve well posed.
            xs = [1.0, 1.0 + 1.0e-9, 2.0, 3.0]
            ys = [4.0, 4.0, 5.0, 6.0]
            k = Kriging(xs, ys, 0.0, 10.0)
            @test all(isfinite, k.b)
            @test isfinite(k(2.5))
            @test isfinite(std_error_at_point(k, 2.5))
            @test std_error_at_point(k, 2.5) ≥ 0.0
        end
    end

    @testset "ND" begin
        lb = [0.0, 0.0, 1.0]
        ub = [5.0, 7.5, 10.0]
        x = sample(5, lb, ub, SobolSample())
        f = v -> v[1] + v[2] * v[3]
        y = f.(x)
        my_theta = [2.0, 2.0, 2.0]
        my_p = [1.9, 1.9, 1.9]

        @testset "update! and prediction" begin
            my_k = Kriging(copy(x), copy(y), lb, ub, p = my_p, theta = my_theta)
            update!(my_k, (4.0, 3.2, 9.5), 34.4)
            update!(my_k, [(1.0, 4.65, 6.4), (2.3, 5.4, 6.7)], [30.76, 38.48])
            @test isfinite(my_k((3.5, 5.5, 6.5)))
            @test my_k((4.0, 3.2, 9.5)) ≈ 34.4
        end

        xs = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        ys = [1.0, 2.0, 3.0]
        flat_p = [1.0, 1.0, 1.0]

        @testset "interpolates without update!" begin
            my_k = Kriging(xs, ys, lb, ub, p = flat_p, theta = my_theta)
            @test my_k((1.0, 2.0, 3.0)) ≈ 1.0
            @test std_error_at_point(my_k, (1.0, 2.0, 3.0)) < 1.0e-6
        end

        @testset "update! with a single point" begin
            my_k = Kriging(copy(xs), copy(ys), lb, ub, p = flat_p, theta = my_theta)
            update!(my_k, (10.0, 11.0, 12.0), 4.0)
            @test my_k((10.0, 11.0, 12.0)) ≈ 4.0
            @test std_error_at_point(my_k, (10.0, 11.0, 12.0)) < 1.0e-6
        end

        @testset "update! with several points" begin
            my_k = Kriging(copy(xs), copy(ys), lb, ub, p = flat_p, theta = my_theta)
            update!(my_k, [(10.0, 11.0, 12.0), (13.0, 14.0, 15.0)], [4.0, 5.0])
            @test my_k((10.0, 11.0, 12.0)) ≈ 4.0
            @test std_error_at_point(my_k, (10.0, 11.0, 12.0)) < 1.0e-6
        end

        @testset "hyperparameter validation" begin
            @test_throws ArgumentError Kriging(xs, ys, lb, ub, p = 3 * flat_p)
            @test_throws ArgumentError Kriging(xs, ys, lb, ub, p = -flat_p)
            @test_throws ArgumentError Kriging(xs, ys, lb, ub, theta = -my_theta)
            @test_throws ArgumentError Kriging(xs, ys, lb, ub, p = 0 * flat_p)
        end

        @testset "duplicate samples are rejected" begin
            dup = [(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
            @test_throws ArgumentError Kriging(
                dup, [1.0, 1.0, 2.0], lb, ub, p = flat_p, theta = my_theta
            )
            k = Kriging(copy(xs), copy(ys), lb, ub, p = flat_p, theta = my_theta)
            @test_logs (:warn, r"already exists") update!(k, (4.0, 5.0, 6.0), 2.0)
            @test length(k.x) == 3
        end

        @testset "dimension checks" begin
            kwarg_krig_ND = Kriging(xs, ys, lb, ub)
            @test_throws ArgumentError kwarg_krig_ND(1.0)
            @test_throws ArgumentError kwarg_krig_ND([1.0])
            @test_throws ArgumentError kwarg_krig_ND([2.0, 3.0])
            @test_throws ArgumentError kwarg_krig_ND(ones(5))
        end

        @testset "hyperparameter initialization" begin
            kwarg_krig_ND = Kriging(xs, ys, lb, ub)
            p_expected = 2.0
            @test all(==(p_expected), kwarg_krig_ND.p)
            @test all(
                kwarg_krig_ND.theta .≈
                    [0.5 / std(x_i[ℓ] for x_i in xs)^p_expected for ℓ in 1:3]
            )
        end

        @testset "std_error_at_point grows away from the samples" begin
            lb2 = [0.0, 0.0]
            ub2 = [5.0, 5.0]
            pts = sample(12, lb2, ub2, SobolSample())
            k = Kriging(pts, (v -> v[1] + v[2]).(pts), lb2, ub2)
            @test all(std_error_at_point(k, p) < 1.0e-6 for p in pts)
            for p in sample(20, lb2, ub2, HaltonSample())
                s = std_error_at_point(k, p)
                @test isfinite(s)
                @test s ≥ 0.0
            end
        end
    end

    @testset "the correlation matrix is solved, not inverted" begin
        # R_fact holds a Cholesky factorization of the regularized correlation
        # matrix. `inverse_of_R` remains as a deprecated property that
        # materializes the inverse from it.
        x = [1.0, 2.0, 3.0, 4.0]
        k = Kriging(x, [4.0, 5.0, 6.0, 9.0], 0.0, 5.0)
        @test k.R_fact isa Cholesky
        R = Matrix(k.R_fact)
        @test R * (k.R_fact \ k.y) ≈ k.y
        inv_R = @test_deprecated k.inverse_of_R
        @test inv_R * R ≈ I
    end
end
