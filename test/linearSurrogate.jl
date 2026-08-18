using Surrogates
using Test

@testset "LinearSurrogate" begin
    @testset "1D affine recovery" begin
        # The surrogate must recover an affine function exactly, including its
        # intercept, and stay exact after consistent updates.
        f = x -> 2 * x + 5
        x = [0.0, 1.0, 2.0, 3.0]
        y = f.(x)
        lin = LinearSurrogate(x, y, 0.0, 7.0)
        @test lin.coeff ≈ [5.0, 2.0]
        @test lin(1.5) ≈ f(1.5)
        update!(lin, 4.0, f(4.0))
        update!(lin, [5.0, 6.0], f.([5.0, 6.0]))
        @test lin.coeff ≈ [5.0, 2.0]
        @test lin(5.5) ≈ f(5.5)
    end

    @testset "1D least-squares optimality" begin
        # For non-affine data, the fit must satisfy the normal equations:
        # residuals are orthogonal to the intercept column and to x.
        x = [1.0, 2.0, 3.0]
        y = [1.5, 3.5, 5.3]
        lb = 0.0
        ub = 7.0
        my_linear_surr_1D = LinearSurrogate(x, y, lb, ub)
        r = y .- my_linear_surr_1D.(x)
        @test isapprox(sum(r), 0.0; atol = 1.0e-10)
        @test isapprox(sum(r .* x), 0.0; atol = 1.0e-10)

        update!(my_linear_surr_1D, 4.0, 7.2)
        update!(my_linear_surr_1D, [5.0, 6.0], [8.3, 9.7])
        @test length(my_linear_surr_1D.x) == 6
        @test length(my_linear_surr_1D.y) == 6
        # Optimality must be restored after the refit
        r = my_linear_surr_1D.y .- my_linear_surr_1D.(my_linear_surr_1D.x)
        @test isapprox(sum(r), 0.0; atol = 1.0e-10)
        @test isapprox(sum(r .* my_linear_surr_1D.x), 0.0; atol = 1.0e-10)

        @test_throws ArgumentError my_linear_surr_1D(Float64[])
        @test_throws ArgumentError my_linear_surr_1D((2.0, 3.0, 4.0))
    end

    @testset "update! matches a fresh fit" begin
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.5, 3.5, 5.3, 6.9]
        lin = LinearSurrogate(x[1:2], y[1:2], 0.0, 7.0)
        update!(lin, x[3:4], y[3:4])
        fresh = LinearSurrogate(x, y, 0.0, 7.0)
        @test lin.coeff ≈ fresh.coeff
        @test lin(2.5) ≈ fresh(2.5)
    end

    @testset "ND affine recovery" begin
        f = x -> 3.0 + 2.0 * x[1] - x[2]
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        x = sample(6, lb, ub, SobolSample())
        y = f.(x)
        lin = LinearSurrogate(x, y, lb, ub)
        @test lin.coeff ≈ [3.0, 2.0, -1.0]
        @test lin((4.0, 8.0)) ≈ f((4.0, 8.0))
        update!(lin, (1.5, 2.5), f((1.5, 2.5)))
        update!(lin, [(8.0, 5.0), (9.0, 9.5)], f.([(8.0, 5.0), (9.0, 9.5)]))
        @test lin.coeff ≈ [3.0, 2.0, -1.0]
        @test lin((4.0, 8.0)) ≈ f((4.0, 8.0))
    end

    @testset "ND least-squares optimality" begin
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        x = sample(5, lb, ub, SobolSample())
        y = [4.0, 5.0, 6.0, 7.0, 8.0]
        my_linear_ND = LinearSurrogate(x, y, lb, ub)
        update!(my_linear_ND, (10.0, 11.0), 9.0)
        update!(my_linear_ND, [(8.0, 5.0), (9.0, 9.5)], [4.0, 5.0])
        # Residuals orthogonal to the intercept column and to every coordinate
        r = my_linear_ND.y .- my_linear_ND.(my_linear_ND.x)
        @test isapprox(sum(r), 0.0; atol = 1.0e-8)
        for j in 1:2
            @test isapprox(
                sum(r .* getindex.(my_linear_ND.x, j)), 0.0; atol = 1.0e-8
            )
        end

        @test_throws ArgumentError my_linear_ND(Float64[])
        @test_throws ArgumentError my_linear_ND(1.0)
        @test_throws ArgumentError my_linear_ND((2.0, 3.0, 4.0))
    end
end
