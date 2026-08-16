using Surrogates
using Test

@testset "InverseDistanceSurrogate" begin
    @testset "hyperparameter validation" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        @test_throws ArgumentError InverseDistanceSurrogate(x, y, 0.0, 5.0, p = 0.0)
        @test_throws ArgumentError InverseDistanceSurrogate(x, y, 0.0, 5.0, p = -1.0)
    end

    @testset "hand-computed weights" begin
        # x = [0, 1], y = [0, 1], p = 2, query at 0.25:
        # w = [1/0.25^2, 1/0.75^2] = [16, 16/9], so the prediction is
        # (16 * 0 + 16/9 * 1) / (16 + 16/9) = 1/10.
        surr = InverseDistanceSurrogate([0.0, 1.0], [0.0, 1.0], 0.0, 1.0, p = 2.0)
        @test surr(0.25) ≈ 0.1
        # Symmetry: equidistant from both nodes gives the plain mean
        @test surr(0.5) ≈ 0.5
    end

    @testset "1D" begin
        obj = x -> sin(x) + sin(x)^2 + sin(x)^3
        lb = 0.0
        ub = 10.0
        x = sample(5, lb, ub, HaltonSample())
        y = obj.(x)
        InverseDistance = InverseDistanceSurrogate(x, y, lb, ub, p = 2.4)
        InverseDistance_kwargs = InverseDistanceSurrogate(x, y, lb, ub)

        # Exact reproduction at every training point
        @test all(InverseDistance(x[i]) ≈ y[i] for i in eachindex(x))

        # Shepard prediction is a convex combination of the responses
        for val in range(lb, ub, length = 27)
            @test minimum(y) ≤ InverseDistance(val) ≤ maximum(y)
        end

        update!(InverseDistance, 5.0, -0.91)
        update!(InverseDistance, [5.1, 5.2], [1.0, 2.0])
        @test InverseDistance(5.0) ≈ -0.91
        @test InverseDistance(5.2) ≈ 2.0

        @test_throws ArgumentError InverseDistance(Float64[])
        @test_throws ArgumentError InverseDistance((2.0, 3.0, 4.0))
    end

    @testset "near-coincident query points" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        surr = InverseDistanceSurrogate(x, y, 0.0, 5.0, p = 2.0)
        # Exactly at a node
        @test surr(1.0) == 4.0
        # One ulp away from a node: the weight is huge but finite
        @test surr(nextfloat(2.0)) ≈ 5.0
        # 1e-8 from the node: the deviation is O(w_other/w_node) ≈ 1e-16
        @test isapprox(surr(1.0 + 1.0e-8), 4.0, atol = 1.0e-12)

        # A query point whose inverse-distance weight overflows to Inf
        # (distance 1e-200 to the node at 0) must return the node response,
        # not NaN from Inf/Inf.
        surr0 = InverseDistanceSurrogate([0.0, 1.0, 2.0], [7.0, 8.0, 9.0], 0.0, 5.0, p = 2.0)
        @test surr0(1.0e-200) == 7.0
        # Two coincident training points: the query hits both, return their mean
        surr2 = InverseDistanceSurrogate(
            [0.0, 1.0, 1.0], [1.0, 2.0, 4.0], 0.0, 5.0, p = 2.0
        )
        @test surr2(1.0) == 3.0
    end

    @testset "ND" begin
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        n = 100
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1] * x[2]^2
        y = f.(x)
        p = 3.0
        InverseDistance = InverseDistanceSurrogate(x, y, lb, ub, p = p)

        # Exact reproduction at every training point
        @test all(InverseDistance(x[i]) ≈ y[i] for i in eachindex(x))

        # Convex-combination bound at scattered query points
        for val in sample(20, lb, ub, HaltonSample())
            @test minimum(y) ≤ InverseDistance(val) ≤ maximum(y)
        end

        update!(InverseDistance, (5.0, 3.4), -0.91)
        update!(InverseDistance, [(5.1, 5.2), (5.3, 6.7)], [1.0, 2.0])
        @test InverseDistance((5.0, 3.4)) ≈ -0.91
        @test InverseDistance((5.3, 6.7)) ≈ 2.0

        @test_throws ArgumentError InverseDistance(Float64[])
        @test_throws ArgumentError InverseDistance(2.0)
        @test_throws ArgumentError InverseDistance((2.0, 3.0, 4.0))
    end

    @testset "multi-output" begin
        f = x -> [x^2, x]
        lb = 1.0
        ub = 10.0
        x = sample(5, lb, ub, SobolSample())
        push!(x, 2.0)
        y = f.(x)
        surrogate = InverseDistanceSurrogate(x, y, lb, ub, p = 1.2)
        surrogate_kwargs = InverseDistanceSurrogate(x, y, lb, ub)
        @test surrogate(2.0) ≈ [4, 2]
        # Componentwise convex-combination bound away from the nodes
        pred = surrogate(3.7)
        for k in 1:2
            @test minimum(getindex.(y, k)) ≤ pred[k] ≤ maximum(getindex.(y, k))
        end

        f = x -> [x[1], x[2]^2]
        lb = [1.0, 2.0]
        ub = [10.0, 8.5]
        x = sample(20, lb, ub, SobolSample())
        push!(x, (1.0, 2.0))
        y = f.(x)
        surrogate = InverseDistanceSurrogate(x, y, lb, ub, p = 1.2)
        surrogate_kwargs = InverseDistanceSurrogate(x, y, lb, ub)
        @test surrogate((1.0, 2.0)) ≈ [1, 4]
        x_new = (2.0, 2.0)
        y_new = f(x_new)
        update!(surrogate, x_new, y_new)
        @test surrogate(x_new) ≈ y_new
        @test all(isfinite, surrogate((0.0, 0.0)))
    end
end
