using Surrogates
using Test

@testset "InverseDistanceSurrogate" begin
    @testset "hyperparameter validation" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        @test_throws ArgumentError InverseDistanceSurrogate(x, y, 0.0, 5.0, p = 0.0)
        @test_throws ArgumentError InverseDistanceSurrogate(x, y, 0.0, 5.0, p = -1.0)
        @test InverseDistanceSurrogate(x, y, 0.0, 5.0).p == 1.0
        @test InverseDistanceSurrogate(x, y, 0.0, 5.0, p = 2).p == 2
    end

    @testset "1D input" begin
        obj = x -> sin(x) + sin(x)^2 + sin(x)^3
        lb = 0.0
        ub = 10.0
        x = sample(15, lb, ub, HaltonSample())
        y = obj.(x)

        @testset "hand-computed weights" begin
            # w = [1/0.25^2, 1/0.75^2] = [16, 16/9], so the value at 0.25 is
            # (16*0 + (16/9)*1) / (16 + 16/9) = 1/10.
            surr = InverseDistanceSurrogate([0.0, 1.0], [0.0, 1.0], 0.0, 1.0, p = 2.0)
            @test surr(0.25) ≈ 0.1
            @test surr(0.75) ≈ 0.9
            # Equidistant from both nodes gives the plain mean, for any p.
            @test surr(0.5) ≈ 0.5
            @test InverseDistanceSurrogate([0.0, 1.0], [0.0, 1.0], 0.0, 1.0, p = 7.3)(0.5) ≈ 0.5
        end

        @testset "interpolation" begin
            for p in (0.5, 1.0, 2.4, 5.0)
                surr = InverseDistanceSurrogate(x, y, lb, ub, p = p)
                @test all(surr(x[i]) ≈ y[i] for i in eachindex(x))
            end
        end

        @testset "convex combination of the responses" begin
            surr = InverseDistanceSurrogate(x, y, lb, ub, p = 2.4)
            for val in range(lb, ub, length = 101)
                @test minimum(y) ≤ surr(val) ≤ maximum(y)
            end
        end

        @testset "monotone in p towards nearest neighbour" begin
            # A larger p lets the nearest node dominate.
            surr_x = [0.0, 1.0, 3.0]
            surr_y = [10.0, 20.0, 30.0]
            val = 0.9
            preds = [InverseDistanceSurrogate(surr_x, surr_y, 0.0, 3.0, p = p)(val)
                     for p in (1.0, 4.0, 16.0, 64.0)]
            @test issorted(abs.(preds .- 20.0), rev = true)
            @test preds[end] ≈ 20.0 atol = 1.0e-6
        end

        @testset "call forms" begin
            surr = InverseDistanceSurrogate(x, y, lb, ub, p = 2.4)
            @test surr(5.0) isa Number
            @test surr([5.0]) ≈ surr(5.0)
            @test surr((5.0,)) ≈ surr(5.0)
            @test_throws ArgumentError surr(Float64[])
            @test_throws ArgumentError surr((2.0, 3.0, 4.0))
        end

        @testset "update!" begin
            surr = InverseDistanceSurrogate(copy(x), copy(y), lb, ub, p = 2.4)
            n = length(surr.x)
            update!(surr, 5.0, -0.91)
            @test length(surr.x) == n + 1
            @test surr(5.0) ≈ -0.91
            update!(surr, [5.1, 5.2], [1.0, 2.0])
            @test length(surr.x) == n + 3
            @test surr(5.1) ≈ 1.0
            @test surr(5.2) ≈ 2.0
        end

        @testset "update! does not mutate the caller's arrays" begin
            xs = [1.0, 2.0]
            ys = [1.0, 2.0]
            surr = InverseDistanceSurrogate(xs, ys, 0.0, 5.0, p = 2.0)
            update!(surr, 3.0, 3.0)
            @test length(xs) == 2
            @test length(ys) == 2
            @test length(surr.x) == 3
        end

        @testset "element types are preserved" begin
            s32 = InverseDistanceSurrogate(Float32.(x), Float32.(y), 0.0f0, 10.0f0, p = 2.0f0)
            @test s32(5.0f0) isa Float32
            xb, yb = BigFloat.(x), BigFloat.(y)
            sb = InverseDistanceSurrogate(xb, yb, BigFloat(0), BigFloat(10), p = 2.0)
            @test sb(BigFloat(5)) isa BigFloat
            @test sb(xb[3]) == yb[3]
        end

        @testset "agreement with a naive BigFloat reference" begin
            # The rescaled weights must give what the plain Shepard formula
            # gives in extended precision.
            function shepard_ref(x, y, p, val)
                d = [abs(BigFloat(val) - BigFloat(xi)) for xi in x]
                w = d .^ (-BigFloat(p))
                return Float64(sum(w .* y) / sum(w))
            end
            grid = range(0.05, 9.95, length = 50)
            for p in (0.5, 1.0, 2.4, 3.7)
                surr = InverseDistanceSurrogate(x, y, lb, ub, p = p)
                err = maximum(abs(surr(v) - shepard_ref(x, y, p, v)) for v in grid)
                @test err < 1.0e-13
            end
        end
    end

    @testset "degenerate query points" begin
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        surr = InverseDistanceSurrogate(x, y, 0.0, 5.0, p = 2.0)
        @test surr(1.0) == 4.0
        @test surr(nextfloat(2.0)) ≈ 5.0
        @test isapprox(surr(1.0 + 1.0e-8), 4.0, atol = 1.0e-12)

        @testset "weight overflow" begin
            # d^(-p) overflows to Inf here, so a direct ratio gives NaN.
            surr0 = InverseDistanceSurrogate(
                [0.0, 1.0, 2.0], [7.0, 8.0, 9.0], 0.0, 5.0, p = 2.0
            )
            @test surr0(1.0e-200) == 7.0
            @test surr0(1.0e-160) == 7.0
        end

        @testset "weight underflow" begin
            # Every d^(-p) underflows to zero here, giving 0 / 0.
            surr = InverseDistanceSurrogate(
                [0.0, 100.0, 200.0], [1.0, 2.0, 3.0], 0.0, 1000.0, p = 400.0
            )
            @test isfinite(surr(50.0))
            @test surr(50.0) ≈ 1.5
        end

        @testset "coincident sample points" begin
            # Two nodes at one location: the limit from either side is the
            # mean of their responses.
            surr = InverseDistanceSurrogate(
                [0.0, 1.0, 1.0], [1.0, 2.0, 4.0], 0.0, 5.0, p = 2.0
            )
            @test surr(1.0) == 3.0
            @test isapprox(surr(1.0 + 1.0e-9), 3.0, atol = 1.0e-8)
            @test isapprox(surr(1.0 - 1.0e-9), 3.0, atol = 1.0e-8)
        end

        @testset "single sample point" begin
            surr = InverseDistanceSurrogate([2.0], [7.0], 0.0, 5.0, p = 2.0)
            @test surr(2.0) == 7.0
            @test surr(4.0) == 7.0
        end
    end

    @testset "ND input" begin
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        x = sample(60, lb, ub, SobolSample())
        f = x -> x[1] * x[2]^2
        y = f.(x)
        surr = InverseDistanceSurrogate(x, y, lb, ub, p = 3.0)

        @testset "interpolation" begin
            @test all(surr(x[i]) ≈ y[i] for i in eachindex(x))
        end

        @testset "convex combination of the responses" begin
            for val in sample(40, lb, ub, HaltonSample())
                @test minimum(y) ≤ surr(val) ≤ maximum(y)
            end
        end

        @testset "call forms agree" begin
            val = (3.0, 7.0)
            @test surr(val) ≈ surr([3.0, 7.0])
            @test surr(val) ≈ surr([3.0 7.0])
            @test_throws ArgumentError surr(Float64[])
            @test_throws ArgumentError surr(2.0)
            @test_throws ArgumentError surr((2.0, 3.0, 4.0))
        end

        @testset "a node reached through a different container" begin
            # A node stored as a Tuple is the same point queried as a Vector.
            nodes = [(1.0, 2.0), (3.0, 4.0), (8.0, 1.5)]
            vals = [1.0, 2.0, 3.0]
            s = InverseDistanceSurrogate(nodes, vals, [0.0, 0.0], [10.0, 10.0], p = 2.0)
            @test s((3.0, 4.0)) == 2.0
            @test s([3.0, 4.0]) == 2.0
            @test s([3.0 4.0]) == 2.0
            # Away from the nodes, where nothing cancels by symmetry
            @test s((2.0, 3.0)) ≈ s([2.0, 3.0]) ≈ s([2.0 3.0])
        end

        @testset "update!" begin
            s = InverseDistanceSurrogate(copy(x), copy(y), lb, ub, p = 3.0)
            n = length(s.x)
            update!(s, (5.0, 3.4), -0.91)
            @test length(s.x) == n + 1
            @test s((5.0, 3.4)) ≈ -0.91
            update!(s, [(5.1, 5.2), (5.3, 6.7)], [1.0, 2.0])
            @test length(s.x) == n + 3
            @test s((5.3, 6.7)) ≈ 2.0
        end

        @testset "rotation invariance" begin
            # The inputs enter only through Euclidean distances.
            θ = 0.7
            R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
            rot(p) = Tuple(R * collect(p))
            s_rot = InverseDistanceSurrogate(rot.(x), y, lb, ub, p = 3.0)
            val = (3.0, 7.0)
            @test s_rot(rot(val)) ≈ surr(val)
        end
    end

    @testset "input types" begin
        xt = [(1.0, 2.0), (3.0, 4.0), (5.0, 1.0)]
        xv = [collect(p) for p in xt]
        yn = [1.0, 2.0, 3.0]
        lb = [0.0, 0.0]
        ub = [6.0, 6.0]
        st = InverseDistanceSurrogate(xt, yn, lb, ub, p = 2.0)

        @testset "integer samples and responses" begin
            s = InverseDistanceSurrogate([1, 2, 3], [1, 4, 9], 0, 4, p = 2.0)
            @test s(2) == 4.0
            @test s(1.5) isa Number
            @test s(1.5) ≈ InverseDistanceSurrogate(
                [1.0, 2.0, 3.0], [1.0, 4.0, 9.0], 0.0, 4.0, p = 2.0)(1.5)
        end

        @testset "ND samples as vectors rather than tuples" begin
            sv = InverseDistanceSurrogate(xv, yn, lb, ub, p = 2.0)
            @test sv([2.0, 3.0]) ≈ st((2.0, 3.0))
            @test sv(xv[2]) == yn[2]
        end

        @testset "bounds as tuples" begin
            s = InverseDistanceSurrogate(xt, yn, (0.0, 0.0), (6.0, 6.0), p = 2.0)
            @test s((2.0, 3.0)) ≈ st((2.0, 3.0))
        end

        @testset "query as a column matrix" begin
            @test st(reshape([2.0, 3.0], 2, 1)) ≈ st((2.0, 3.0))
        end

        @testset "three-dimensional input" begin
            x3 = [(1.0, 2.0, 3.0), (2.0, 1.0, 0.5), (0.0, 0.0, 1.0)]
            s = InverseDistanceSurrogate(x3, yn, [0.0, 0, 0], [6.0, 6, 6], p = 2.0)
            @test s(x3[2]) == yn[2]
            @test minimum(yn) ≤ s((1.0, 1.0, 1.0)) ≤ maximum(yn)
        end

        @testset "update! with a single sample wrapped in a vector" begin
            # The shape a batch of one takes; it must append one point rather
            # than a nested container.
            s1 = InverseDistanceSurrogate([1.0, 2.0, 3.0], [1.0, 4.0, 9.0], 0.0, 5.0, p = 2.0)
            update!(s1, [4.0], [16.0])
            @test length(s1.x) == 4
            @test s1(4.0) == 16.0

            s2 = InverseDistanceSurrogate(xt, yn, lb, ub, p = 2.0)
            update!(s2, [(2.0, 2.0)], [7.0])
            @test length(s2.x) == 4
            @test s2((2.0, 2.0)) == 7.0
        end
    end

    @testset "ND output" begin
        @testset "1D input" begin
            f = x -> [x^2, x]
            lb = 1.0
            ub = 10.0
            x = sample(5, lb, ub, SobolSample())
            push!(x, 2.0)
            y = f.(x)
            surr = InverseDistanceSurrogate(x, y, lb, ub, p = 1.2)
            @test surr(2.0) ≈ [4, 2]
            @test all(surr(x[i]) ≈ y[i] for i in eachindex(x))

            # Each component is a convex combination of that component's responses.
            pred = surr(3.7)
            @test length(pred) == 2
            for k in 1:2
                @test minimum(getindex.(y, k)) ≤ pred[k] ≤ maximum(getindex.(y, k))
            end

            # The weights ignore the responses, so a vector response equals
            # the vector of componentwise surrogates.
            s1 = InverseDistanceSurrogate(x, getindex.(y, 1), lb, ub, p = 1.2)
            s2 = InverseDistanceSurrogate(x, getindex.(y, 2), lb, ub, p = 1.2)
            @test surr(3.7) ≈ [s1(3.7), s2(3.7)]

            # The overflow branch must return a whole response.
            s0 = InverseDistanceSurrogate(
                [0.0, 1.0, 2.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 0.0, 2.0, p = 2.0
            )
            @test s0(1.0e-200) ≈ [1.0, 2.0]
            @test s0(1.0) ≈ [3.0, 4.0]
        end

        @testset "ND input" begin
            f = x -> [x[1], x[2]^2]
            lb = [1.0, 2.0]
            ub = [10.0, 8.5]
            x = sample(20, lb, ub, SobolSample())
            push!(x, (1.0, 2.0))
            y = f.(x)
            surr = InverseDistanceSurrogate(x, y, lb, ub, p = 1.2)
            @test surr((1.0, 2.0)) ≈ [1, 4]
            @test all(surr(x[i]) ≈ y[i] for i in eachindex(x))

            x_new = (2.0, 2.0)
            y_new = f(x_new)
            update!(surr, x_new, y_new)
            @test surr(x_new) ≈ y_new
            @test all(isfinite, surr((0.0, 0.0)))
        end
    end
end
