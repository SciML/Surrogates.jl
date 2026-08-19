using Surrogates
using Test

@testset "LinearSurrogate" begin
    @testset "1D input" begin
        @testset "affine recovery" begin
            # The surrogate must recover an affine function exactly, including
            # its intercept, and stay exact after consistent updates.
            f = x -> 2 * x + 5
            x = [0.0, 1.0, 2.0, 3.0]
            lin = LinearSurrogate(x, f.(x), 0.0, 7.0)
            @test lin.coeff ≈ [5.0, 2.0]
            @test lin(1.5) ≈ f(1.5)
            update!(lin, 4.0, f(4.0))
            update!(lin, [5.0, 6.0], f.([5.0, 6.0]))
            @test lin.coeff ≈ [5.0, 2.0]
            @test lin(5.5) ≈ f(5.5)
        end

        @testset "least-squares optimality" begin
            # For non-affine data the fit must satisfy the normal equations:
            # residuals orthogonal to the intercept column and to x.
            x = [1.0, 2.0, 3.0]
            y = [1.5, 3.5, 5.3]
            lin = LinearSurrogate(x, y, 0.0, 7.0)
            r = y .- lin.(x)
            @test isapprox(sum(r), 0.0; atol = 1.0e-10)
            @test isapprox(sum(r .* x), 0.0; atol = 1.0e-10)

            update!(lin, 4.0, 7.2)
            update!(lin, [5.0, 6.0], [8.3, 9.7])
            @test length(lin.x) == 6
            @test length(lin.y) == 6
            # Optimality must be restored after the refit.
            r = lin.y .- lin.(lin.x)
            @test isapprox(sum(r), 0.0; atol = 1.0e-10)
            @test isapprox(sum(r .* lin.x), 0.0; atol = 1.0e-10)
        end

        @testset "update! matches a fresh fit" begin
            x = [1.0, 2.0, 3.0, 4.0]
            y = [1.5, 3.5, 5.3, 6.9]
            lin = LinearSurrogate(x[1:2], y[1:2], 0.0, 7.0)
            update!(lin, x[3:4], y[3:4])
            fresh = LinearSurrogate(x, y, 0.0, 7.0)
            @test lin.coeff ≈ fresh.coeff
            @test lin(2.5) ≈ fresh(2.5)
            # A one-element batch is still a batch, not a single sample.
            update!(lin, [5.0], [8.1])
            @test length(lin.x) == 5
        end

        @testset "sample-count regimes" begin
            # A single sample leaves the fit underdetermined; the least-squares
            # solve returns the minimum-norm answer, which still interpolates.
            one_pt = LinearSurrogate([1.0], [3.0], 0.0, 2.0)
            @test one_pt(1.0) ≈ 3.0
            # Repeated samples leave the design rank deficient but tall, so the
            # solve returns a fit rather than raising.
            dup = LinearSurrogate([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], 0.0, 5.0)
            @test dup(2.0) ≈ 1.0
        end

        @testset "integer samples promote" begin
            lin = LinearSurrogate([1, 2, 3], [7, 9, 11], 0, 5)
            @test lin.coeff ≈ [5.0, 2.0]
            @test lin(4) ≈ 13.0
        end

        @testset "input dimension is validated" begin
            lin = LinearSurrogate([1.0, 2.0, 3.0], [7.0, 9.0, 11.0], 0.0, 5.0)
            @test_throws ArgumentError lin(Float64[])
            @test_throws ArgumentError lin((2.0, 3.0))
        end
    end

    @testset "ND input" begin
        f = x -> 3.0 + 2.0 * x[1] - x[2]

        @testset "affine recovery (tuple points)" begin
            lb = [0.0, 0.0]
            ub = [10.0, 10.0]
            x = sample(6, lb, ub, SobolSample())
            lin = LinearSurrogate(x, f.(x), lb, ub)
            @test lin.coeff ≈ [3.0, 2.0, -1.0]
            @test lin((4.0, 8.0)) ≈ f((4.0, 8.0))
            update!(lin, (1.5, 2.5), f((1.5, 2.5)))
            update!(lin, [(8.0, 5.0), (9.0, 9.5)], f.([(8.0, 5.0), (9.0, 9.5)]))
            @test lin.coeff ≈ [3.0, 2.0, -1.0]
            @test lin((4.0, 8.0)) ≈ f((4.0, 8.0))
        end

        @testset "vector points" begin
            # A point container may be a vector rather than a tuple.
            x = [[1.0, 2.0], [3.0, 1.0], [2.0, 5.0], [6.0, 4.0]]
            lin = LinearSurrogate(x, f.(x), [0.0, 0.0], [10.0, 10.0])
            @test lin.coeff ≈ [3.0, 2.0, -1.0]
            @test lin([4.0, 8.0]) ≈ f([4.0, 8.0])
            update!(lin, [1.5, 2.5], f([1.5, 2.5]))
            update!(lin, [[7.0, 7.0], [8.0, 8.0]], f.([[7.0, 7.0], [8.0, 8.0]]))
            @test length(lin.x) == 7
            @test lin.coeff ≈ [3.0, 2.0, -1.0]
        end

        @testset "update! requires the sample container type to match" begin
            # `x` is stored in a concretely typed field, so a surrogate built
            # from tuples cannot absorb a vector-valued point. Current
            # limitation, pinned so a future fix is noticed.
            x = [(1.0, 2.0), (3.0, 1.0), (2.0, 5.0), (6.0, 4.0)]
            lin = LinearSurrogate(x, f.(x), [0.0, 0.0], [10.0, 10.0])
            @test_throws MethodError update!(lin, [1.5, 2.5], f([1.5, 2.5]))
            update!(lin, (1.5, 2.5), f((1.5, 2.5)))
            @test length(lin.x) == 5
        end

        @testset "one-dimensional points wrapped in 1-tuples" begin
            # d == 1 in the ND container form takes the ND design-matrix branch
            # but must give the same affine fit.
            x = [(1.0,), (2.0,), (3.0,)]
            lin = LinearSurrogate(x, [2t[1] + 5 for t in x], 0.0, 5.0)
            @test lin.coeff ≈ [5.0, 2.0]
            @test lin((4.0,)) ≈ 13.0
        end

        @testset "least-squares optimality" begin
            lb = [0.0, 0.0]
            ub = [10.0, 10.0]
            x = sample(5, lb, ub, SobolSample())
            lin = LinearSurrogate(x, [4.0, 5.0, 6.0, 7.0, 8.0], lb, ub)
            update!(lin, (10.0, 11.0), 9.0)
            update!(lin, [(8.0, 5.0), (9.0, 9.5)], [4.0, 5.0])
            # Residuals orthogonal to the intercept column and every coordinate.
            r = lin.y .- lin.(lin.x)
            @test isapprox(sum(r), 0.0; atol = 1.0e-8)
            for j in 1:2
                @test isapprox(sum(r .* getindex.(lin.x, j)), 0.0; atol = 1.0e-8)
            end
        end

        @testset "sample-count regimes" begin
            # Exactly determined: 3 samples for 3 coefficients, fitted exactly.
            x3 = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
            @test LinearSurrogate(x3, f.(x3), [0.0, 0.0], [1.0, 1.0]).coeff ≈
                [3.0, 2.0, -1.0]
            # Underdetermined: fewer samples than coefficients gives the
            # minimum-norm solution rather than an error, and still reproduces
            # the data it was given.
            x2 = [(0.0, 0.0), (1.0, 0.0)]
            lin = LinearSurrogate(x2, f.(x2), [0.0, 0.0], [1.0, 1.0])
            @test all(isapprox(lin(p), f(p); atol = 1.0e-10) for p in x2)
        end

        @testset "call forms agree" begin
            x = [(1.0, 2.0), (3.0, 1.0), (2.0, 5.0), (6.0, 4.0)]
            lin = LinearSurrogate(x, f.(x), [0.0, 0.0], [10.0, 10.0])
            expected = f((4.0, 8.0))
            # Tuple, vector, and the 1xd row-matrix form that bounds arithmetic
            # produces must all evaluate identically.
            @test lin((4.0, 8.0)) ≈ expected
            @test lin([4.0, 8.0]) ≈ expected
            @test lin([4.0 8.0]) ≈ expected
        end

        @testset "tuple bounds are accepted" begin
            x = [(1.0, 2.0), (3.0, 1.0), (2.0, 5.0), (6.0, 4.0)]
            lin = LinearSurrogate(x, f.(x), (0.0, 0.0), (10.0, 10.0))
            @test lin.coeff ≈ [3.0, 2.0, -1.0]
        end

        @testset "input dimension is validated" begin
            x = [(1.0, 2.0), (3.0, 1.0), (2.0, 5.0), (6.0, 4.0)]
            lin = LinearSurrogate(x, f.(x), [0.0, 0.0], [10.0, 10.0])
            @test_throws ArgumentError lin(Float64[])
            @test_throws ArgumentError lin(2.0)
            @test_throws ArgumentError lin((1.0, 2.0, 3.0))
        end
    end

    @testset "ND output" begin
        @testset "1D input, vector responses" begin
            f = x -> [2 * x + 5, -x + 1]
            x = [0.0, 1.0, 2.0, 3.0]
            lin = LinearSurrogate(x, f.(x), 0.0, 7.0)
            # One column per output: [intercepts; slopes].
            @test size(lin.coeff) == (2, 2)
            @test lin.coeff ≈ [5.0 1.0; 2.0 -1.0]
            @test lin(1.5) ≈ f(1.5)
            @test lin(1.5) isa AbstractVector
        end

        @testset "ND input, vector responses" begin
            f = x -> [3.0 + 2.0 * x[1] - x[2], x[1] + x[2]]
            lb = [0.0, 0.0]
            ub = [10.0, 10.0]
            x = sample(6, lb, ub, SobolSample())
            lin = LinearSurrogate(x, f.(x), lb, ub)
            @test size(lin.coeff) == (3, 2)
            @test lin.coeff ≈ [3.0 0.0; 2.0 1.0; -1.0 1.0]
            @test lin((4.0, 8.0)) ≈ f((4.0, 8.0))
        end

        @testset "each output is fitted independently" begin
            # Fitting m outputs jointly must equal fitting each one separately:
            # the design matrix is shared, so the solve decouples by column.
            g1 = x -> x^2
            g2 = x -> 3 * x - 1
            x = sample(12, 0.0, 10.0, SobolSample())
            joint = LinearSurrogate(x, [[g1(t), g2(t)] for t in x], 0.0, 10.0)
            sep1 = LinearSurrogate(x, g1.(x), 0.0, 10.0)
            sep2 = LinearSurrogate(x, g2.(x), 0.0, 10.0)
            @test joint.coeff[:, 1] ≈ sep1.coeff
            @test joint.coeff[:, 2] ≈ sep2.coeff
            @test joint(4.2) ≈ [sep1(4.2), sep2(4.2)]
        end

        @testset "update! with vector responses" begin
            f = x -> [2 * x + 5, -x + 1]
            x = [0.0, 1.0, 2.0]
            lin = LinearSurrogate(x, f.(x), 0.0, 7.0)
            # A single vector response must not be splatted into two samples.
            update!(lin, 3.0, f(3.0))
            @test length(lin.x) == 4
            @test length(lin.y) == 4
            update!(lin, [4.0, 5.0], f.([4.0, 5.0]))
            @test length(lin.x) == 6
            @test lin.coeff ≈ [5.0 1.0; 2.0 -1.0]
            @test lin(6.0) ≈ f(6.0)
        end

        @testset "scalar responses stay scalar" begin
            # The container of the responses decides the container of the
            # prediction; scalars must not become one-element vectors.
            lin = LinearSurrogate([0.0, 1.0, 2.0], [5.0, 7.0, 9.0], 0.0, 3.0)
            @test lin(1.5) isa Number
            vec_lin = LinearSurrogate([0.0, 1.0, 2.0], [[5.0], [7.0], [9.0]], 0.0, 3.0)
            @test vec_lin(1.5) isa AbstractVector
            @test length(vec_lin(1.5)) == 1
        end
    end
end
