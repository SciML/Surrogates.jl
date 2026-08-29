using Surrogates
using Test
using ForwardDiff
using Zygote

@testset "EarthSurrogate" begin
    # A piecewise-linear response lies exactly in the span of the hinge basis,
    # so it is the one family the surrogate can reproduce to machine precision
    # and the one that pins down its behavior without a tolerance to argue over.
    hinge_1d = t -> 1 + 2 * t + 3 * max(0, t - 4)
    grid_1d = collect(0.0:0.5:10.0)

    lb_nd = [0.0, 0.0]
    ub_nd = [10.0, 10.0]
    x_nd = sample(60, lb_nd, ub_nd, SobolSample())
    additive_nd = p -> 2 * p[1] + 3 * max(0, p[2] - 5)

    @testset "the mean response is not counted twice" begin
        # The basis is fitted to the response about its mean and the mean is
        # added back at evaluation. Fitting the raw response instead lets the
        # coefficients absorb the mean, which is then added a second time — a
        # bias of order the mean itself, and invisible in any test that only
        # asks whether the surrogate returns a number.
        #
        # Least squares makes the residual orthogonal to every column of the
        # design, and the model carries a constant, so the residual over the
        # training samples must have zero mean. That is the property the double
        # count destroys, so it is the property asserted here.
        for (x, y) in (
                (grid_1d, hinge_1d.(grid_1d)),
                (x_nd, additive_nd.(x_nd)),
            )
            earth = EarthSurrogate(x, y, first(x), last(x))
            residual = [earth(p) - fp for (p, fp) in zip(x, y)]
            @test isapprox(sum(residual) / length(residual), 0.0, atol = 1.0e-8)
            @test earth.intercept ≈ sum(y) / length(y)
            # And the fit must actually beat the constant model it is built on.
            spread = sum(abs2, y .- sum(y) / length(y))
            @test sum(abs2, residual) < 1.0e-3 * spread
        end
    end

    @testset "1D recovery of a piecewise-linear response" begin
        y = hinge_1d.(grid_1d)
        earth = EarthSurrogate(grid_1d, y, 0.0, 10.0)
        # Exact at the samples, and between them: the target is in the span.
        @test maximum(abs, [earth(p) - hinge_1d(p) for p in grid_1d]) < 1.0e-10
        for p in (0.3, 1.7, 3.9, 4.0, 4.1, 7.3, 9.8)
            @test earth(p) ≈ hinge_1d(p) atol = 1.0e-10
        end
        @test earth(3.0) isa Number
    end

    @testset "1D regression of a smooth response" begin
        x = sample(50, 0.0, 5.0, SobolSample())
        f = t -> t^2
        y = f.(x)
        earth = EarthSurrogate(x, y, 0.0, 5.0)
        # Piecewise linear through a parabola: close, but not an interpolant.
        @test maximum(abs, [earth(p) - f(p) for p in x]) < 0.05 * (maximum(y) - minimum(y))
        @test !all(isapprox(earth(p), f(p); atol = 1.0e-10) for p in x)
    end

    @testset "ND recovery of an additive response" begin
        y = additive_nd.(x_nd)
        earth = EarthSurrogate(x_nd, y, lb_nd, ub_nd)
        @test maximum(abs, [earth(p) - additive_nd(p) for p in x_nd]) <
            0.02 * (maximum(y) - minimum(y))
        # Both coordinates carry structure, so both must be selected.
        @test sort(unique(t.dim for t in earth.basis)) == [1, 2]
        @test earth((2.0, 7.0)) isa Number
    end

    @testset "knots are sampled coordinates" begin
        earth = EarthSurrogate(grid_1d, hinge_1d.(grid_1d), 0.0, 10.0)
        @test all(t -> t.knot in grid_1d, earth.basis)
        # The forward pass adds reflected pairs, so every knot appears with a
        # hinge and its mirror.
        for knot in unique(t.knot for t in earth.basis)
            @test sort([t.mirror for t in earth.basis if t.knot == knot]) == [false, true]
        end
    end

    @testset "input containers" begin
        y = hinge_1d.(grid_1d)
        earth = EarthSurrogate(grid_1d, y, 0.0, 10.0)
        # A one-dimensional point is its own coordinate however it is wrapped.
        @test earth(3.7) ≈ earth([3.7])
        @test earth(3.7) ≈ earth((3.7,))

        y_nd = additive_nd.(x_nd)
        tuples = EarthSurrogate(x_nd, y_nd, lb_nd, ub_nd)
        vectors = EarthSurrogate(collect.(x_nd), y_nd, lb_nd, ub_nd)
        @test tuples((3.0, 8.0)) ≈ vectors((3.0, 8.0))
        @test tuples((3.0, 8.0)) ≈ tuples([3.0, 8.0])
    end

    @testset "element types" begin
        # Knots are stored at the samples' own type; the design matrix and the
        # coefficients are promoted the way `\` would promote them.
        for (T, xs) in (
                (Float32, Float32.(grid_1d)),
                (BigFloat, BigFloat.(grid_1d)),
                (Int, collect(0:1:20)),
                (Rational{Int}, collect((0 // 1):(1 // 2):(10 // 1))),
            )
            y = [1 + 2 * t + 3 * max(0, t - 4) for t in xs]
            earth = EarthSurrogate(xs, y, first(xs), last(xs))
            @test typeof(first(earth.basis).knot) == T
            @test eltype(earth.coeff) == float(T)
            @test earth(xs[7]) ≈ y[7] atol = sqrt(eps(float(T)))
        end
        # Precision is carried, not silently narrowed to Float64.
        big_earth = EarthSurrogate(
            BigFloat.(grid_1d), BigFloat.(hinge_1d.(grid_1d)), BigFloat(0), BigFloat(10)
        )
        @test big_earth(BigFloat("3.7")) isa BigFloat
    end

    @testset "keywords bound the basis" begin
        y = hinge_1d.(grid_1d)
        # `n_max_terms` counts reflected pairs, so it caps the basis at twice it.
        for pairs in (1, 2, 3)
            earth = EarthSurrogate(
                grid_1d, y, 0.0, 10.0; n_max_terms = pairs, n_min_terms = 1
            )
            @test length(earth.basis) <= 2 * pairs
        end
        # A heavier GCV penalty can never leave a larger basis behind. Past the
        # point where the effective parameter count saturates the samples every
        # candidate scores `Inf`; pruning has to keep going there rather than
        # stall, or a heavy penalty returns the whole unpruned forward basis.
        sizes = [
            length(
                    EarthSurrogate(
                        grid_1d, y, 0.0, 10.0; penalty = p, n_min_terms = 1
                    ).basis
                ) for p in (0.0, 2.0, 10.0, 20.0, 50.0, 200.0)
        ]
        @test issorted(sizes, rev = true)
        @test first(sizes) > last(sizes)
        # The basis can never outgrow the samples that have to determine it.
        few = [0.0, 2.5, 4.0, 6.0, 10.0]
        @test length(EarthSurrogate(few, hinge_1d.(few), 0.0, 10.0).basis) <= length(few)
    end

    @testset "responses with nothing to explain are rejected" begin
        # A constant response leaves the intercept as the whole model, so no
        # hinge can reduce a residual that is already zero.
        @test_throws ArgumentError EarthSurrogate(grid_1d, fill(3.0, length(grid_1d)), 0.0, 10.0)
        # So does a response whose variation is below the stopping tolerance.
        @test_throws ArgumentError EarthSurrogate(
            grid_1d, hinge_1d.(grid_1d), 0.0, 10.0; rel_res_error = 1.1
        )
    end

    @testset "query dimension is checked" begin
        my_ear1d = EarthSurrogate(grid_1d, hinge_1d.(grid_1d), 0.0, 10.0)
        @test_throws ArgumentError my_ear1d(Float64[])
        @test_throws ArgumentError my_ear1d((2.0, 3.0, 4.0))

        my_earnd = EarthSurrogate(x_nd, additive_nd.(x_nd), lb_nd, ub_nd)
        @test_throws ArgumentError my_earnd(Float64[])
        @test_throws ArgumentError my_earnd(2.0)
        @test_throws ArgumentError my_earnd((2.0, 3.0, 4.0))
    end

    @testset "update!" begin
        @testset "1D" begin
            x = copy(grid_1d)
            y = hinge_1d.(x)
            earth = EarthSurrogate(x, y, 0.0, 12.0)
            update!(earth, 11.0, hinge_1d(11.0))
            @test length(earth.x) == length(grid_1d) + 1
            @test earth.intercept ≈ sum(earth.y) / length(earth.y)
            @test earth(11.0) ≈ hinge_1d(11.0) atol = 1.0e-8
            # The caller's arrays are left alone.
            @test x == grid_1d
            @test length(y) == length(grid_1d)

            # A batch of samples, not one.
            update!(earth, [11.5, 12.0], hinge_1d.([11.5, 12.0]))
            @test length(earth.x) == length(grid_1d) + 3
            @test earth(12.0) ≈ hinge_1d(12.0) atol = 1.0e-8
        end

        @testset "ND" begin
            earth = EarthSurrogate(x_nd, additive_nd.(x_nd), lb_nd, ub_nd)
            before = length(earth.x)
            update!(earth, (2.0, 2.0), additive_nd((2.0, 2.0)))
            @test length(earth.x) == before + 1
            @test earth.intercept ≈ sum(earth.y) / length(earth.y)
            @test length(earth.coeff) == length(earth.basis)

            update!(earth, [(1.0, 1.0), (9.0, 9.0)], additive_nd.([(1.0, 1.0), (9.0, 9.0)]))
            @test length(earth.x) == before + 3
        end
    end

    @testset "automatic differentiation" begin
        y = hinge_1d.(grid_1d)
        earth = EarthSurrogate(grid_1d, y, 0.0, 10.0)

        @testset "1D" begin
            # The target is in the span, so away from its knot the slope is
            # exact rather than merely close.
            @test ForwardDiff.derivative(earth, 2.0) ≈ 2.0 atol = 1.0e-8
            @test ForwardDiff.derivative(earth, 7.0) ≈ 5.0 atol = 1.0e-8
            # Piecewise linear: the second derivative vanishes inside a segment.
            @test ForwardDiff.derivative(t -> ForwardDiff.derivative(earth, t), 2.3) ≈ 0.0 atol = 1.0e-8
            # Forward and reverse mode see the same function.
            for p in (1.3, 2.0, 6.6, 8.4)
                @test Zygote.gradient(earth, p)[1] ≈ ForwardDiff.derivative(earth, p)
            end
        end

        @testset "ND" begin
            earth_nd = EarthSurrogate(x_nd, additive_nd.(x_nd), lb_nd, ub_nd)
            g = ForwardDiff.gradient(earth_nd, [3.0, 8.0])
            @test g isa AbstractVector
            @test g ≈ [2.0, 3.0] atol = 0.1
            # The model is additive, so each partial depends on its own
            # coordinate alone.
            @test ForwardDiff.gradient(earth_nd, [3.0, 8.0])[1] ≈
                ForwardDiff.gradient(earth_nd, [3.0, 2.0])[1] atol = 1.0e-8
            @test Zygote.gradient(earth_nd, [3.0, 8.0])[1] ≈ g
        end
    end
end
