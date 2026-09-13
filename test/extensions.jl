using SafeTestsets, Test

@safetestset "AbstractGPSurrogate" begin
    using AbstractGPs
    using Zygote
    using Surrogates

    @testset "1D -> 1D" begin
        lb = 0.0
        ub = 3.0
        f = x -> log(x) * exp(x)
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        agp1D = AbstractGPSurrogate(x, y, gp = GP(SqExponentialKernel()), Σy = 0.05)
        x_new = 2.5
        y_actual = f.(x_new)
        y_predicted = agp1D([x_new])[1]
        @test isapprox(y_predicted, y_actual, atol = 0.1)
    end

    @testset "add points 1D" begin
        lb = 0.0
        ub = 3.0
        f = x -> x^2
        x_points = sample(5, lb, ub, SobolSample())
        y_points = f.(x_points)
        agp1D = AbstractGPSurrogate(
            [x_points[1]], [y_points[1]],
            gp = GP(SqExponentialKernel()), Σy = 0.05
        )
        x_new = 2.5
        y_actual = f.(x_new)
        for i in 2:length(x_points)
            update!(agp1D, [x_points[i]], [y_points[i]])
        end
        y_predicted = agp1D(x_new)
        @test isapprox(y_predicted, y_actual, atol = 0.1)
    end

    @testset "2D -> 1D" begin
        lb = [0.0; 0.0]
        ub = [2.0; 2.0]
        log_exp_f = x -> log(x[1]) * exp(x[2])
        x = sample(50, lb, ub, SobolSample())
        y = log_exp_f.(x)
        agp_2D = AbstractGPSurrogate(x, y)
        x_new_2D = (2.0, 1.0)
        y_actual = log_exp_f(x_new_2D)
        y_predicted = agp_2D(x_new_2D)
        @test isapprox(y_predicted, y_actual, atol = 0.1)
    end

    @testset "add points 2D" begin
        lb = [0.0; 0.0]
        ub = [2.0; 2.0]
        sphere = x -> x[1]^2 + x[2]^2
        x = sample(20, lb, ub, SobolSample())
        y = sphere.(x)
        agp_2D = AbstractGPSurrogate([x[1]], [y[1]])
        logpdf_vals = []
        push!(logpdf_vals, logpdf_surrogate(agp_2D))
        for i in 2:length(x)
            update!(agp_2D, [x[i]], [y[i]])
            push!(logpdf_vals, logpdf_surrogate(agp_2D))
        end
        @test first(logpdf_vals) < last(logpdf_vals) #as more points are added log marginal posterior predictive probability increases
    end

    @testset "check ND prediction" begin
        lb = [-1.0; -1.0; -1.0]
        ub = [1.0; 1.0; 1.0]
        f = x -> hypot(x...)
        x = sample(25, lb, ub, SobolSample())
        y = f.(x)
        agpND = AbstractGPSurrogate(x, y, gp = GP(SqExponentialKernel()), Σy = 0.05)
        x_new = (-0.8, 0.8, 0.8)
        @test agpND(x_new) ≈ f(x_new) atol = 0.2
    end

    @testset "check working of logpdf_surrogate 1D" begin
        lb = 0.0
        ub = 3.0
        f = x -> log(x) * exp(x)
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        agp1D = AbstractGPSurrogate(x, y, gp = GP(SqExponentialKernel()), Σy = 0.05)
        logpdf_surrogate(agp1D)
    end

    @testset "check working of logpdf_surrogate ND" begin
        lb = [0.0; 0.0]
        ub = [2.0; 2.0]
        f = x -> log(x[1]) * exp(x[2])
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        agpND = AbstractGPSurrogate(x, y, gp = GP(SqExponentialKernel()), Σy = 0.05)
        logpdf_surrogate(agpND)
    end

    @testset "Gradients" begin
        @testset "1D" begin
            lb = 0.0
            ub = 3.0
            n = 100
            x = sample(n, lb, ub, SobolSample())
            f = x -> x^2
            y = f.(x)
            agp1D = AbstractGPSurrogate(x, y, gp = GP(SqExponentialKernel()), Σy = 0.05)
            g = x -> Zygote.gradient(agp1D, x)
            x_val = 2.0
            @test g(x_val)[1] ≈ 2 * x_val rtol = 1.0e-1
        end
        @testset "ND" begin
            lb = [0.0, 0.0]
            ub = [10.0, 10.0]
            n = 100
            x = sample(n, lb, ub, SobolSample())
            f = x -> x[1] * x[2]
            y = f.(x)
            my_agp = AbstractGPSurrogate(x, y, gp = GP(SqExponentialKernel()), Σy = 0.05)
            g = x -> Zygote.gradient(my_agp, x)
            x_val = (2.0, 5.0)
            g_val = g(x_val)[1]
            @test g_val[1] ≈ x_val[2] rtol = 1.0e-1
            @test g_val[2] ≈ x_val[1] rtol = 1.0e-1
        end
    end
end

@safetestset "SurrogatesFlux" begin
    using Surrogates
    using Flux
    using LinearAlgebra
    using Optimisers
    using Random
    using Zygote

    @testset "1D" begin
        a = 0.0
        b = 10.0
        obj_1D = x -> 2 * x + 3
        x = sample(10, 0.0, 10.0, SobolSample())
        y = obj_1D.(x)
        my_model = Chain(Dense(1, 1))
        my_neural_kwargs = NeuralSurrogate(x, y, a, b, model = my_model)
        @test all(
            p === q for
                (p, q) in zip(
                    my_neural_kwargs.ps,
                    Optimisers.trainables(my_neural_kwargs.model)
                )
        )
        my_neural = NeuralSurrogate(x, y, a, b)
        update!(my_neural, [8.5], [20.0])
        update!(my_neural, [3.2, 3.5], [7.4, 8.0])
        @test all(
            p === q for
                (p, q) in zip(my_neural.ps, Optimisers.trainables(my_neural.model))
        )
        val = my_neural(5.0)
    end

    @testset "GENN works with the optimizers" begin
        # `GENNSurrogate` has to work with every optimization method: the
        # optimizers add one lone sample at a time, pass the gradient
        # positionally, and index `genn.x[i]` as a point.
        f1 = x -> (x - 3.7)^2 + 1.0
        f2 = z -> (z[1] - 2.5)^2 + (z[2] - 7.5)^2 + 1.0

        function build(dim)
            lb, ub, obj = dim == 1 ? (1.0, 6.0, f1) : ([1.0, 1.0], [6.0, 6.0], f2)
            Random.seed!(3)
            x = sample(20, lb, ub, SobolSample())
            y = obj.(x)
            dydx = dim == 1 ?
                reshape((t -> 2 * (t - 3.7)).(x), length(x), 1) :
                reduce(
                    vcat,
                    [reshape([2 * (p[1] - 2.5), 2 * (p[2] - 7.5)], 1, 2) for p in x]
                )
            return lb, ub, obj, GENNSurrogate(x, y, lb, ub, dydx, n_epochs = 10)
        end

        @testset "samples are stored as points" begin
            for dim in (1, 2)
                _, _, _, g = build(dim)
                # `length` must be the sample count, not the element count.
                @test length(g.x) == 20
                @test length(g.y) == 20
                @test length(collect(g.x[1])) == dim
            end
        end

        @testset "$(dim)-D $(name)" for dim in (1, 2),
                (name, alg) in (("SRBF", SRBF()), ("DYCORS", DYCORS()), ("SOP", SOP(2)))

            lb, ub, obj, g = build(dim)
            result = surrogate_optimize!(
                obj, alg, lb, ub, g, SobolSample();
                maxiters = 4, needs_gradient = true
            )
            @test result isa Tuple && length(result) == 2
            # The returned point must be a point of the right dimension, and its
            # value must be the objective there. Matrix storage failed both: the
            # "point" was one coordinate and the pair was (coordinate, value).
            @test length(collect(result[1])) == dim
            @test result[2] isa Number
            @test isapprox(
                result[2], obj(dim == 1 ? result[1] : collect(result[1]));
                atol = 1.0e-8
            )
            @test all(lb .- 1.0e-8 .<= collect(result[1]) .<= ub .+ 1.0e-8)
            @test length(g.x) == length(g.y)
            # A gradient-enhanced model must keep one gradient per sample.
            @test size(g.dydx, 3) == length(g.x)
        end

        @testset "$(dim)-D ask-tell" for dim in (1, 2)
            lb, ub, _, g = build(dim)
            points, scores = potential_optimal_points(
                SRBF(), MinimumConstantLiar(), lb, ub, g, SobolSample(), 2
            )
            @test length(points) == length(scores) == 2
            @test all(p -> length(collect(p)) == dim, points)
            @test length(g.x) == 20   # the caller's surrogate is untouched
        end

        @testset "update! takes the gradient positionally" begin
            # `_update_with_sample!` and the virtual-point strategies both call
            # `update!(surr, x, y, gradient)`, as they do for `GEK`.
            lb, ub, obj, g = build(2)
            n_before = length(g.x)
            update!(g, (3.5, 1.4), obj([3.5, 1.4]), [2 * (3.5 - 2.5), 2 * (1.4 - 7.5)])
            @test length(g.x) == n_before + 1
            @test length(g.y) == n_before + 1
            @test size(g.dydx, 3) == length(g.x)
            @test collect(g.x[end]) ≈ [3.5, 1.4]
        end
    end

    @testset "prediction shape and optimizer compatibility" begin
        # The call must return a scalar for a single-output model, not the
        # `k x 1` matrix Flux hands back, since every optimizer compares
        # responses with `<`. A chain ending in `first` already does; both
        # forms are covered below.
        lb, ub = 1.0, 10.0
        h = x -> (x - 3.7)^2 + 1.0
        g = x -> [x, sin(x)]
        Random.seed!(8)
        x = sample(30, lb, ub, RandomSample())

        single() = NeuralSurrogate(
            x, h.(x), lb, ub,
            model = Chain(Dense(1, 6, tanh), Dense(6, 1)), n_epochs = 10
        )
        multi() = NeuralSurrogate(
            x, g.(x), lb, ub,
            model = Chain(Dense(1, 6, tanh), Dense(6, 2)), n_epochs = 10
        )

        @testset "shape matches the fitted responses" begin
            @test single()(5.0) isa Number
            @test multi()(5.0) isa AbstractVector
            @test length(multi()(5.0)) == 2
            # A model that already unwraps, via `first` in the chain, must pass
            # straight through rather than be unwrapped twice.
            withfirst = NeuralSurrogate(
                x, h.(x), lb, ub,
                model = Chain(Dense(1, 1), first), n_epochs = 5
            )
            @test withfirst(5.0) isa Number

            # Every accepted query form gives the same shape.
            lbn, ubn = [0.0, 0.0], [5.0, 5.0]
            Random.seed!(1)
            xn = sample(20, lbn, ubn, SobolSample())
            fn = z -> z[1] * z[2]
            nd = NeuralSurrogate(
                xn, fn.(xn), lbn, ubn,
                model = Chain(Dense(2, 4, tanh), Dense(4, 1)), n_epochs = 5
            )
            @test nd([3.4, 1.4]) isa Number
            @test nd((3.4, 1.4)) isa Number
        end

        @testset "$(name)" for (name, alg) in (
                ("SRBF", SRBF()), ("DYCORS", DYCORS()), ("SOP", SOP(2)),
            )
            surr = single()
            n_before = length(surr.x)
            result = surrogate_optimize!(
                h, alg, lb, ub, surr, SobolSample();
                maxiters = 5
            )
            @test result isa Tuple && length(result) == 2
            # The value is a scalar and is the surrogate's own best observation:
            # this is what a `Matrix{Float32}` response made impossible, since
            # the comparison threw before any of it could be checked.
            @test result[2] isa Number
            @test result[2] ≈ minimum(surr.y)
            # Growth is bounded above but not below. `SOP` adds a sample only on
            # a successful candidate — its failure branch increments the failure
            # count and adds nothing — so a short run may legitimately add none.
            @test n_before <= length(surr.x) <= n_before + 5
            @test length(surr.x) == length(surr.y)
        end

        @testset "ask-tell batches" begin
            surr = single()
            points, scores = potential_optimal_points(
                SRBF(), MinimumConstantLiar(), lb, ub, surr, SobolSample(), 2
            )
            @test length(points) == length(scores) == 2
            @test all(p -> lb - 1.0e-8 <= p <= ub + 1.0e-8, points)
            # The caller's surrogate is untouched by an ask-tell batch.
            @test length(surr.x) == 30
        end

        # AD through the call is covered in `test/AD_compatibility.jl`, testset
        # `AD for extension surrogates`, which asserts the same and additionally
        # cross-checks ForwardDiff against Zygote on the same model.
    end

    @testset "sample storage and update!" begin
        # `NeuralSurrogate` stores the package's own layout — a vector of
        # points and a vector of responses — and converts to matrices only at
        # the Flux boundary. The optimizers read `length(surr.x)` as the sample
        # count and `surr.x[i]` as a point; these pin that contract.
        lb, ub = 1.0, 10.0
        g = x -> [x, sin(x)]
        h = x -> (x - 3.7)^2 + 1.0
        Random.seed!(8)
        x = sample(20, lb, ub, RandomSample())

        @testset "multi-output" begin
            surr = NeuralSurrogate(
                x, g.(x), lb, ub,
                model = Chain(Dense(1, 6, tanh), Dense(6, 2)), n_epochs = 5
            )
            # One entry per sample, on both sides, so `length` is the count.
            @test length(surr.x) == 20
            @test length(surr.y) == 20
            @test length(surr.x) == length(surr.y)
            @test surr.y[1] isa AbstractVector && length(surr.y[1]) == 2
            @test surr.y[1] ≈ g(x[1])

            # A lone multi-output response is one sample, not two scalars. This
            # is the case that used to throw `DimensionMismatch` because
            # `reduce(hcat, [y1, y2])` gave a 1x2 row where a 2x1 was meant.
            update!(surr, 5.0, g(5.0))
            @test length(surr.x) == 21 && length(surr.y) == 21
            @test surr.x[end] ≈ 5.0
            @test surr.y[end] ≈ g(5.0)

            update!(surr, [2.0, 3.0], [g(2.0), g(3.0)])
            @test length(surr.y) == 23
            @test surr.y[end - 1] ≈ g(2.0)
            @test surr.y[end] ≈ g(3.0)
        end

        @testset "scalar output" begin
            surr = NeuralSurrogate(
                x, h.(x), lb, ub,
                model = Chain(Dense(1, 6, tanh), Dense(6, 1)), n_epochs = 5
            )
            @test length(surr.x) == 20 && length(surr.y) == 20
            @test surr.y[1] isa Number
            @test surr.y[1] ≈ h(x[1])

            update!(surr, 5.0, h(5.0))
            @test length(surr.y) == 21
            @test surr.y[end] ≈ h(5.0)

            update!(surr, [2.0, 3.0], [h(2.0), h(3.0)])
            @test length(surr.y) == 23
            @test surr.y[(end - 1):end] ≈ [h(2.0), h(3.0)]
        end

        @testset "N-D points stay points" begin
            lbn, ubn = [0.0, 0.0], [3.0, 3.0]
            Random.seed!(8)
            xn = sample(20, lbn, ubn, RandomSample())
            gn = z -> [z[1]^2 + z[2]^2, (z[1] - 2.0)^2 + z[2]^2]
            surr = NeuralSurrogate(
                xn, gn.(xn), lbn, ubn,
                model = Chain(Dense(2, 6, tanh), Dense(6, 2)), n_epochs = 5
            )
            # `length` must be the sample count, not the element count: this is
            # the exact confusion that produced the wrong Pareto front.
            @test length(surr.x) == 20
            @test length(collect(surr.x[1])) == 2
            @test length(surr.y) == 20
            update!(surr, (1.0, 1.0), gn((1.0, 1.0)))
            @test length(surr.x) == 21
            @test surr.y[end] ≈ gn((1.0, 1.0))
        end
    end

    @testset "ND" begin
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        x = sample(5, lb, ub, SobolSample())
        obj_ND_neural(x) = x[1] * x[2]
        y = obj_ND_neural.(x)
        my_model = Chain(Dense(2, 1))
        my_opt = Descent(0.01)
        my_neural = NeuralSurrogate(
            x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1
        )
        my_neural_kwargs = NeuralSurrogate(x, y, lb, ub, model = my_model)
        my_neural([3.4, 1.4])
        update!(my_neural, [[3.5, 1.4]], [4.9])
        update!(my_neural, [[3.5, 1.4], [1.5, 1.4], [1.3, 1.2]], [1.3, 1.4, 1.5])
    end

    @testset "Multioutput" begin
        f = x -> [x^2, x]
        lb = 1.0
        ub = 10.0
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        my_model = Chain(Dense(1, 2))
        my_opt = Descent(0.01)
        surrogate = NeuralSurrogate(
            x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1
        )

        f = x -> [x[1], x[2]^2]
        lb = [1.0, 2.0]
        ub = [10.0, 8.5]
        x = sample(20, lb, ub, SobolSample())
        y = f.(x)
        my_model = Chain(Dense(2, 2))
        my_opt = Descent(0.01)
        surrogate = NeuralSurrogate(
            x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1
        )
        surrogate([1.0, 2.0])
        x_new = [[2.0, 2.0]]
        y_new = [f(x_new[1])]
        update!(surrogate, x_new, y_new)
    end

    @testset "1D Optimization" begin
        lb = 0.0
        ub = 10.0
        x = sample(5, lb, ub, SobolSample())
        objective_function_1D = z -> 2 * z + 3
        y = objective_function_1D.(x)
        model = Chain(Dense(1, 1), first)
        my_neural_1D_neural = NeuralSurrogate(x, y, lb, ub, model = model)
        surrogate_optimize!(
            objective_function_1D, SRBF(), lb, ub, my_neural_1D_neural,
            SobolSample(), maxiters = 15
        )
    end

    @testset "ND Optimization" begin
        lb = [1.0, 1.0]
        ub = [6.0, 6.0]
        x = sample(5, lb, ub, SobolSample())
        objective_function_ND = z -> 3 * norm(z) + 1
        y = objective_function_ND.(x)
        model = Chain(Dense(2, 1), first)
        opt = Descent(0.01)
        my_neural_ND_neural = NeuralSurrogate(x, y, lb, ub, model = model, loss = Flux.mse)
        surrogate_optimize!(
            objective_function_ND, SRBF(), lb, ub, my_neural_ND_neural,
            SobolSample(), maxiters = 15
        )
    end

    # AD Compatibility
    lb = 0.0
    ub = 3.0
    n = 10
    x = sample(n, lb, ub, SobolSample())
    f = x -> x^2
    y = f.(x)
    #NN
    @testset "NN" begin
        my_model = Chain(Dense(1, 1), first)
        my_opt = Descent(0.01)
        my_neural = NeuralSurrogate(
            x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1
        )
        g = x -> my_neural'(x)
        g(3.4)
    end

    lb = [0.0, 0.0]
    ub = [10.0, 10.0]
    n = 5
    x = sample(n, lb, ub, SobolSample())
    f = x -> x[1] * x[2]
    y = f.(x)

    #NN
    @testset "NN ND" begin
        my_model = Chain(Dense(2, 1), first)
        my_opt = Descent(0.01)
        my_neural = NeuralSurrogate(
            x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1
        )
        # A bare `Zygote.gradient(my_neural, x)` with the result discarded used
        # to sit here. `test/AD_compatibility.jl` now differentiates this
        # surrogate with both backends and checks they agree.
    end

    # ###### ND -> ND ######

    lb = [0.0, 0.0]
    ub = [10.0, 2.0]
    # Six points, not five: a full quadratic in two dimensions has
    # 1 + 2d + d(d - 1) / 2 = 6 coefficients, so five samples leave
    # SecondOrderPolynomialSurrogate underdetermined.
    n = 6
    x = sample(n, lb, ub, SobolSample())
    f = x -> [x[1]^2, x[2]]
    y = f.(x)

    # The multi-output AD checks that used to sit here made no assertions — they
    # called `Zygote.gradient` and dropped the result. They now live in
    # `test/AD_compatibility.jl`, testset `multi-output AD`, with both backends
    # cross-checked and the Jacobian actually verified.
end


@safetestset "GENNSurrogate" begin
    using Surrogates
    using Flux
    using Flux.Optimisers
    using LinearAlgebra
    using Random
    Random.seed!(42)

    @testset "1D" begin
        model = Chain(
            Dense(1, 12, relu),
            Dense(12, 12, relu),
            Dense(12, 1)
        ) |> Flux.f64
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(50, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        genn = GENNSurrogate(x, y, lb, ub, dydx, model = model, n_epochs = 500)
        val = genn(5.0)
        @test val isa Number
        @test isapprox(val, f(5.0), atol = 2.0)
        grad_pred = predict_derivative(genn, [5.0])
        @test grad_pred isa Vector
        @test length(grad_pred) == 1
        @test isapprox(grad_pred[1], df(5.0), atol = 2.0)
    end

    @testset "1D update" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        genn = GENNSurrogate(x, y, lb, ub, dydx, n_epochs = 50)
        update!(genn, [8.5], [72.25], dydx_new = reshape([17.0], 1, 1))
        val = genn(5.0)
        @test val isa Number
    end

    @testset "ND" begin
        model = Chain(
            Dense(2, 12, relu),
            Dense(12, 12, relu),
            Dense(12, 1)
        ) |> Flux.f64
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        f = x -> x[1] * x[2]
        # Gradient: [x[2], x[1]]
        x = sample(50, lb, ub, SobolSample())
        y = f.(x)
        dydx = reduce(hcat, ([xi[2], xi[1]] for xi in x))'
        genn = GENNSurrogate(x, y, lb, ub, dydx, model = model, n_epochs = 500)
        val = genn([3.4, 1.4])
        @test val isa Number
        @test isapprox(val, f([3.4, 1.4]), atol = 2.0)
        grad_pred = predict_derivative(genn, [3.4, 1.4])
        @test grad_pred isa Vector
        @test length(grad_pred) == 2
        @test isapprox(grad_pred[1], 1.4, atol = 2.0)
        @test isapprox(grad_pred[2], 3.4, atol = 2.0)
    end

    @testset "Multi-output" begin
        model = Chain(
            Dense(2, 12, relu),
            Dense(12, 12, relu),
            Dense(12, 2)
        ) |> Flux.f64
        lb = [0.0, 0.0]
        ub = [1.0, 1.0]
        f = x -> [x[1] + 2x[2], 3x[1] - x[2]]
        x = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.25]]
        y = hcat(f.(x)...)
        grad_template = [1.0 2.0; 3.0 -1.0]
        dydx = Array{Float64, 3}(undef, 2, 2, length(x))
        for (i, _) in enumerate(x)
            dydx[:, :, i] = grad_template
        end
        genn = GENNSurrogate(x, y, lb, ub, dydx, model = model, n_epochs = 500, lambda = 0.0)
        val_vec = vec(genn([0.5, 0.25]))
        y_true = f([0.5, 0.25])
        @test length(val_vec) == 2
        @test isapprox(val_vec[1], y_true[1], atol = 0.5)
        @test isapprox(val_vec[2], y_true[2], atol = 0.5)
        grad_pred = predict_derivative(genn, [0.5, 0.25])
        @test size(grad_pred) == (2, 2)
        @test isapprox(grad_pred[1, 1], 1.0, atol = 0.5)
        @test isapprox(grad_pred[1, 2], 2.0, atol = 0.5)
        @test isapprox(grad_pred[2, 1], 3.0, atol = 0.5)
        @test isapprox(grad_pred[2, 2], -1.0, atol = 0.5)
    end

    @testset "ND update" begin
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        f = x -> x[1] * x[2]
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        dydx = reduce(hcat, ([xi[2], xi[1]] for xi in x))'
        genn = GENNSurrogate(x, y, lb, ub, dydx, n_epochs = 50)
        update!(genn, [[3.5, 1.4]], [4.9], dydx_new = [1.4 3.5])  # one sample: gradient at (3.5,1.4) is [1.4, 3.5]
        update!(genn, [[3.5, 1.4], [1.5, 1.4]], [4.9, 2.1], dydx_new = [1.4 3.5; 1.4 1.5])
        val = genn([3.4, 1.4])
        @test val isa Number
    end

    @testset "Different input formats" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        genn = GENNSurrogate(x, y, lb, ub, dydx, n_epochs = 50)
        @test genn(5.0) isa Number
        @test genn([5.0]) isa Number
        @test genn((5.0,)) isa Number
    end

    @testset "Custom optimizer" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(10, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        model = Chain(Dense(1, 8, relu), Dense(8, 1)) |> Flux.f64
        opt = Optimisers.Adam(0.01)
        genn = GENNSurrogate(x, y, lb, ub, dydx, model = model, opt = opt, n_epochs = 50)
        val = genn(5.0)
        @test val isa Number
    end

    @testset "With normalization" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(10, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        model = Chain(Dense(1, 1)) |> Flux.f64
        genn = GENNSurrogate(x, y, lb, ub, dydx, model = model, is_normalize = true, n_epochs = 50)
        val = genn(5.0)
        @test val isa Number
    end

    @testset "Gradient enhancement coefficient" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(10, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        model = Chain(Dense(1, 1)) |> Flux.f64
        genn_low_gamma = GENNSurrogate(x, y, lb, ub, dydx, model = model, gamma = 0.1, n_epochs = 100)
        genn_high_gamma = GENNSurrogate(x, y, lb, ub, dydx, model = model, gamma = 10.0, n_epochs = 100)
        val_low = genn_low_gamma(5.0)
        val_high = genn_high_gamma(5.0)
        @test val_low isa Number
        @test val_high isa Number
    end

    @testset "Normalization" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(100, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        model = Chain(
            Dense(1, 12, relu),
            Dense(12, 12, relu),
            Dense(12, 1)
        ) |> Flux.f64
        genn = GENNSurrogate(x, y, lb, ub, dydx, model = model, is_normalize = true, n_epochs = 100)
        val = genn(5.0)
        @test val isa Number
        @test isapprox(val, f(5.0), atol = 2.0)
        grad_pred = predict_derivative(genn, [5.0])
        @test grad_pred isa Vector
        @test length(grad_pred) == 1
        @test isapprox(grad_pred[1], df(5.0), atol = 2.0)
    end

    @testset "Multi-output update" begin
        lb = [0.0, 0.0]
        ub = [1.0, 1.0]
        f = x -> [x[1] + 2x[2], 3x[1] - x[2]]
        x = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        y = hcat(f.(x)...)
        grad_template = [1.0 2.0; 3.0 -1.0]
        dydx = Array{Float64, 3}(undef, 2, 2, length(x))
        for (i, _) in enumerate(x)
            dydx[:, :, i] = grad_template
        end
        model = Chain(
            Dense(2, 12, relu),
            Dense(12, 12, relu),
            Dense(12, 2)
        ) |> Flux.f64
        genn = GENNSurrogate(x, y, lb, ub, dydx, model = model, n_epochs = 200, lambda = 0.0)
        x_new = [[1.0, 1.0], [0.5, 0.5]]
        y_new = hcat(f.(x_new)...)
        dydx_new = Array{Float64, 3}(undef, 2, 2, 2)
        dydx_new[:, :, 1] = grad_template
        dydx_new[:, :, 2] = grad_template
        update!(genn, x_new, y_new, dydx_new = dydx_new)
        val = genn([0.5, 0.25])
        # Same shape contract as every other multi-output surrogate: a vector,
        # not the raw `k x 1` matrix Flux returns. A matrix response cannot be
        # compared with `<`, which is what the optimizers do.
        @test val isa AbstractVector
        @test length(val) == 2
    end

    @testset "Edge cases" begin
        lb = 0.0
        ub = 10.0
        x = [[1.0], [2.0]]
        y = [1.0, 4.0]
        dydx = reshape([2.0, 4.0], 2, 1)  # df/dx = 2x at x=1,2
        genn = GENNSurrogate(x, y, lb, ub, dydx, n_epochs = 10)
        @test genn(1.5) isa Number
        x = sample(5, lb, ub, SobolSample())
        y = (x -> x^2).(x)
        dydx = reshape(2 .* x, 5, 1)
        genn = GENNSurrogate(x, y, lb, ub, dydx, lambda = 0.0, n_epochs = 50)
        @test genn(5.0) isa Number
    end
end


@safetestset "PolynomialChaosSurrogates" begin
    using Surrogates
    using PolyChaos
    using Zygote

    @testset "Scalar Inputs" begin
        n = 20
        lb = 0.0
        ub = 4.0
        f = x -> 2 * x
        x = sample(n, lb, ub, SobolSample())
        y = f.(x)
        my_pce = PolynomialChaosSurrogate(x, y, lb, ub)
        x_val = 1.2
        @test my_pce(x_val) ≈ f(x_val)
        update!(my_pce, [3.0], [6.0])
        my_pce_changed = PolynomialChaosSurrogate(
            x, y, lb, ub; orthopolys = Uniform01OrthoPoly(1)
        )
        @test my_pce_changed(x_val) ≈ f(x_val)
    end

    @testset "Vector Inputs" begin
        n = 60
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        f = x -> x[1] * x[2]
        x = collect.(sample(n, lb, ub, SobolSample()))
        y = f.(x)
        my_pce = PolynomialChaosSurrogate(x, y, lb, ub)
        x_val = [1.2, 1.4]
        @test my_pce(x_val) ≈ f(x_val)
        update!(my_pce, [[2.0, 3.0]], [6.0])
        @test my_pce(x_val) ≈ f(x_val)
        op1 = Uniform01OrthoPoly(1)
        op2 = Beta01OrthoPoly(2, 2, 1.2)
        ops = [op1, op2]
        multi_poly = MultiOrthoPoly(ops, min(1, 2))
        my_pce_changed = PolynomialChaosSurrogate(x, y, lb, ub, orthopolys = multi_poly)
    end

    @testset "Derivative" begin
        lb = 0.0
        ub = 3.0
        f = x -> x^2
        n = 50
        x = collect(sample(n, lb, ub, SobolSample()))
        y = f.(x)
        my_poli = PolynomialChaosSurrogate(x, y, lb, ub)
        g = x -> my_poli'(x)
        x_val = 3.0
        @test g(x_val) ≈ 2 * x_val
    end

    @testset "Gradient" begin
        n = 50
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        x = collect.(sample(n, lb, ub, SobolSample()))
        f = x -> x[1] * x[2]
        y = f.(x)
        my_poli_ND = PolynomialChaosSurrogate(x, y, lb, ub)
        g = x -> Zygote.gradient(my_poli_ND, x)[1]
        x_val = [1.0, 2.0]
        @test g(x_val) ≈ [x_val[2], x_val[1]]
    end
end

@safetestset "XGBoostSurrogate" begin
    using Surrogates
    using XGBoost: xgboost, predict

    @testset "1D" begin
        obj_1D = x -> 3 * x + 1
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = obj_1D.(x)
        a = 0.0
        b = 10.0
        num_round = 2
        my_forest_1D = XGBoostSurrogate(x, y, a, b; num_round = 2)
        xgboost1 = xgboost((reshape(x, length(x), 1), y); num_round = 2)
        val = my_forest_1D(3.5)
        @test predict(xgboost1, [3.5;;])[1] == val
        update!(my_forest_1D, [6.0], [19.0])
        update!(my_forest_1D, [7.0, 8.0], obj_1D.([7.0, 8.0]))
    end

    @testset "ND" begin
        lb = [0.0, 0.0, 0.0]
        ub = [10.0, 10.0, 10.0]
        x = sample(5, lb, ub, SobolSample())
        obj_ND = x -> x[1] * x[2]^2 * x[3]
        y = obj_ND.(x)
        my_forest_ND = XGBoostSurrogate(x, y, lb, ub; num_round = 2)
        xgboostND = xgboost((reduce(hcat, collect.(x))', y); num_round = 2)
        val = my_forest_ND([1.0, 1.0, 1.0])
        @test predict(xgboostND, reshape([1.0, 1.0, 1.0], 3, 1))[1] == val
        update!(my_forest_ND, [[1.0, 1.0, 1.0]], [1.0])
        update!(my_forest_ND, [[1.2, 1.2, 1.0], [1.5, 1.5, 1.0]], [1.728, 3.375])
    end

    @testset "samples are stored as points, and the optimizers agree" begin
        # `x` was stored as a samples-by-features matrix, so `length(xgb.x)`
        # counted elements and `xgb.x[i]` was a lone coordinate. In N-D that made
        # `SRBF` report a coordinate as the optimum — `(3.03125, 5.158203125)`,
        # a coordinate paired with the objective value — with no error raised,
        # while `DYCORS`, `SOP` and ask-tell died on `BoundsError`.
        using Random, LinearAlgebra
        f2 = z -> (z[1] - 2.5)^2 + (z[2] - 7.5)^2 + 1.0
        f1 = t -> (t - 3.7)^2 + 1.0

        @testset "storage" begin
            lb, ub = [1.0, 1.0], [6.0, 6.0]
            Random.seed!(3)
            x = sample(20, lb, ub, SobolSample())
            s = XGBoostSurrogate(x, f2.(x), lb, ub; num_round = 2)
            @test length(s.x) == 20
            @test length(s.y) == 20
            @test length(collect(s.x[1])) == 2
        end

        @testset "N-D $(name)" for (name, alg) in (
                ("SRBF", SRBF()), ("DYCORS", DYCORS()), ("SOP", SOP(2)),
            )
            lb, ub = [1.0, 1.0], [6.0, 6.0]
            Random.seed!(3)
            x = sample(20, lb, ub, SobolSample())
            s = XGBoostSurrogate(x, f2.(x), lb, ub; num_round = 2)
            result = surrogate_optimize!(
                f2, alg, lb, ub, s, SobolSample();
                maxiters = 4
            )
            # Both halves matter: matrix storage returned a coordinate as the
            # point *and* a value that was not the objective there.
            @test length(collect(result[1])) == 2
            @test isapprox(result[2], f2(collect(result[1])); atol = 1.0e-8)
            @test all(lb .- 1.0e-8 .<= collect(result[1]) .<= ub .+ 1.0e-8)
            @test length(s.x) == length(s.y)
        end

        @testset "N-D ask-tell" begin
            lb, ub = [1.0, 1.0], [6.0, 6.0]
            Random.seed!(3)
            x = sample(20, lb, ub, SobolSample())
            s = XGBoostSurrogate(x, f2.(x), lb, ub; num_round = 2)
            points, scores = potential_optimal_points(
                SRBF(), MinimumConstantLiar(), lb, ub, s, SobolSample(), 2
            )
            @test length(points) == length(scores) == 2
            @test all(p -> length(collect(p)) == 2, points)
            @test length(s.x) == 20   # caller's surrogate untouched
        end

        @testset "1-D is unaffected" begin
            Random.seed!(3)
            x = sample(20, 1.0, 6.0, SobolSample())
            s = XGBoostSurrogate(x, f1.(x), 1.0, 6.0; num_round = 2)
            @test length(s.x) == 20
            @test s(3.0) isa Number
            result = surrogate_optimize!(
                f1, SRBF(), 1.0, 6.0, s, SobolSample();
                maxiters = 4
            )
            @test result[1] isa Number
            @test isapprox(result[2], f1(result[1]); atol = 1.0e-8)
        end
    end
end

@safetestset "SVMSurrogate" begin
    using Surrogates
    using LIBSVM

    @testset "1D" begin
        obj_1D = x -> 2 * x + 1
        a = 0.0
        b = 10.0
        x = sample(5, a, b, SobolSample())
        y = obj_1D.(x)
        svm = LIBSVM.fit!(SVC(), reshape(x, length(x), 1), y)
        my_svm_1D = SVMSurrogate(x, y, a, b)
        val = my_svm_1D([5.0])
        @test LIBSVM.predict(svm, [5.0;;])[1] == val
        update!(my_svm_1D, [3.1], [7.2])
        update!(my_svm_1D, [3.2, 3.5], [7.4, 8.0])
        svm = LIBSVM.fit!(SVC(), reshape(my_svm_1D.x, length(my_svm_1D.x), 1), my_svm_1D.y)
        val = my_svm_1D(3.1)
        @test LIBSVM.predict(svm, [3.1;;])[1] == val
    end

    @testset "ND" begin
        obj_N = x -> x[1]^2 * x[2]
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        x = sample(100, lb, ub, RandomSample())
        y = obj_N.(x)
        svm = LIBSVM.fit!(SVC(), transpose(reduce(hcat, collect.(x))), y)
        my_svm_ND = SVMSurrogate(x, y, lb, ub)
        x_test = [5.0, 1.2]
        val = my_svm_ND(x_test)
        @test LIBSVM.predict(svm, reshape(x_test, 1, 2))[1] == val
        update!(my_svm_ND, [(1.0, 1.0)], [1.0])
        update!(my_svm_ND, [(1.2, 1.2), (1.5, 1.5)], [1.728, 3.375])
        svm = LIBSVM.fit!(
            SVC(), transpose(reduce(hcat, collect.(my_svm_ND.x))), my_svm_ND.y
        )
        x_test = [1.0, 1.0]
        val = my_svm_ND(x_test)
        @test LIBSVM.predict(svm, reshape(x_test, 1, 2))[1] == val
    end
end

@safetestset "MOE" begin
    using StableRNGs, Random
    SEED = 42
    Random.seed!(StableRNG(SEED), SEED)
    @safetestset "1D" begin
        using Surrogates, GaussianMixtures, Flux, PolyChaos, XGBoost

        function discont_1D(x)
            if x < 0.0
                return -5.0
            elseif x >= 0.0
                return 5.0
            end
        end

        lb = -1.0
        ub = 1.0
        x = sample(50, lb, ub, SobolSample())
        y = discont_1D.(x)

        # Radials vs MOE
        RAD_1D = RadialBasis(
            x, y, lb, ub, rad = linearRadial(), scale_factor = 1.0,
            sparse = false
        )
        expert_types = [
            RadialBasisStructure(
                radial_function = linearRadial(), scale_factor = 1.0,
                sparse = false
            ),
            RadialBasisStructure(
                radial_function = cubicRadial(), scale_factor = 1.0,
                sparse = false
            ),
        ]

        MOE_1D_RAD_RAD = MOE(x, y, expert_types)
        MOE_at0 = MOE_1D_RAD_RAD(0.0)
        RAD_at0 = RAD_1D(0.0)
        true_val = 5.0
        @test (abs(RAD_at0 - true_val) > abs(MOE_at0 - true_val))

        # Krig vs MOE
        KRIG_1D = Kriging(x, y, lb, ub, p = 1.0, theta = 1.0)
        expert_types = [
            InverseDistanceStructure(p = 1.0),
            KrigingStructure(p = 1.0, theta = 1.0),
        ]
        MOE_1D_INV_KRIG = MOE(x, y, expert_types)
        MOE_at0 = MOE_1D_INV_KRIG(0.0)
        KRIG_at0 = KRIG_1D(0.0)
        true_val = 5.0
        @test (abs(KRIG_at0 - true_val) > abs(MOE_at0 - true_val))
    end

    @safetestset "ND" begin
        using Surrogates, GaussianMixtures, Flux, PolyChaos, XGBoost

        # helper to test accuracy of predictors
        function rmse(a, b)
            a = vec(a)
            b = vec(b)
            if (size(a) != size(b))
                println("error in inputs")
                return
            end
            n = size(a, 1)
            return sqrt(sum((a - b) .^ 2) / n)
        end

        # multidimensional input function
        function discont_NDIM(x)
            if (x[1] >= 0.0 && x[2] >= 0.0)
                return sum(x .^ 2) + 5
            else
                return sum(x .^ 2) - 5
            end
        end
        lb = [-1.0, -1.0]
        ub = [1.0, 1.0]
        n = 150
        x = sample(n, lb, ub, SobolSample())
        y = discont_NDIM.(x)
        x_test = sample(9, lb, ub, GoldenSample())

        expert_types = [
            KrigingStructure(p = [1.0, 1.0], theta = [1.0, 1.0]),
            RadialBasisStructure(
                radial_function = linearRadial(), scale_factor = 1.0,
                sparse = false
            ),
        ]
        moe_nd_krig_rad = MOE(x, y, expert_types, ndim = 2, quantile = 5)
        moe_pred_vals = moe_nd_krig_rad.(x_test)
        true_vals = discont_NDIM.(x_test)
        moe_rmse = rmse(true_vals, moe_pred_vals)
        rbf = RadialBasis(x, y, lb, ub)
        rbf_pred_vals = rbf.(x_test)
        rbf_rmse = rmse(true_vals, rbf_pred_vals)
        krig = Kriging(x, y, lb, ub, p = [1.0, 1.0], theta = [1.0, 1.0])
        krig_pred_vals = krig.(x_test)
        krig_rmse = rmse(true_vals, krig_pred_vals)
        @test (rbf_rmse > moe_rmse)
        @test (krig_rmse > moe_rmse)
    end

    @safetestset "Miscellaneous" begin
        using Surrogates, GaussianMixtures, Flux, PolyChaos, XGBoost

        # multidimensional input function
        function discont_NDIM(x)
            if (x[1] >= 0.0 && x[2] >= 0.0)
                return sum(x .^ 2) + 5
            else
                return sum(x .^ 2) - 5
            end
        end
        lb = [-1.0, -1.0]
        ub = [1.0, 1.0]
        n = 120
        x = sample(n, lb, ub, LatinHypercubeSample())
        y = discont_NDIM.(x)
        x_test = sample(10, lb, ub, GoldenSample())

        # test if MOE handles 3 experts including SurrogatesFlux
        expert_types = [
            RadialBasisStructure(
                radial_function = linearRadial(), scale_factor = 1.0,
                sparse = false
            ),
            LinearStructure(),
            InverseDistanceStructure(p = 1.0),
        ]
        moe_nd_3_experts = MOE(x, y, expert_types, ndim = 2, n_clusters = 3)
        moe_pred_vals = moe_nd_3_experts.(x_test)

        # test if MOE handles SurrogatesFlux
        model = Chain(Dense(2, 1), first)
        loss = Flux.mse
        opt = Descent(0.01)
        n_epochs = 1
        expert_types = [
            NeuralStructure(model = model, loss = loss, opt = opt, n_epochs = n_epochs),
            LinearStructure(),
        ]
        moe_nn_ln = MOE(x, y, expert_types, ndim = 2)
        moe_pred_vals = moe_nn_ln.(x_test)
    end

    @safetestset "Add Point 1D" begin
        using Surrogates, GaussianMixtures, Flux, PolyChaos, XGBoost

        function discont_1D(x)
            if x < 0.0
                return -5.0
            elseif x >= 0.0
                return 5.0
            end
        end
        lb = -1.0
        ub = 1.0
        x = sample(50, lb, ub, SobolSample())
        y = discont_1D.(x)

        expert_types = [
            RadialBasisStructure(
                radial_function = linearRadial(), scale_factor = 1.0,
                sparse = false
            ),
            RadialBasisStructure(
                radial_function = cubicRadial(), scale_factor = 1.0,
                sparse = false
            ),
        ]
        moe = MOE(x, y, expert_types)
        Surrogates.update!(moe, 0.5, 5.0)
    end

    @safetestset "Add Point ND" begin
        using Surrogates, GaussianMixtures, Flux, PolyChaos, XGBoost

        # multidimensional input function
        function discont_NDIM(x)
            if (x[1] >= 0.0 && x[2] >= 0.0)
                return sum(x .^ 2) + 5
            else
                return sum(x .^ 2) - 5
            end
        end
        lb = [-1.0, -1.0]
        ub = [1.0, 1.0]
        n = 110
        x = sample(n, lb, ub, LatinHypercubeSample())
        y = discont_NDIM.(x)
        expert_types = [
            InverseDistanceStructure(p = 1.0),
            RadialBasisStructure(
                radial_function = linearRadial(), scale_factor = 1.0,
                sparse = false
            ),
        ]
        moe_nd_inv_rad = MOE(x, y, expert_types, ndim = 2)
        Surrogates.update!(moe_nd_inv_rad, (0.5, 0.5), sum((0.5, 0.5) .^ 2) + 5)
    end

    @safetestset "every supported expert type builds" begin
        using Surrogates, GaussianMixtures

        f = x -> x < 5.0 ? 2x : 3x + 5
        lb, ub = 0.0, 10.0
        x = sample(60, lb, ub, SobolSample())
        y = f.(x)

        experts = [
            RadialBasisStructure(
                radial_function = linearRadial(), scale_factor = 1.0, sparse = false
            ),
            KrigingStructure(p = 1.0, theta = 1.0),
            LinearStructure(),
            InverseDistanceStructure(p = 1.0),
            LobachevskyStructure(alpha = 2.0, n = 6, sparse = false),
            SecondOrderPolynomialStructure(),
            WendlandStructure(eps = 1.0, maxiters = 300, tol = 1.0e-6),
        ]
        for e in experts
            moe = MOE(x, y, [e]; ndim = 1, n_clusters = 2)
            @test isfinite(moe(3.0))
        end

        # `GEK` needs `n(1 + d)` observations, values then gradients; a cluster
        # carries one response per point, so the branch could only ever throw.
        @test_throws ArgumentError MOE(
            x, y, [GEKStructure(p = 2.0, theta = 0.5)]; ndim = 1, n_clusters = 2
        )
        # An unsupported name used to `throw` a bare `String`, which is not an
        # `Exception` and so could not be caught by type.
        @test_throws ArgumentError MOE(
            x, y, [(name = "NotASurrogate",)]; ndim = 1, n_clusters = 2
        )
    end

    @safetestset "update! leaves the caller's containers alone" begin
        using Surrogates, GaussianMixtures

        f = x -> x < 5.0 ? 2x : 3x + 5
        lb, ub = 0.0, 10.0
        x = sample(60, lb, ub, SobolSample())
        y = f.(x)
        moe = MOE(
            x, y,
            [
                RadialBasisStructure(
                    radial_function = linearRadial(), scale_factor = 1.0, sparse = false
                ),
            ];
            ndim = 1, n_clusters = 2
        )
        Surrogates.update!(moe, 4.4, f(4.4))
        @test length(x) == 60
        @test length(y) == 60
        @test length(moe.x) == 61
        # Kept inline rather than routed through `check_no_caller_aliasing`:
        # a `@safetestset` is an isolated module, so sharing the helper here
        # would mean an `include` inside every such block, which costs more
        # clarity than the four lines it would save.
    end
end

@safetestset "SurrogatesBase parameter interface for extensions" begin
    using Surrogates
    using SurrogatesBase
    using Random
    using Flux
    using AbstractGPs
    using PolyChaos
    using XGBoost
    using GaussianMixtures
    using LIBSVM

    # The split is the one the core models use: `parameters` is what the fit
    # produced, `hyperparameters` is what governed it. None of these carries a
    # routine that fits its own configuration, so none gets
    # `update_hyperparameters!`; the five that do are covered in
    # `test/interface_tests.jl`.
    f = t -> (t - 3.7)^2 + 1.0
    lb, ub = 1.0, 6.0
    Random.seed!(3)
    x = sample(30, lb, ub, SobolSample())
    y = f.(x)
    dydx = reshape((t -> 2 * (t - 3.7)).(x), length(x), 1)
    experts = [
        RadialBasisStructure(
            radial_function = linearRadial(),
            scale_factor = 1.0, sparse = false
        ),
        RadialBasisStructure(
            radial_function = cubicRadial(),
            scale_factor = 1.0, sparse = false
        ),
    ]

    cases = [
        (
            "NeuralSurrogate",
            () -> NeuralSurrogate(
                x, y, lb, ub,
                model = Chain(Dense(1, 6, tanh), Dense(6, 1)), n_epochs = 5
            ),
            (:ps, :model), (:loss, :opt, :n_epochs),
        ),
        (
            "GENNSurrogate",
            () -> GENNSurrogate(x, y, lb, ub, dydx, n_epochs = 5),
            (:ps, :model, :x_mean, :x_std, :y_mean, :y_std),
            (:opt, :n_epochs, :gamma, :is_normalize),
        ),
        (
            "AbstractGPSurrogate",
            () -> AbstractGPSurrogate(x, y, gp = GP(SqExponentialKernel()), Σy = 0.05),
            (:gp_posterior,), (:gp, :Sigma_y),
        ),
        (
            "PolynomialChaosSurrogate",
            () -> PolynomialChaosSurrogate(x, y, lb, ub),
            (:coeff,), (:orthopolys, :num_of_multi_indexes),
        ),
        (
            "XGBoostSurrogate",
            () -> XGBoostSurrogate(x, y, lb, ub; num_round = 2),
            (:bst,), (:num_round,),
        ),
        ("SVMSurrogate", () -> SVMSurrogate(x, y, lb, ub), (:model,), ()),
        (
            "MOE", () -> MOE(x, y, experts),
            (:cluster_model, :cluster_distributions, :experts),
            (:expert_types, :ndim, :n_clusters, :quantile),
        ),
    ]

    @testset "$(name)" for (name, mk, want_params, want_hyper) in cases
        surr = mk()
        @test parameters(surr) isa NamedTuple
        @test hyperparameters(surr) isa NamedTuple
        # Every advertised key must exist on the struct — a wrong field name
        # here is exactly the bug that slipped through for `GEKPLS`.
        @test keys(parameters(surr)) == want_params
        @test keys(hyperparameters(surr)) == want_hyper
        # These hand back live internal objects, so reading must not disturb
        # the model.
        before = surr(3.0)
        parameters(surr)
        hyperparameters(surr)
        @test surr(3.0) == before
        # No fitting routine, so no refit method.
        @test !hasmethod(update_hyperparameters!, Tuple{typeof(surr)})
    end
end

@safetestset "extension surrogates reject bad input" begin
    using Surrogates
    using Surrogates: sample, SobolSample, update!
    using Flux, NNlib, Optimisers, LIBSVM, PolyChaos, AbstractGPs
    import XGBoost
    using Test

    lb, ub = [0.0, 0.0], [5.0, 5.0]
    x2 = sample(40, lb, ub, SobolSample())
    obj(p) = p[1]^2 + p[2]^2
    y2 = obj.(x2)
    x1 = collect(range(0.0, 5.0, length = 30))
    y1 = x1 .^ 2

    @testset "a query of the wrong dimension is rejected" begin
        # XGBoost's own `predict` accepts a matrix with the wrong feature count
        # and answers anyway, so without the package's own check a scalar query
        # against a 2-dimensional model returned a number.
        xgb = XGBoostSurrogate(x2, y2, lb, ub)
        @test_throws ArgumentError xgb(1.0)
        @test_throws ArgumentError xgb([1.0, 2.0, 3.0])
        @test xgb((1.0, 2.0)) isa Number
    end

    @testset "an untrained fit is rejected" begin
        # Zero rounds and zero epochs both leave the model at its initial state,
        # which is a useless surrogate rather than an error unless it is caught.
        @test_throws ArgumentError XGBoostSurrogate(x1, y1, 0.0, 5.0; num_round = 0)
        @test_throws ArgumentError XGBoostSurrogate(x1, y1, 0.0, 5.0; num_round = -1)
        @test_throws ArgumentError NeuralSurrogate(x1, y1, 0.0, 5.0; n_epochs = 0)
        @test_throws ArgumentError NeuralSurrogate(x1, y1, 0.0, 5.0; n_epochs = -1)
        dydx = reshape(2 .* x1, length(x1), 1)
        @test_throws ArgumentError GENNSurrogate(x1, y1, 0.0, 5.0, dydx; n_epochs = 0)
    end

    @testset "GENNSurrogate demands the gradients it is named for" begin
        # Without them the fit is an ordinary `NeuralSurrogate` under another
        # name, and `predict_derivative` reports slopes nothing constrained.
        @test_throws ArgumentError GENNSurrogate(x1, y1, 0.0, 5.0, nothing)
        # A gradient array of the wrong shape is caught too.
        @test_throws ArgumentError GENNSurrogate(
            x1, y1, 0.0, 5.0, reshape(2 .* x1, 1, length(x1))
        )
    end

    @testset "an under-determined polynomial chaos fit is rejected" begin
        @test_throws ErrorException PolynomialChaosSurrogate(
            x1[1:3], y1[1:3], 0.0, 5.0
        )
    end

    @testset "update! with mismatched points and responses is rejected" begin
        # Each backend reports this differently, but none may accept it.
        @test_throws Exception update!(
            NeuralSurrogate(x1, y1, 0.0, 5.0), [6.0, 7.0], [36.0]
        )
        @test_throws Exception update!(
            XGBoostSurrogate(x1, y1, 0.0, 5.0), [6.0, 7.0], [36.0]
        )
        @test_throws Exception update!(
            PolynomialChaosSurrogate(x1, y1, 0.0, 5.0), [6.0, 7.0], [36.0]
        )
        @test_throws Exception update!(
            AbstractGPSurrogate(x1, y1), [6.0, 7.0], [36.0]
        )
    end

    @testset "a component a composite cannot build is rejected" begin
        genn = GENNStructure(
            model = nothing, opt = Optimisers.Adam(), n_epochs = 1, gamma = 1.0
        )
        @test_throws ArgumentError MOE(x1, y1, [genn, LinearStructure()])
        # A descriptor built by hand, without the `type` field the builder
        # dispatches on.
        @test_throws ArgumentError MOE(
            x1, y1, [(name = "LinearSurrogate",), LinearStructure()]
        )
    end
end

@safetestset "MOE keeps the split it was built with" begin
    using Surrogates
    using Surrogates: sample, SobolSample, update!
    using GaussianMixtures
    using Test

    f = x -> x < 5.0 ? 2x : x^2
    x = sample(60, 0.0, 10.0, SobolSample())
    y = f.(x)
    experts = [
        RadialBasisStructure(
            radial_function = linearRadial(), scale_factor = 1.0,
            sparse = false
        ),
        LinearStructure(),
    ]

    # `quantile` is the row stride of the held-out test split. It was a
    # constructor keyword that nothing stored, so `update!` refit on stride 10
    # whatever the caller asked for.
    moe = MOE(x, y, experts; quantile = 4)
    @test moe.q == 4
    @test hyperparameters(moe).quantile == 4
    update!(moe, 4.4, f(4.4))
    @test moe.q == 4
    @test length(moe.x) == 61
end

@safetestset "extension update! accepts a point in either representation" begin
    using Surrogates
    using Surrogates: sample, SobolSample, update!
    using Flux, NNlib, Optimisers, LIBSVM, PolyChaos, AbstractGPs, GaussianMixtures
    import XGBoost
    using Test

    # The same contract the core surrogates hold in `test/interface_tests.jl`.
    # `Surrogates._match_stored` used to exist only as two private copies here,
    # so `AbstractGPSurrogate`, `PolynomialChaosSurrogate`, `SVMSurrogate` and
    # `MOE` rejected a coordinate-vector point against a tuple-stored design.
    lb, ub = [0.0, 0.0], [5.0, 5.0]
    obj(p) = p[1]^2 + p[2]^2
    x_tuples = sample(50, lb, ub, SobolSample())
    x_vectors = [collect(p) for p in x_tuples]
    y = obj.(x_tuples)
    labels = round.(Int, y) .% 2

    experts = [LinearStructure(), InverseDistanceStructure(p = 2.0)]
    cases = [
        ("NeuralSurrogate", (x, r) -> NeuralSurrogate(x, r, lb, ub), y, true),
        ("XGBoostSurrogate", (x, r) -> XGBoostSurrogate(x, r, lb, ub), y, true),
        ("AbstractGPSurrogate", (x, r) -> AbstractGPSurrogate(x, r), y, true),
        (
            "PolynomialChaosSurrogate",
            (x, r) -> PolynomialChaosSurrogate(x, r, lb, ub), y, true,
        ),
        ("SVMSurrogate", (x, r) -> SVMSurrogate(x, r, lb, ub), labels, true),
        ("MOE", (x, r) -> MOE(x, r, experts; ndim = 2), y, true),
    ]

    @testset "$(name)" for (name, build, responses, batch) in cases
        one = name == "SVMSurrogate" ? 1 : obj((1.0, 2.0))
        two = name == "SVMSurrogate" ? [1, 0] :
            [obj((1.0, 2.0)), obj((2.0, 3.0))]

        grew(store, new_x, new_y) = begin
            surr = build(store, responses)
            n = length(surr.x)
            update!(surr, new_x, new_y)
            length(surr.x) - n
        end

        @test grew(x_tuples, (1.0, 2.0), one) == 1
        @test grew(x_tuples, [1.0, 2.0], one) == 1
        @test grew(x_vectors, (1.0, 2.0), one) == 1
        @test grew(x_vectors, [1.0, 2.0], one) == 1
        if batch
            @test grew(x_tuples, [(1.0, 2.0), (2.0, 3.0)], two) == 2
            @test grew(x_tuples, [[1.0, 2.0], [2.0, 3.0]], two) == 2
        end
    end

    @testset "GENNSurrogate" begin
        # `_normalize_x` judges shape from the argument alone, so a coordinate
        # vector looked like `d` one-dimensional samples until the stored design
        # was brought in to settle it.
        gmat = reduce(
            vcat,
            [reshape([2p[1], 2p[2]], 1, 2) for p in x_tuples]
        )
        for point in ((1.0, 2.0), [1.0, 2.0])
            genn = GENNSurrogate(x_tuples, y, lb, ub, gmat; n_epochs = 2)
            n = length(genn.x)
            update!(genn, point, obj(point); dydx_new = [2.0 4.0])
            @test length(genn.x) == n + 1
            @test size(genn.dydx, 3) == length(genn.x)
        end
    end
end

@safetestset "the Flux surrogates infer their return type" begin
    using Surrogates
    using Surrogates: sample, SobolSample
    using Flux, NNlib, Optimisers
    using Test

    # The call branched on `size(out, 1)` at runtime, so it inferred
    # `Union{Float32, Vector{Float32}}` and every optimizer comparison had to
    # resolve it dynamically. A surrogate's output count is fixed when it is
    # fitted and the stored response type already records it, so the shape is
    # settled by dispatch instead.
    lb, ub = [0.0, 0.0], [5.0, 5.0]
    x = sample(30, lb, ub, SobolSample())
    scalar_y = [p[1]^2 + p[2]^2 for p in x]
    vector_y = [[p[1]^2 + p[2]^2, 2 * (p[1] + p[2])] for p in x]
    dydx = reduce(vcat, [reshape([2p[1], 2p[2]], 1, 2) for p in x])
    dydx3 = Array{Float64, 3}(undef, 2, 2, length(x))
    for (i, p) in enumerate(x)
        dydx3[1, :, i] = [2p[1], 2p[2]]
        dydx3[2, :, i] = [2.0, 2.0]
    end
    q = (1.0, 2.0)

    @testset "NeuralSurrogate" begin
        single = NeuralSurrogate(x, scalar_y, lb, ub)
        multi = NeuralSurrogate(x, vector_y, lb, ub; model = Chain(Dense(2, 2)))
        @test (@inferred single(q)) isa Number
        @test (@inferred multi(q)) isa AbstractVector
        @test length(multi(q)) == 2
    end

    @testset "GENNSurrogate" begin
        single = GENNSurrogate(x, scalar_y, lb, ub, dydx; n_epochs = 2)
        multi = GENNSurrogate(
            x, vector_y, lb, ub, dydx3;
            n_epochs = 2, model = Chain(Dense(2, 8, relu), Dense(8, 2))
        )
        @test (@inferred single(q)) isa Number
        @test (@inferred multi(q)) isa AbstractVector
        @test length(multi(q)) == 2
    end

    @testset "multi-output is given the same way as every other surrogate" begin
        # A vector of per-sample response vectors. The matrix form is accepted
        # too, and must produce the identical design.
        matrix_y = reduce(hcat, vector_y)
        from_vectors = NeuralSurrogate(x, vector_y, lb, ub; model = Chain(Dense(2, 2)))
        from_matrix = NeuralSurrogate(x, matrix_y, lb, ub; model = Chain(Dense(2, 2)))
        @test from_vectors.x == from_matrix.x
        @test from_vectors.y == from_matrix.y

        genn_vectors = GENNSurrogate(
            x, vector_y, lb, ub, dydx3; n_epochs = 2,
            model = Chain(Dense(2, 8, relu), Dense(8, 2))
        )
        genn_matrix = GENNSurrogate(
            x, matrix_y, lb, ub, dydx3; n_epochs = 2,
            model = Chain(Dense(2, 8, relu), Dense(8, 2))
        )
        @test genn_vectors.x == genn_matrix.x
        @test genn_vectors.y == genn_matrix.y
    end
end
