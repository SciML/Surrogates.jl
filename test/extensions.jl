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
    using Zygote

    @testset "1D" begin
        a = 0.0
        b = 10.0
        obj_1D = x -> 2 * x + 3
        x = sample(10, 0.0, 10.0, SobolSample())
        y = obj_1D.(x)
        my_model = Chain(Dense(1, 1))
        my_neural_kwargs = NeuralSurrogate(x, y, a, b, model = my_model)
        my_neural = NeuralSurrogate(x, y, a, b)
        update!(my_neural, [8.5], [20.0])
        update!(my_neural, [3.2, 3.5], [7.4, 8.0])
        val = my_neural(5.0)
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
        g = x -> Zygote.gradient(my_neural, x)
        g([2.0, 5.0])
    end

    # ###### ND -> ND ######

    lb = [0.0, 0.0]
    ub = [10.0, 2.0]
    n = 5
    x = sample(n, lb, ub, SobolSample())
    f = x -> [x[1]^2, x[2]]
    y = f.(x)

    #NN
    @testset "NN ND -> ND" begin
        my_model = Chain(Dense(2, 2))
        my_opt = Descent(0.01)
        my_neural = NeuralSurrogate(
            x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1
        )
        Zygote.gradient(x -> sum(my_neural(x)), [2.0, 5.0])

        my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
        Zygote.gradient(x -> sum(my_rad(x)), (2.0, 5.0))

        my_p = 1.4
        my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
        my_inverse([2.0, 5.0])
        Zygote.gradient(x -> sum(my_inverse(x)), [2.0, 5.0])

        my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
        Zygote.gradient(x -> sum(my_second(x)), [2.0, 5.0])
    end
end


@safetestset "GENNSurrogate" begin
    using Surrogates
    using Flux
    using Flux.Optimisers
    using LinearAlgebra

    @testset "1D without gradients" begin
        lb = 0.0
        ub = 10.0
        f = x -> 2 * x + 3
        x = sample(10, lb, ub, SobolSample())
        y = f.(x)
        genn = GENNSurrogate(x, y, lb, ub, n_epochs = 100)
        val = genn(5.0)
        @test val isa Number
        @test isapprox(val, f(5.0), atol = 1.0)
    end

    @testset "1D with gradients" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(10, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        genn = GENNSurrogate(x, y, lb, ub, dydx = dydx, n_epochs = 500)
        val = genn(5.0)
        @test val isa Number
        @test isapprox(val, f(5.0), atol = 2.0)
        
        # Test derivative prediction
        grad_pred = predict_derivative(genn, [5.0])
        @test grad_pred isa Vector
        @test length(grad_pred) == 1
        @test isapprox(grad_pred[1], df(5.0), atol = 1.0)
    end

    @testset "1D update" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        genn = GENNSurrogate(x, y, lb, ub, n_epochs = 50)
        update!(genn, [8.5], [72.25])
        update!(genn, [3.2, 3.5], [10.24, 12.25])
        val = genn(5.0)
        @test val isa Number
    end

    @testset "1D update with gradients" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        genn = GENNSurrogate(x, y, lb, ub, dydx = dydx, n_epochs = 50)
        update!(genn, [8.5], [72.25], dydx_new = reshape([17.0], 1, 1))
        val = genn(5.0)
        @test val isa Number
    end

    @testset "ND without gradients" begin
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        f = x -> x[1] * x[2]
        x = sample(10, lb, ub, SobolSample())
        y = f.(x)
        genn = GENNSurrogate(x, y, lb, ub, n_epochs = 100)
        val = genn([3.4, 1.4])
        @test val isa Number
        @test isapprox(val, f([3.4, 1.4]), atol = 2.0)
    end

    @testset "ND with gradients" begin
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        f = x -> x[1] * x[2]
        # Gradient: [x[2], x[1]]
        x = sample(10, lb, ub, SobolSample())
        y = f.(x)
        dydx = reduce(hcat, ([xi[2], xi[1]] for xi in x))'
        genn = GENNSurrogate(x, y, lb, ub, dydx = dydx, n_epochs = 500)
        val = genn([3.4, 1.4])
        @test val isa Number
        @test isapprox(val, f([3.4, 1.4]), atol = 2.0)
        
        # Test derivative prediction
        grad_pred = predict_derivative(genn, [3.4, 1.4])
        @test grad_pred isa Vector
        @test length(grad_pred) == 2
        @test isapprox(grad_pred[1], 1.4, atol = 1.0)
        @test isapprox(grad_pred[2], 3.4, atol = 1.0)
    end

    @testset "Multi-output with gradients" begin
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

        model = Chain(Dense(2, 2))
        genn = GENNSurrogate(x, y, lb, ub; dydx = dydx, model = model, n_epochs = 400, lambda = 0.0)

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
        genn = GENNSurrogate(x, y, lb, ub, n_epochs = 50)
        update!(genn, [[3.5, 1.4]], [4.9])
        update!(genn, [[3.5, 1.4], [1.5, 1.4]], [4.9, 2.1])
        val = genn([3.4, 1.4])
        @test val isa Number
    end

    @testset "Different input formats" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        x = sample(5, lb, ub, SobolSample())
        y = f.(x)
        genn = GENNSurrogate(x, y, lb, ub, n_epochs = 50)
        
        # Test different input formats
        @test genn(5.0) isa Number
        @test genn([5.0]) isa Number
        @test genn((5.0,)) isa Number
    end

    @testset "Custom model and optimizer" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        x = sample(10, lb, ub, SobolSample())
        y = f.(x)
        model = Chain(Dense(1, 8, relu), Dense(8, 1))
        opt = Optimisers.Adam(0.01)
        genn = GENNSurrogate(x, y, lb, ub, model = model, opt = opt, n_epochs = 50)
        val = genn(5.0)
        @test val isa Number
    end

    @testset "With normalization" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        x = sample(10, lb, ub, SobolSample())
        y = f.(x)
        genn = GENNSurrogate(x, y, lb, ub, is_normalize = true, n_epochs = 50)
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
        
        # Test with different gamma values
        genn_low_gamma = GENNSurrogate(x, y, lb, ub, dydx = dydx, gamma = 0.1, n_epochs = 100)
        genn_high_gamma = GENNSurrogate(x, y, lb, ub, dydx = dydx, gamma = 10.0, n_epochs = 100)
        
        val_low = genn_low_gamma(5.0)
        val_high = genn_high_gamma(5.0)
        @test val_low isa Number
        @test val_high isa Number
    end

    @testset "Normalization with gradients" begin
        lb = 0.0
        ub = 10.0
        f = x -> x^2
        df = x -> 2 * x
        x = sample(20, lb, ub, SobolSample())
        y = f.(x)
        dydx = reshape(df.(x), :, 1)
        
        genn = GENNSurrogate(x, y, lb, ub, dydx = dydx, is_normalize = true, n_epochs = 500)
        val = genn(5.0)
        @test val isa Number
        @test isapprox(val, f(5.0), atol = 2.0)
        
        # Test derivative prediction with normalization
        grad_pred = predict_derivative(genn, [5.0])
        @test grad_pred isa Vector
        @test length(grad_pred) == 1
        @test isapprox(grad_pred[1], df(5.0), atol = 2.0)
    end

    @testset "Multi-output update with gradients" begin
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
        
        genn = GENNSurrogate(x, y, lb, ub; dydx = dydx, n_epochs = 200, lambda = 0.0)
        
        x_new = [[1.0, 1.0], [0.5, 0.5]]
        y_new = hcat(f.(x_new)...)
        dydx_new = Array{Float64, 3}(undef, 2, 2, 2)
        dydx_new[:, :, 1] = grad_template
        dydx_new[:, :, 2] = grad_template
        
        update!(genn, x_new, y_new, dydx_new = dydx_new)
        
        val = genn([0.5, 0.25])
        @test val isa AbstractMatrix
        @test length(val) == 2
    end

    @testset "Edge cases" begin
        # Test with minimal data
        lb = 0.0
        ub = 10.0
        x = [[1.0], [2.0]]
        y = [1.0, 4.0]
        genn = GENNSurrogate(x, y, lb, ub, n_epochs = 10)
        @test genn(1.5) isa Number
        
        # Test with lambda = 0 (no regularization)
        x = sample(5, lb, ub, SobolSample())
        y = (x -> x^2).(x)
        genn = GENNSurrogate(x, y, lb, ub, lambda = 0.0, n_epochs = 50)
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
end
