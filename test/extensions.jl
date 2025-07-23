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
        agp1D = AbstractGPSurrogate([x_points[1]], [y_points[1]],
            gp = GP(SqExponentialKernel()), Σy = 0.05)
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
        @test agpND(x_new)≈f(x_new) atol=0.2
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
            @test g(x_val)[1]≈2 * x_val rtol=1e-1
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
            @test g_val[1]≈x_val[2] rtol=1e-1
            @test g_val[2]≈x_val[1] rtol=1e-1
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
        my_neural = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1)
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
        surrogate = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1)

        f = x -> [x[1], x[2]^2]
        lb = [1.0, 2.0]
        ub = [10.0, 8.5]
        x = sample(20, lb, ub, SobolSample())
        y = f.(x)
        my_model = Chain(Dense(2, 2))
        my_opt = Descent(0.01)
        surrogate = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1)
        surrogate([1.0, 2.0])
        x_new = [[2.0, 2.0]]
        y_new = [f(x_new[1])]
        update!(surrogate, x_new, y_new)
    end

    @testset "Optimization" begin
        lb = [1.0, 1.0]
        ub = [6.0, 6.0]
        x = sample(5, lb, ub, SobolSample())
        objective_function_ND = z -> 3 * norm(z) + 1
        y = objective_function_ND.(x)
        model = Chain(Dense(2, 1), first)
        opt = Descent(0.01)
        my_neural_ND_neural = NeuralSurrogate(x, y, lb, ub, model = model, loss = Flux.mse)
        surrogate_optimize!(objective_function_ND, SRBF(), lb, ub, my_neural_ND_neural,
            SobolSample(), maxiters = 15)
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
        my_neural = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1)
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
        my_neural = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1)
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
        my_neural = NeuralSurrogate(x, y, lb, ub, model = my_model, loss = Flux.mse,
            opt = my_opt, n_epochs = 1)
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