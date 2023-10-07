using SafeTestsets

@safetestset "PolynomialChaosSurrogates" begin
    using Surrogates
    using PolyChaos
    using Surrogates: sample, SobolSample
    using SurrogatesPolyChaos
    using Zygote

    #1D
    n = 20
    lb = 0.0
    ub = 4.0
    f = x -> 2 * x
    x = sample(n, lb, ub, SobolSample())
    y = f.(x)

    my_pce = PolynomialChaosSurrogate(x, y, lb, ub)
    val = my_pce(2.0)
    add_point!(my_pce, 3.0, 6.0)
    my_pce_changed = PolynomialChaosSurrogate(x, y, lb, ub, op = Uniform01OrthoPoly(1))

    #ND
    n = 60
    lb = [0.0, 0.0]
    ub = [5.0, 5.0]
    f = x -> x[1] * x[2]
    x = sample(n, lb, ub, SobolSample())
    y = f.(x)

    my_pce = PolynomialChaosSurrogate(x, y, lb, ub)
    val = my_pce((2.0, 2.0))
    add_point!(my_pce, (2.0, 3.0), 6.0)

    op1 = Uniform01OrthoPoly(1)
    op2 = Beta01OrthoPoly(2, 2, 1.2)
    ops = [op1, op2]
    multi_poly = MultiOrthoPoly(ops, min(1, 2))
    my_pce_changed = PolynomialChaosSurrogate(x, y, lb, ub, op = multi_poly)

    # Surrogate optimization test
    lb = 0.0
    ub = 15.0
    p = 1.99
    a = 2
    b = 6
    objective_function = x -> 2 * x + 1
    x = sample(20, lb, ub, SobolSample())
    y = objective_function.(x)
    my_poly1d = PolynomialChaosSurrogate(x, y, lb, ub)
    @test_broken surrogate_optimize(objective_function, SRBF(), a, b, my_poly1d,
                                    LowDiscrepancySample(; base = 2))

    lb = [0.0, 0.0]
    ub = [10.0, 10.0]
    obj_ND = x -> log(x[1]) * exp(x[2])
    x = sample(40, lb, ub, RandomSample())
    y = obj_ND.(x)
    my_polyND = PolynomialChaosSurrogate(x, y, lb, ub)
    surrogate_optimize(obj_ND, SRBF(), lb, ub, my_polyND, SobolSample(), maxiters = 15)

    # AD Compatibility

    lb = 0.0
    ub = 3.0
    n = 10
    x = sample(n, lb, ub, SobolSample())
    f = x -> x^2
    y = f.(x)

    # #Polynomialchaos
    @testset "Polynomial Chaos" begin
        f = x -> x^2
        n = 50
        x = sample(n, lb, ub, SobolSample())
        y = f.(x)
        my_poli = PolynomialChaosSurrogate(x, y, lb, ub)
        g = x -> my_poli'(x)
        g(3.0)
    end

    # #PolynomialChaos
    @testset "Polynomial Chaos ND" begin
        n = 50
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1] * x[2]
        y = f.(x)
        my_poli_ND = PolynomialChaosSurrogate(x, y, lb, ub)
        g = x -> Zygote.gradient(my_poli_ND, x)
        g((1.0, 1.0))

        n = 10
        d = 2
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1]^2 + x[2]^2
        y1 = f.(x)
        grad1 = x -> 2 * x[1]
        grad2 = x -> 2 * x[2]
        function create_grads(n, d, grad1, grad2, y)
            c = 0
            y2 = zeros(eltype(y[1]), n * d)
            for i in 1:n
                y2[i + c] = grad1(x[i])
                y2[i + c + 1] = grad2(x[i])
                c = c + 1
            end
            return y2
        end
        y2 = create_grads(n, d, grad1, grad2, y)
        y = vcat(y1, y2)
        my_gek_ND = GEK(x, y, lb, ub)
        g = x -> Zygote.gradient(my_gek_ND, x)
        @test_broken g((2.0, 5.0)) #breaks after Zygote version 0.6.43
    end
end
