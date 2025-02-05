using SafeTestsets, Test

@safetestset "PolynomialChaosSurrogates" begin
    using Surrogates
    using PolyChaos
    using SurrogatesPolyChaos
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
            x, y, lb, ub; orthopolys = Uniform01OrthoPoly(1))
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
