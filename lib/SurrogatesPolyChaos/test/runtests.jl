using SafeTestsets

@safetestset "PolynomialChaosSurrogates" begin
    using Surrogates
    using PolyChaos
    using SurrogatesPolyChaos
    using Zygote

    @testset "1D" begin
        n = 20
        lb = [0.0]
        ub = [4.0]
        f = x -> 2 * x
        x = sample(n, lb, ub, SobolSample())
        y = f.(x)
        my_pce = PolynomialChaosSurrogate(x, y, lb, ub)
        val = my_pce(2.0)
        update!(my_pce, [3.0], [6.0])
        my_pce_changed = PolynomialChaosSurrogate(x, y, lb, ub; op = Uniform01OrthoPoly(1))
    end

    @testset "ND" begin
        n = 60
        lb = [0.0, 0.0]
        ub = [5.0, 5.0]
        f = x -> x[1] * x[2]
        x = collect.(sample(n, lb, ub, SobolSample()))
        y = f.(x)

        my_pce = PolynomialChaosSurrogate(x, y, lb, ub)
        val = my_pce([2.0, 2.0])
        update!(my_pce, [[2.0, 3.0]], [6.0])

        op1 = Uniform01OrthoPoly(1)
        op2 = Beta01OrthoPoly(2, 2, 1.2)
        ops = [op1, op2]
        multi_poly = MultiOrthoPoly(ops, min(1, 2))
        my_pce_changed = PolynomialChaosSurrogate(x, y, lb, ub, op = multi_poly)
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
        g(3.0)
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
        g([1.0, 1.0])
    end
end
