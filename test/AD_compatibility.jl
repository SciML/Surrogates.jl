using Surrogates
using LinearAlgebra
using Zygote
using ForwardDiff
using Test
using GaussianMixtures
using Flux
using Random

Random.seed!(42)

@testset "ForwardDiff" begin
    @testset "1D" begin
        lb = 0.0
        ub = 10.0
        n = 1000
        x = sample(n, lb, ub, SobolSample())
        f = x -> x^2
        y = f.(x)

        @testset "Radials" begin
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
            g = x -> ForwardDiff.derivative(my_rad, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1.0e-1)
        end

        @testset "Kriging" begin
            my_p = 1.5
            my_krig = Kriging(x, y, lb, ub, p = my_p)
            g = x -> ForwardDiff.derivative(my_krig, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1.0e-1)
        end

        @testset "Linear Surrogate" begin
            my_linear = LinearSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.derivative(my_linear, x)
            @test g(5.0) isa Number
            # Affine model: the derivative is the fitted slope, everywhere.
            @test g(1.0) == g(9.0)
            @test g(5.0) == my_linear.coeff[2]
            # The slope is 2 * mean(x) -> f'(5.0) = 10; needs the intercept.
            @test isapprox(g(5.0), 10.0, atol = 1.0e-2)

            # Vector responses: the derivative is the row of slopes.
            y_multi = [[t, t^2] for t in x]
            my_linear_multi = LinearSurrogate(x, y_multi, lb, ub)
            gm = t -> ForwardDiff.derivative(my_linear_multi, t)
            @test gm(5.0) isa AbstractVector
            @test gm(5.0) ≈ my_linear_multi.coeff[2, :]
            @test gm(1.0) == gm(9.0)
        end

        @testset "Inverse Distance" begin
            my_p = 1.4
            my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
            g = x -> ForwardDiff.derivative(my_inverse, x)
            @test g(5.0) isa Number
            # Shepard is stationary at every sample point, so its derivative
            # does not approximate f'. Checked against central differences.
            h = 1.0e-6
            @test isapprox(g(5.0), (my_inverse(5.0 + h) - my_inverse(5.0 - h)) / 2h,
                atol = 1.0e-4)
            # On a sample point the weight is non-finite; the derivative
            # must still come out finite.
            @test g(x[3]) == 0.0

            # Vector responses differentiate componentwise.
            y_multi = [[t, t^2] for t in x]
            my_inverse_multi = InverseDistanceSurrogate(x, y_multi, lb, ub, p = my_p)
            gm = ForwardDiff.derivative(my_inverse_multi, 5.0)
            @test gm isa AbstractVector
            @test length(gm) == 2
            @test all(isfinite, gm)
        end

        @testset "Lobachevsky" begin
            n = 4
            α = 2.4
            my_loba = LobachevskySurrogate(x, y, lb, ub, alpha = α, n = n)
            g = x -> ForwardDiff.derivative(my_loba, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1.0e-1)
        end

        @testset "Second Order Polynomial" begin
            my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.derivative(my_second, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1.0e-1)
        end

        @testset "Wendland" begin
            # maxiters = 5000: the default 300 leaves the solve unconverged on
            # this many samples, which now warns and fits poorly.
            my_wend = Wendland(x, y, lb, ub, maxiters = 5000)
            g = x -> ForwardDiff.derivative(my_wend, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1.0)
            h = 1.0e-6
            @test isapprox(g(5.0), (my_wend(5.0 + h) - my_wend(5.0 - h)) / 2h,
                atol = 1.0e-4)
        end

        @testset "GEK" begin
            y1 = y
            der = x -> 2 * x
            y2 = der.(x)
            y_gek = vcat(y1, y2)
            my_gek = GEK(x, y_gek, lb, ub)
            g = x -> ForwardDiff.derivative(my_gek, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            # @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        @testset "GEKPLS" begin
            grads = Zygote.gradient.(f, x)
            n_comp = 1
            delta_x = 0.0001
            extra_points = 1
            initial_theta = [0.01 for i in 1:n_comp]
            my_gekpls = GEKPLS(
                x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta
            )
            g = x -> ForwardDiff.derivative(my_gekpls, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1.0e-1)
        end

        @testset "Earth" begin
            my_earth = EarthSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.derivative(my_earth, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            # @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        @testset "VariableFidelity" begin
            my_varfid = VariableFidelitySurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.derivative(my_varfid, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1.0e-1)
        end

        @testset "MOE" begin
            expert_types = [
                RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0, sparse = false),
                RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0, sparse = false),
            ]
            my_moe = MOE(x, y, expert_types, ndim = 1, n_clusters = 2)
            g = x -> ForwardDiff.derivative(my_moe, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1.0e-1)
        end

        @testset "GENN" begin
            df = x -> 2 * x
            dydx = reshape(df.(x), :, 1)
            my_genn = GENNSurrogate(x[1:200], y[1:200], lb, ub, dydx[1:200, :], n_epochs = 500)
            g = x -> ForwardDiff.derivative(my_genn, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 2.0)
        end
    end

    @testset "ND" begin
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        n = 1000
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1] * x[2]
        y = f.(x)

        @testset "Radials" begin
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
            g = x -> ForwardDiff.gradient(my_rad, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0e-1)
        end

        @testset "Kriging" begin
            my_theta = [2.0, 2.0]
            my_p = [1.9, 1.9]
            my_krig = Kriging(x, y, lb, ub, p = my_p, theta = my_theta)
            g = x -> ForwardDiff.gradient(my_krig, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0e-1)
        end

        @testset "Linear Surrogate" begin
            my_linear = LinearSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.gradient(my_linear, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # The fit has constant slopes near [5, 5], not the pointwise
            # ∇f([2, 5]) = [5, 2]; the gradient is those slopes, everywhere.
            @test g([2.0, 5.0]) ≈ my_linear.coeff[2:end]
            @test g([2.0, 5.0]) == g([9.0, 0.5])

            # Vector responses: the Jacobian is the transposed slope block,
            # the same matrix at any point.
            y_multi = [[p[1] * p[2], p[1] + p[2]] for p in x]
            my_linear_multi = LinearSurrogate(x, y_multi, lb, ub)
            J = ForwardDiff.jacobian(my_linear_multi, [2.0, 5.0])
            @test size(J) == (2, 2)
            @test J ≈ permutedims(my_linear_multi.coeff[2:end, :])
            @test J == ForwardDiff.jacobian(my_linear_multi, [9.0, 0.5])
        end

        @testset "Inverse Distance" begin
            my_p = 1.4
            my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
            g = x -> ForwardDiff.gradient(my_inverse, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # As in 1D, checked against central differences, not ∇f.
            h = 1.0e-6
            cd = [(my_inverse([2.0, 5.0] + h * e) - my_inverse([2.0, 5.0] - h * e)) / 2h
                  for e in ([1.0, 0.0], [0.0, 1.0])]
            @test isapprox(g([2.0, 5.0]), cd, atol = 1.0e-4)
            # `norm` of a zero vector of duals is NaN, so this goes through
            # the coincidence branch.
            @test g(collect(x[3])) == [0.0, 0.0]
        end

        @testset "Lobachevsky" begin
            alpha = [1.4, 1.4]
            n = 4
            my_loba_ND = LobachevskySurrogate(x, y, lb, ub, alpha = alpha, n = n)
            g = x -> ForwardDiff.gradient(my_loba_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0e-1)
        end

        @testset "SecondOrderPolynomialSurrogate" begin
            my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.gradient(my_second, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0e-1)
        end

        @testset "Wendland" begin
            my_wend_ND = Wendland(x, y, lb, ub)
            g = x -> ForwardDiff.gradient(my_wend_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0)
            h = 1.0e-6
            cd = [(my_wend_ND([2.0, 5.0] + h * e) - my_wend_ND([2.0, 5.0] - h * e)) / 2h
                  for e in ([1.0, 0.0], [0.0, 1.0])]
            @test isapprox(g([2.0, 5.0]), cd, atol = 1.0e-4)

            # At a sample point `norm` of the zero difference is NaN under AD.
            # The kernel peaks there, so its value has to survive too: a
            # composed objective feeds that value through the chain rule.
            node = collect(x[3])
            @test all(isfinite, g(node))
            obj = q -> my_wend_ND(q)^2
            cdn = [(obj(node + h * e) - obj(node - h * e)) / 2h
                   for e in ([1.0, 0.0], [0.0, 1.0])]
            @test isapprox(ForwardDiff.gradient(obj, node), cdn, atol = 1.0e-4)
        end

        @testset "GEK" begin
            y1 = y
            der = x -> [x[2], x[1]]  # Gradient of f(x) = x[1] * x[2]
            y2 = vcat([der(xi) for xi in x]...)  # Flatten gradients by point
            y_gek = vcat(y1, y2)
            my_gek = GEK(x, y_gek, lb, ub)
            g = x -> ForwardDiff.gradient(my_gek, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            # @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        @testset "GEKPLS" begin
            grads = Zygote.gradient.(f, x)
            n_comp = 2
            delta_x = 0.0001
            extra_points = 2
            initial_theta = [0.01 for i in 1:n_comp]
            my_gekpls_ND = GEKPLS(
                x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta
            )
            g = x -> ForwardDiff.gradient(my_gekpls_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0e-1)
        end

        @testset "GENN" begin
            der = x -> [x[2], x[1]]  # Gradient of f(x) = x[1] * x[2]
            dydx = reduce(hcat, (der(xi) for xi in x))'  # (n_samples, n_inputs)
            my_genn_ND = GENNSurrogate(x[1:200], y[1:200], lb, ub, dydx[1:200, :], n_epochs = 500)
            g = x -> ForwardDiff.gradient(my_genn_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 2.0)
        end

        @testset "Earth" begin
            my_earth_ND = EarthSurrogate(x[1:10], y[1:10], lb, ub)
            g = x -> ForwardDiff.gradient(my_earth_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            # @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        @testset "VariableFidelity" begin
            my_varfid_ND = VariableFidelitySurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.gradient(my_varfid_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0e-1)
        end

        @testset "MOE" begin
            expert_types = [
                RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0, sparse = false),
                RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0, sparse = false),
            ]
            my_moe_ND = MOE(x, y, expert_types, ndim = 2, n_clusters = 2)
            g = x -> ForwardDiff.gradient(my_moe_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0e-1)
        end
    end

end

@testset "Zygote" begin
    @testset "1D" begin
        lb = 0.0
        ub = 10.0
        n = 1000
        x = sample(n, lb, ub, SobolSample())
        f = x -> x^2
        y = f.(x)

        @testset "Radials" begin
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
            g = x -> Zygote.gradient(my_rad, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1.0e-1)
        end

        @testset "Kriging" begin
            my_p = 1.5
            my_krig = Kriging(x, y, lb, ub, p = my_p)
            g = x -> Zygote.gradient(my_krig, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1.0e-1)
        end

        @testset "Linear Surrogate" begin
            my_linear = LinearSurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_linear, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Same slope everywhere, and agreeing with forward mode.
            @test result[1] ≈ my_linear.coeff[2]
            @test result[1] == g(9.0)[1]
            @test result[1] ≈ ForwardDiff.derivative(my_linear, 5.0)

            # Vector responses go through Zygote.jacobian.
            y_multi = [[t, t^2] for t in x]
            my_linear_multi = LinearSurrogate(x, y_multi, lb, ub)
            @test Zygote.jacobian(my_linear_multi, 5.0)[1] ≈ my_linear_multi.coeff[2, :]
        end

        @testset "Inverse Distance" begin
            my_p = 1.4
            my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
            g = x -> Zygote.gradient(my_inverse, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Reverse mode has to agree with forward mode.
            @test result[1] ≈ ForwardDiff.derivative(my_inverse, 5.0)
        end

        @testset "Lobachevsky" begin
            n = 4
            α = 2.4
            my_loba = LobachevskySurrogate(x, y, lb, ub, alpha = α, n = 4)
            g = x -> Zygote.gradient(my_loba, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1.0e-1)
        end

        @testset "Second Order Polynomial" begin
            my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_second, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1.0e-1)
        end

        @testset "Wendland" begin
            my_wend = Wendland(x, y, lb, ub, maxiters = 5000)
            g = x -> Zygote.gradient(my_wend, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1.0)
            # Reverse mode has to agree with forward mode.
            @test result[1] ≈ ForwardDiff.derivative(my_wend, 5.0)
        end

        @testset "GEK" begin
            y1 = y
            der = x -> 2 * x
            y2 = der.(x)
            y_gek = vcat(y1, y2)
            my_gek = GEK(x, y_gek, lb, ub)
            g = x -> Zygote.gradient(my_gek, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            # @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        @testset "GEKPLS" begin
            grads = Zygote.gradient.(f, x)
            n_comp = 2
            delta_x = 0.0001
            extra_points = 2
            initial_theta = [0.01 for i in 1:n_comp]
            my_gekpls = GEKPLS(
                x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta
            )
            g = x -> Zygote.gradient(my_gekpls, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1.0e-1)
        end

        @testset "GENN" begin
            df = x -> 2 * x
            dydx = reshape(df.(x), :, 1)
            my_genn = GENNSurrogate(x[1:200], y[1:200], lb, ub, dydx[1:200, :], n_epochs = 500)
            g = x -> Zygote.gradient(my_genn, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 2.0)
        end

        @testset "Earth" begin
            my_earth = EarthSurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_earth, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            # @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        @testset "VariableFidelity" begin
            my_varfid = VariableFidelitySurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_varfid, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1.0e-1)
        end

        @testset "MOE" begin
            expert_types = [
                RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0, sparse = false),
                RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0, sparse = false),
            ]
            my_moe = MOE(x, y, expert_types, ndim = 1, n_clusters = 2)
            g = x -> Zygote.gradient(my_moe, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1.0e-1)
        end
    end

    @testset "ND" begin
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        n = 1000
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1] * x[2]
        y = f.(x)

        @testset "Radials" begin
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), scale_factor = 2.1)
            g = x -> Zygote.gradient(my_rad, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0e-1))
        end

        @testset "Kriging" begin
            my_theta = [2.0, 2.0]
            my_p = [1.9, 1.9]
            my_krig = Kriging(x, y, lb, ub, p = my_p, theta = my_theta)
            g = x -> Zygote.gradient(my_krig, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0e-1))
        end

        @testset "Linear Surrogate" begin
            my_linear = LinearSurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_linear, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # The fitted slopes rather than the true ∇f, constant in x.
            @test all(isapprox.(result[1], Tuple(my_linear.coeff[2:end])))
            @test all(result[1] .== g((9.0, 0.5))[1])
            @test all(isapprox.(result[1],
                Tuple(ForwardDiff.gradient(my_linear, [2.0, 5.0]))))

            # Vector responses on vector points.
            x_vec = [collect(p) for p in x]
            y_multi = [[p[1] * p[2], p[1] + p[2]] for p in x]
            my_linear_multi = LinearSurrogate(x_vec, y_multi, lb, ub)
            J = Zygote.jacobian(my_linear_multi, [2.0, 5.0])[1]
            @test size(J) == (2, 2)
            @test J ≈ permutedims(my_linear_multi.coeff[2:end, :])
        end

        @testset "Inverse Distance" begin
            my_p = 1.4
            my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
            g = x -> Zygote.gradient(my_inverse, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            @test all(isapprox.(
                result[1], Tuple(ForwardDiff.gradient(my_inverse, [2.0, 5.0]))))
        end

        @testset "Lobachevsky" begin
            alpha = [1.4, 1.4]
            n = 4
            my_loba_ND = LobachevskySurrogate(x, y, lb, ub, alpha = alpha, n = n)
            g = x -> Zygote.gradient(my_loba_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0e-1))
        end

        @testset "SecondOrderPolynomialSurrogate" begin
            my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_second, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0e-1))
        end

        @testset "Wendland" begin
            my_wend_ND = Wendland(x, y, lb, ub)
            g = x -> Zygote.gradient(my_wend_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0))
            @test all(isapprox.(
                result[1], Tuple(ForwardDiff.gradient(my_wend_ND, [2.0, 5.0]))))
        end

        @testset "GEK" begin
            y1 = y
            der = x -> [x[2], x[1]]  # Gradient of f(x) = x[1] * x[2]
            y2 = vcat([der(xi) for xi in x]...)  # Flatten gradients by point
            y_gek = vcat(y1, y2)
            my_gek = GEK(x, y_gek, lb, ub)
            g = x -> Zygote.gradient(my_gek, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            # @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        @testset "GEKPLS" begin
            grads = Zygote.gradient.(f, x)
            n_comp = 2
            delta_x = 0.0001
            extra_points = 2
            initial_theta = [0.01 for i in 1:n_comp]
            my_gekpls_ND = GEKPLS(
                x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta
            )
            g = x -> Zygote.gradient(my_gekpls_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0e-1))
        end

        @testset "GENN" begin
            der = x -> [x[2], x[1]]  # Gradient of f(x) = x[1] * x[2]
            dydx = reduce(hcat, (der(xi) for xi in x))'  # (n_samples, n_inputs)
            my_genn_ND = GENNSurrogate(x[1:200], y[1:200], lb, ub, dydx[1:200, :], n_epochs = 500)
            g = x -> Zygote.gradient(my_genn_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 2.0))
        end

        @testset "Earth" begin
            my_earth_ND = EarthSurrogate(x[1:10], y[1:10], lb, ub)
            g = x -> Zygote.gradient(my_earth_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            # @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        @testset "VariableFidelity" begin
            my_varfid_ND = VariableFidelitySurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_varfid_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0e-1))
        end

        @testset "MOE" begin
            expert_types = [
                RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0, sparse = false),
                RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0, sparse = false),
            ]
            my_moe_ND = MOE(x, y, expert_types, ndim = 2, n_clusters = 2)
            g = x -> Zygote.gradient(my_moe_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0e-1))
        end
    end
end
