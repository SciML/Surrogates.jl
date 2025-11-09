using Surrogates
using LinearAlgebra
using Zygote
using ForwardDiff
using Test
using GaussianMixtures

@testset "ForwardDiff" begin
    @testset "1D" begin
        lb = 0.0
        ub = 10.0
        n = 1000
        x = sample(n, lb, ub, SobolSample())
        f = x -> x^2
        y = f.(x)

        #Radials
        @testset "Radials" begin
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
            g = x -> ForwardDiff.derivative(my_rad, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        #Kriging
        @testset "Kriging" begin
            my_p = 1.5
            my_krig = Kriging(x, y, lb, ub, p = my_p)
            g = x -> ForwardDiff.derivative(my_krig, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        #Linear Surrogate
        @testset "Linear Surrogate" begin
            my_linear = LinearSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.derivative(my_linear, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        #Inverse distance
        @testset "Inverse Distance" begin
            my_p = 1.4
            my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
            g = x -> ForwardDiff.derivative(my_inverse, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        #Lobachevsky
        @testset "Lobachevsky" begin
            n = 4
            α = 2.4
            my_loba = LobachevskySurrogate(x, y, lb, ub, alpha = α, n = n)
            g = x -> ForwardDiff.derivative(my_loba, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        #Second order polynomial
        @testset "Second Order Polynomial" begin
            my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.derivative(my_second, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        #Wendland
        # @testset "Wendland" begin
        #     my_wend = Wendland(x, y, lb, ub)
        #     g = x -> ForwardDiff.derivative(my_wend, x)
        #     @test g(5.0) isa Number
        #     # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
        #     @test isapprox(g(5.0), 10.0, atol = 1e-1)
        # end

        #GEK
        @testset "GEK" begin
            y1 = y
            der = x -> 2 * x
            y2 = der.(x)
            y_gek = vcat(y1, y2)
            my_gek = GEK(x, y_gek, lb, ub)
            g = x -> ForwardDiff.derivative(my_gek, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        #GEKPLS
        # @testset "GEKPLS" begin
        #     grads = Zygote.gradient.(f, x)
        #     n_comp = 1
        #     delta_x = 0.0001
        #     extra_points = 1
        #     initial_theta = [0.01 for i in 1:n_comp]
        #     my_gekpls = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
        #     g = x -> ForwardDiff.derivative(my_gekpls, x)
        #     @test g(5.0) isa Number
        #     # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
        #     @test isapprox(g(5.0), 10.0, atol = 1e-1)
        # end

        #Earth
        # @testset "Earth" begin
        #     my_earth = EarthSurrogate(x, y, lb, ub)
        #     g = x -> ForwardDiff.derivative(my_earth, x)
        #     @test g(5.0) isa Number
        #     # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
        #     @test isapprox(g(5.0), 10.0, atol = 1e-1)
        # end

        #VariableFidelity
        @testset "VariableFidelity" begin
            my_varfid = VariableFidelitySurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.derivative(my_varfid, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end

        #MOE
        @testset "MOE" begin
            expert_types = [
                RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0, sparse = false),
                RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0, sparse = false)
            ]
            my_moe = MOE(x, y, expert_types, ndim = 1, n_clusters = 2)
            g = x -> ForwardDiff.derivative(my_moe, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1e-1)
        end
    end

    @testset "ND" begin
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        n = 1000
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1] * x[2]
        y = f.(x)

        #Radials
        @testset "Radials" begin
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
            g = x -> ForwardDiff.gradient(my_rad, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #Kriging
        @testset "Kriging" begin
            my_theta = [2.0, 2.0]
            my_p = [1.9, 1.9]
            my_krig = Kriging(x, y, lb, ub, p = my_p, theta = my_theta)
            g = x -> ForwardDiff.gradient(my_krig, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #Linear Surrogate
        @testset "Linear Surrogate" begin
            my_linear = LinearSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.gradient(my_linear, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #Inverse Distance
        @testset "Inverse Distance" begin
            my_p = 1.4
            my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
            g = x -> ForwardDiff.gradient(my_inverse, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #Lobachevsky
        @testset "Lobachevsky" begin
            alpha = [1.4, 1.4]
            n = 4
            my_loba_ND = LobachevskySurrogate(x, y, lb, ub, alpha = alpha, n = n)
            g = x -> ForwardDiff.gradient(my_loba_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #Second order polynomial
        @testset "SecondOrderPolynomialSurrogate" begin
            my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.gradient(my_second, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #Wendland
        @testset "Wendland" begin
            my_wend_ND = Wendland(x, y, lb, ub)
            g = x -> ForwardDiff.gradient(my_wend_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #GEK
        @testset "GEK" begin
            y1 = y
            der = x -> [x[2], x[1]]  # Gradient of f(x) = x[1] * x[2]
            y2 = vcat([der(xi) for xi in x]...)  # Flatten gradients by point
            y_gek = vcat(y1, y2)
            my_gek = GEK(x, y_gek, lb, ub)
            g = x -> ForwardDiff.gradient(my_gek, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #GEKPLS
        @testset "GEKPLS" begin
            grads = Zygote.gradient.(f, x)
            n_comp = 2
            delta_x = 0.0001
            extra_points = 2
            initial_theta = [0.01 for i in 1:n_comp]
            my_gekpls_ND = GEKPLS(
                x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
            g = x -> ForwardDiff.gradient(my_gekpls_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #Earth
        # @testset "Earth" begin
        #     my_earth_ND = EarthSurrogate(x, y, lb, ub)
        #     g = x -> ForwardDiff.gradient(my_earth_ND, x)
        #     @test g([2.0, 5.0]) isa AbstractVector
        #     # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
        #     @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        # end

        #VariableFidelity
        @testset "VariableFidelity" begin
            my_varfid_ND = VariableFidelitySurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.gradient(my_varfid_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
        end

        #MOE
        @testset "MOE" begin
            expert_types = [
                RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0, sparse = false),
                RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0, sparse = false)
            ]
            my_moe_ND = MOE(x, y, expert_types, ndim = 2, n_clusters = 2)
            g = x -> ForwardDiff.gradient(my_moe_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1e-1)
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

        #Radials
        @testset "Radials" begin
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial())
            g = x -> Zygote.gradient(my_rad, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        #Kriging
        @testset "Kriging" begin
            my_p = 1.5
            my_krig = Kriging(x, y, lb, ub, p = my_p)
            g = x -> Zygote.gradient(my_krig, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        #Linear Surrogate
        # @testset "Linear Surrogate" begin
        #     my_linear = LinearSurrogate(x, y, lb, ub)
        #     g = x -> Zygote.gradient(my_linear, x)
        #     result = g(5.0)
        #     @test result isa Tuple
        #     @test length(result) == 1
        #     @test result[1] isa Number
        # end

        #Inverse distance
        @testset "Inverse Distance" begin
            my_p = 1.4
            my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
            g = x -> Zygote.gradient(my_inverse, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        #Lobachevsky
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
            @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        #Second order polynomial
        @testset "Second Order Polynomial" begin
            my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_second, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        #Wendland
        # @testset "Wendland" begin
        #     my_wend = Wendland(x, y, lb, ub)
        #     g = x -> Zygote.gradient(my_wend, x)
        #     result = g(3.0)
        #     @test result isa Tuple
        #     @test length(result) == 1
        #     @test result[1] isa Number
        # end

        #GEK
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
            @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        #GEKPLS
        # @testset "GEKPLS" begin
        #     grads = Zygote.gradient.(f, x)
        #     n_comp = 2
        #     delta_x = 0.0001
        #     extra_points = 2
        #     initial_theta = [0.01 for i in 1:n_comp]
        #     my_gekpls = GEKPLS(x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
        #     g = x -> Zygote.gradient(my_gekpls, x)
        #     result = g(5.0)
        #     @test result isa Tuple
        #     @test length(result) == 1
        #     @test result[1] isa Number
        #     # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
        #     @test isapprox(result[1], 10.0, atol = 1e-1)
        # end

        #Earth
        @testset "Earth" begin
            my_earth = EarthSurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_earth, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        #VariableFidelity
        @testset "VariableFidelity" begin
            my_varfid = VariableFidelitySurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_varfid, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1e-1)
        end

        #MOE
        @testset "MOE" begin
            expert_types = [
                RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0, sparse = false),
                RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0, sparse = false)
            ]
            my_moe = MOE(x, y, expert_types, ndim = 1, n_clusters = 2)
            g = x -> Zygote.gradient(my_moe, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1e-1)
        end
    end

    @testset "ND" begin
        lb = [0.0, 0.0]
        ub = [10.0, 10.0]
        n = 1000
        x = sample(n, lb, ub, SobolSample())
        f = x -> x[1] * x[2]
        y = f.(x)

        #Radials
        @testset "Radials" begin
            my_rad = RadialBasis(x, y, lb, ub, rad = linearRadial(), scale_factor = 2.1)
            g = x -> Zygote.gradient(my_rad, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        #Kriging
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
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        #Linear Surrogate
        @testset "Linear Surrogate" begin
            my_linear = LinearSurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_linear, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        #Inverse Distance
        @testset "Inverse Distance" begin
            my_p = 1.4
            my_inverse = InverseDistanceSurrogate(x, y, lb, ub, p = my_p)
            g = x -> Zygote.gradient(my_inverse, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        #Lobachevsky
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
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        #Second order polynomial
        @testset "SecondOrderPolynomialSurrogate" begin
            my_second = SecondOrderPolynomialSurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_second, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        #Wendland
        # @testset "Wendland" begin
        #     my_wend_ND = Wendland(x, y, lb, ub)
        #     g = x -> Zygote.gradient(my_wend_ND, x)
        #     result = g((2.0, 5.0))
        #     @test result isa Tuple
        #     @test length(result) == 1
        #     @test result[1] isa Tuple
        # end

        #GEK
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
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        #GEKPLS
        @testset "GEKPLS" begin
            grads = Zygote.gradient.(f, x)
            n_comp = 2
            delta_x = 0.0001
            extra_points = 2
            initial_theta = [0.01 for i in 1:n_comp]
            my_gekpls_ND = GEKPLS(
                x, y, grads, n_comp, delta_x, lb, ub, extra_points, initial_theta)
            g = x -> Zygote.gradient(my_gekpls_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        #Earth
        # @testset "Earth" begin
        #     my_earth_ND = EarthSurrogate(x, y, lb, ub)
        #     g = x -> Zygote.gradient(my_earth_ND, x)
        #     result = g((2.0, 5.0))
        #     @test result isa Tuple
        #     @test length(result) == 1
        #     @test result[1] isa Tuple
        # end

        #VariableFidelity
        @testset "VariableFidelity" begin
            my_varfid_ND = VariableFidelitySurrogate(x, y, lb, ub)
            g = x -> Zygote.gradient(my_varfid_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end

        #MOE
        @testset "MOE" begin
            expert_types = [
                RadialBasisStructure(radial_function = linearRadial(), scale_factor = 1.0, sparse = false),
                RadialBasisStructure(radial_function = cubicRadial(), scale_factor = 1.0, sparse = false)
            ]
            my_moe_ND = MOE(x, y, expert_types, ndim = 2, n_clusters = 2)
            g = x -> Zygote.gradient(my_moe_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1e-1))
        end
    end
end
