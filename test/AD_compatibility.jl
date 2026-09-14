using Surrogates
using LinearAlgebra
using Zygote
using ForwardDiff
using Test
using GaussianMixtures
using Flux
using AbstractGPs
using PolyChaos
using LIBSVM
using Random
import XGBoost

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
            @test isapprox(
                g(5.0), (my_inverse(5.0 + h) - my_inverse(5.0 - h)) / 2h,
                atol = 1.0e-4
            )
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
            h = 1.0e-6
            @test isapprox(
                g(5.0), (my_loba(5.0 + h) - my_loba(5.0 - h)) / 2h,
                atol = 1.0e-4
            )
            # The kernel is a truncated power, so a derivative on a sample
            # point still has to come out finite.
            @test isfinite(g(x[3]))
        end

        @testset "Lobachevsky multi-output" begin
            xm = sample(60, lb, ub, SobolSample())
            ym = (t -> [t^2, sin(t)]).(xm)
            my_loba_multi = LobachevskySurrogate(xm, ym, lb, ub, alpha = 2.4, n = 4)
            J = ForwardDiff.jacobian(t -> my_loba_multi(t[1]), [5.0])
            @test size(J) == (2, 1)
            # d/dx x^2 = 2x and d/dx sin(x) = cos(x) at x = 5.0
            @test isapprox(J[1], 10.0, atol = 1.0e-1)
            @test isapprox(J[2], cos(5.0), atol = 1.0e-1)
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
            @test isapprox(
                g(5.0), (my_wend(5.0 + h) - my_wend(5.0 - h)) / 2h,
                atol = 1.0e-4
            )
        end

        @testset "GEK" begin
            y1 = y
            der = x -> 2 * x
            y2 = der.(x)
            y_gek = vcat(y1, y2)
            my_gek = GEK(x, y_gek, lb, ub; optimize_theta = false)
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
            # `std_error_at_point` used to reconstruct a `Cholesky` from a stored
            # plain matrix on every call, which ForwardDiff tolerates but Zygote
            # cannot differentiate through; see the "Zygote" testset below.
            # Checked on a much sparser fit than `my_gekpls`: at this design's
            # n=1000 density over [0,10], the predictive variance sits at its
            # numerical floor (~1e-16) everywhere, so its derivative is noise,
            # not signal — `isfinite` there is a coin flip across BLAS/LAPACK
            # implementations, not a real regression guard.
            se_x = sample(5, lb, ub, SobolSample())
            se_gekpls = GEKPLS(
                se_x, f.(se_x), Zygote.gradient.(f, se_x), n_comp, delta_x, lb, ub,
                extra_points, initial_theta
            )
            se = x -> ForwardDiff.derivative(t -> std_error_at_point(se_gekpls, t), x)
            @test se(5.0) isa Number && isfinite(se(5.0))
        end

        @testset "KPLS" begin
            my_kpls = KPLS(x, y, 1, [lb], [ub], [1.0]; optimize_theta = false)
            g = x -> ForwardDiff.derivative(my_kpls, x)
            @test g(5.0) isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(g(5.0), 10.0, atol = 1.0)
            # See the GEKPLS testset above for why this uses a sparser fit.
            se_x = sample(5, lb, ub, SobolSample())
            se_kpls = KPLS(se_x, f.(se_x), 1, [lb], [ub], [1.0]; optimize_theta = false)
            se = x -> ForwardDiff.derivative(t -> std_error_at_point(se_kpls, t), x)
            @test se(5.0) isa Number && isfinite(se(5.0))
        end

        @testset "KPLSK" begin
            my_kplsk = KPLSK(x, y, 1, [lb], [ub], [1.0]; optimize_theta = false)
            g = x -> ForwardDiff.derivative(my_kplsk, x)
            @test g(5.0) isa Number
            @test isapprox(g(5.0), 10.0, atol = 1.0)
            # See the GEKPLS testset above for why this uses a sparser fit.
            se_x = sample(5, lb, ub, SobolSample())
            se_kplsk = KPLSK(se_x, f.(se_x), 1, [lb], [ub], [1.0]; optimize_theta = false)
            se = x -> ForwardDiff.derivative(t -> std_error_at_point(se_kplsk, t), x)
            @test se(5.0) isa Number && isfinite(se(5.0))
        end

        @testset "Earth" begin
            my_earth = EarthSurrogate(x, y, lb, ub)
            g = x -> ForwardDiff.derivative(my_earth, x)
            @test g(5.0) isa Number
            # f'(x) = 2x is out of reach: EarthSurrogate is piecewise linear, so
            # its derivative is a step function, constant between knots.
            # Accuracy is asserted against a target inside the model's own span
            # instead — a hinge, whose slopes the surrogate reproduces exactly.
            f_hinge = t -> 1 + 2 * t + 3 * max(0, t - 4)
            x_hinge = collect(0.0:0.5:10.0)
            earth_hinge = EarthSurrogate(x_hinge, f_hinge.(x_hinge), lb, ub)
            dh = t -> ForwardDiff.derivative(earth_hinge, t)
            @test isapprox(dh(2.0), 2.0, atol = 1.0e-8)
            @test isapprox(dh(7.0), 5.0, atol = 1.0e-8)
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
            cd = [
                (my_inverse([2.0, 5.0] + h * e) - my_inverse([2.0, 5.0] - h * e)) / 2h
                    for e in ([1.0, 0.0], [0.0, 1.0])
            ]
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
            h = 1.0e-6
            cd = [
                (my_loba_ND([2.0, 5.0] + h * e) - my_loba_ND([2.0, 5.0] - h * e)) / 2h
                    for e in ([1.0, 0.0], [0.0, 1.0])
            ]
            @test isapprox(g([2.0, 5.0]), cd, atol = 1.0e-4)
            @test all(isfinite, g(collect(x[3])))
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
            cd = [
                (my_wend_ND([2.0, 5.0] + h * e) - my_wend_ND([2.0, 5.0] - h * e)) / 2h
                    for e in ([1.0, 0.0], [0.0, 1.0])
            ]
            @test isapprox(g([2.0, 5.0]), cd, atol = 1.0e-4)

            # At a sample point `norm` of the zero difference is NaN under AD.
            # The kernel peaks there, so its value has to survive too: a
            # composed objective feeds that value through the chain rule.
            node = collect(x[3])
            @test all(isfinite, g(node))
            obj = q -> my_wend_ND(q)^2
            cdn = [
                (obj(node + h * e) - obj(node - h * e)) / 2h
                    for e in ([1.0, 0.0], [0.0, 1.0])
            ]
            @test isapprox(ForwardDiff.gradient(obj, node), cdn, atol = 1.0e-4)
        end

        @testset "GEK" begin
            y1 = y
            der = x -> [x[2], x[1]]  # Gradient of f(x) = x[1] * x[2]
            y2 = vcat([der(xi) for xi in x]...)  # Flatten gradients by point
            y_gek = vcat(y1, y2)
            my_gek = GEK(x, y_gek, lb, ub; optimize_theta = false)
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
            # `std_error_at_point` is a `sqrt` of a variance that this dense a
            # design pins near its numerical floor almost everywhere; at
            # [2.0, 5.0] specifically that floor is close enough to zero that
            # the `sqrt`'s derivative blows up. [1.0, 1.0] sits away from that.
            se = x -> ForwardDiff.gradient(t -> std_error_at_point(my_gekpls_ND, t), x)
            @test se([1.0, 1.0]) isa AbstractVector && all(isfinite, se([1.0, 1.0]))
        end

        @testset "KPLS" begin
            my_kpls_ND = KPLS(x, y, 2, lb, ub, [1.0, 1.0]; optimize_theta = false)
            g = x -> ForwardDiff.gradient(my_kpls_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0)
            se = x -> ForwardDiff.gradient(t -> std_error_at_point(my_kpls_ND, t), x)
            @test se([2.0, 5.0]) isa AbstractVector && all(isfinite, se([2.0, 5.0]))
        end

        @testset "KPLSK" begin
            my_kplsk_ND = KPLSK(x, y, 2, lb, ub, [1.0, 1.0]; optimize_theta = false)
            g = x -> ForwardDiff.gradient(my_kplsk_ND, x)
            @test g([2.0, 5.0]) isa AbstractVector
            @test isapprox(g([2.0, 5.0]), [5.0, 2.0], atol = 1.0)
            se = x -> ForwardDiff.gradient(t -> std_error_at_point(my_kplsk_ND, t), x)
            @test se([2.0, 5.0]) isa AbstractVector && all(isfinite, se([2.0, 5.0]))
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
            # f(x) = x[1] * x[2] is an interaction, and EarthSurrogate selects
            # hinges one coordinate at a time, so the model is additive and
            # cannot represent it. The gradient is asserted against an additive
            # target instead, which the model does span.
            f_add = p -> 2 * p[1] + 3 * max(0, p[2] - 5)
            x_add = sample(60, lb, ub, SobolSample())
            earth_add = EarthSurrogate(x_add, f_add.(x_add), lb, ub)
            @test isapprox(
                ForwardDiff.gradient(earth_add, [3.0, 8.0]), [2.0, 3.0], atol = 1.0e-1
            )
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
            # Reverse mode has to agree with forward mode.
            @test result[1] ≈ ForwardDiff.derivative(my_loba, 5.0)
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
            my_gek = GEK(x, y_gek, lb, ub; optimize_theta = false)
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
            # `std_error_at_point` used to reconstruct a `Cholesky` from a stored
            # plain matrix on every call; Zygote has no adjoint for that
            # constructor, so reverse mode used to fail here entirely. Checked
            # on a sparser fit than `my_gekpls`, and cross-checked against
            # ForwardDiff like KPLS/KPLSK below: at `my_gekpls`'s n=1000
            # density, the predictive variance sits at its numerical floor and
            # its derivative is BLAS/LAPACK-rounding noise, not signal, so
            # forward/reverse mode can disagree by tens of percent there for
            # reasons that have nothing to do with either backend being wrong.
            se_x = sample(5, lb, ub, SobolSample())
            se_gekpls = GEKPLS(
                se_x, f.(se_x), Zygote.gradient.(f, se_x), n_comp, delta_x, lb, ub,
                extra_points, initial_theta
            )
            se = Zygote.gradient(t -> std_error_at_point(se_gekpls, t), 5.0)[1]
            @test se isa Number && isfinite(se)
            @test se ≈ ForwardDiff.derivative(t -> std_error_at_point(se_gekpls, t), 5.0) rtol = 1.0e-3
        end

        @testset "KPLS" begin
            # The callable used to wrap the query point in an array, which
            # reverse-mode AD cannot push a gradient back through.
            my_kpls = KPLS(x, y, 1, [lb], [ub], [1.0]; optimize_theta = false)
            g = x -> Zygote.gradient(my_kpls, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            # Accuracy test: f(x) = x^2, f'(x) = 2x, so f'(5.0) = 10.0
            @test isapprox(result[1], 10.0, atol = 1.0)
            # Reverse mode has to agree with forward mode.
            @test result[1] ≈ ForwardDiff.derivative(my_kpls, 5.0)
            # See the GEKPLS testset above for why this uses a sparser fit.
            se_x = sample(5, lb, ub, SobolSample())
            se_kpls = KPLS(se_x, f.(se_x), 1, [lb], [ub], [1.0]; optimize_theta = false)
            se = Zygote.gradient(t -> std_error_at_point(se_kpls, t), 5.0)[1]
            @test se isa Number && isfinite(se)
            @test se ≈ ForwardDiff.derivative(t -> std_error_at_point(se_kpls, t), 5.0) rtol = 1.0e-3
        end

        @testset "KPLSK" begin
            my_kplsk = KPLSK(x, y, 1, [lb], [ub], [1.0]; optimize_theta = false)
            g = x -> Zygote.gradient(my_kplsk, x)
            result = g(5.0)
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Number
            @test isapprox(result[1], 10.0, atol = 1.0)
            @test result[1] ≈ ForwardDiff.derivative(my_kplsk, 5.0)
            # See the GEKPLS testset above for why this uses a sparser fit.
            se_x = sample(5, lb, ub, SobolSample())
            se_kplsk = KPLSK(se_x, f.(se_x), 1, [lb], [ub], [1.0]; optimize_theta = false)
            se = Zygote.gradient(t -> std_error_at_point(se_kplsk, t), 5.0)[1]
            @test se isa Number && isfinite(se)
            @test se ≈ ForwardDiff.derivative(t -> std_error_at_point(se_kplsk, t), 5.0) rtol = 1.0e-3
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
            # As in the ForwardDiff testset: the smooth f'(x) = 2x is out of
            # reach for a piecewise-linear model, so the slope is asserted
            # against a hinge target, which lies in the model's own span.
            f_hinge = t -> 1 + 2 * t + 3 * max(0, t - 4)
            x_hinge = collect(0.0:0.5:10.0)
            earth_hinge = EarthSurrogate(x_hinge, f_hinge.(x_hinge), lb, ub)
            @test isapprox(Zygote.gradient(earth_hinge, 2.0)[1], 2.0, atol = 1.0e-8)
            @test isapprox(Zygote.gradient(earth_hinge, 7.0)[1], 5.0, atol = 1.0e-8)
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
            @test all(
                isapprox.(
                    result[1],
                    Tuple(ForwardDiff.gradient(my_linear, [2.0, 5.0]))
                )
            )

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
            @test all(
                isapprox.(
                    result[1], Tuple(ForwardDiff.gradient(my_inverse, [2.0, 5.0]))
                )
            )
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
            @test all(
                isapprox.(
                    result[1], Tuple(ForwardDiff.gradient(my_loba_ND, [2.0, 5.0]))
                )
            )
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
            @test all(
                isapprox.(
                    result[1], Tuple(ForwardDiff.gradient(my_wend_ND, [2.0, 5.0]))
                )
            )
        end

        @testset "GEK" begin
            y1 = y
            der = x -> [x[2], x[1]]  # Gradient of f(x) = x[1] * x[2]
            y2 = vcat([der(xi) for xi in x]...)  # Flatten gradients by point
            y_gek = vcat(y1, y2)
            my_gek = GEK(x, y_gek, lb, ub; optimize_theta = false)
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
            # [1.0, 1.0], not [2.0, 5.0]; see the ForwardDiff "ND" GEKPLS testset.
            # Not cross-checked against ForwardDiff, unlike KPLS/KPLSK below;
            # see the 1D GEKPLS testset above for why.
            se = Zygote.gradient(t -> std_error_at_point(my_gekpls_ND, t), (1.0, 1.0))[1]
            @test se isa Tuple && all(isfinite, se)
        end

        @testset "KPLS" begin
            my_kpls_ND = KPLS(x, y, 2, lb, ub, [1.0, 1.0]; optimize_theta = false)
            g = x -> Zygote.gradient(my_kpls_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            # Accuracy test: f(x) = x[1] * x[2], ∇f = [x[2], x[1]], so ∇f([2.0, 5.0]) = [5.0, 2.0]
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0))
            se = Zygote.gradient(t -> std_error_at_point(my_kpls_ND, t), (2.0, 5.0))[1]
            @test se isa Tuple && all(isfinite, se)
            fd = ForwardDiff.gradient(t -> std_error_at_point(my_kpls_ND, t), [2.0, 5.0])
            @test collect(se) ≈ fd rtol = 1.0e-3
        end

        @testset "KPLSK" begin
            my_kplsk_ND = KPLSK(x, y, 2, lb, ub, [1.0, 1.0]; optimize_theta = false)
            g = x -> Zygote.gradient(my_kplsk_ND, x)
            result = g((2.0, 5.0))
            @test result isa Tuple
            @test length(result) == 1
            @test result[1] isa Tuple
            @test all(isapprox.(result[1], (5.0, 2.0), atol = 1.0))
            se = Zygote.gradient(t -> std_error_at_point(my_kplsk_ND, t), (2.0, 5.0))[1]
            @test se isa Tuple && all(isfinite, se)
            fd = ForwardDiff.gradient(t -> std_error_at_point(my_kplsk_ND, t), [2.0, 5.0])
            @test collect(se) ≈ fd rtol = 1.0e-3
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
            # As in the ForwardDiff testset: x[1] * x[2] is an interaction that
            # an additive model cannot represent, so the gradient is asserted
            # against an additive target, which it does span.
            f_add = p -> 2 * p[1] + 3 * max(0, p[2] - 5)
            x_add = sample(60, lb, ub, SobolSample())
            earth_add = EarthSurrogate(x_add, f_add.(x_add), lb, ub)
            @test all(
                isapprox.(
                    Zygote.gradient(earth_add, (3.0, 8.0))[1], (2.0, 3.0), atol = 1.0e-1
                )
            )
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

# AD for the extension-backed surrogates. Five of the seven differentiate under
# both backends; `XGBoostSurrogate` and `SVMSurrogate` wrap gradient-boosted
# trees and a LIBSVM model and cannot, which is asserted below rather than
# assumed.
@testset "AD for extension surrogates" begin
    f1 = t -> (t - 3.7)^2 + 1.0
    f2 = z -> (z[1] - 2.5)^2 + (z[2] - 7.5)^2 + 1.0
    df1 = t -> 2 * (t - 3.7)

    @testset "NeuralSurrogate" begin
        @testset "1-D" begin
            Random.seed!(3)
            lb, ub = 1.0, 6.0
            x = sample(20, lb, ub, SobolSample())
            s = NeuralSurrogate(
                x, f1.(x), lb, ub,
                model = Chain(Dense(1, 6, tanh), Dense(6, 1)), n_epochs = 20
            )
            z = only(Zygote.gradient(t -> s(t), 3.0))
            fd = ForwardDiff.derivative(t -> s(t), 3.0)
            @test z isa Number && isfinite(z)
            @test fd isa Number && isfinite(fd)
            # The two backends must agree on the same model.
            @test isapprox(z, fd; rtol = 1.0e-3)
        end

        @testset "N-D" begin
            Random.seed!(3)
            lb, ub = [1.0, 1.0], [6.0, 6.0]
            x = sample(20, lb, ub, SobolSample())
            s = NeuralSurrogate(
                x, f2.(x), lb, ub,
                model = Chain(Dense(2, 6, tanh), Dense(6, 1)), n_epochs = 20
            )
            z = only(Zygote.gradient(v -> s(v), [3.0, 3.0]))
            fd = ForwardDiff.gradient(v -> s(v), [3.0, 3.0])
            @test length(z) == 2 && all(isfinite, z)
            @test length(fd) == 2 && all(isfinite, fd)
            @test isapprox(collect(z), fd; rtol = 1.0e-3)
        end
    end

    @testset "AbstractGPSurrogate" begin
        @testset "1-D" begin
            Random.seed!(3)
            lb, ub = 1.0, 6.0
            x = sample(20, lb, ub, SobolSample())
            s = AbstractGPSurrogate(
                x, f1.(x), gp = GP(SqExponentialKernel()),
                Σy = 0.05
            )
            z = only(Zygote.gradient(t -> s(t), 3.0))
            fd = ForwardDiff.derivative(t -> s(t), 3.0)
            @test isfinite(z) && isfinite(fd)
            @test isapprox(z, fd; rtol = 1.0e-6)
            # A GP interpolating a smooth target should have the right sign.
            @test sign(z) == sign(df1(3.0))
        end

        @testset "N-D" begin
            # The design is stored as tuples and the query must match: a vector
            # raises `DimensionMismatch: dimensionality of x (2) is not ...`,
            # because `[a, b]` reads as two one-dimensional points.
            Random.seed!(3)
            lb, ub = [1.0, 1.0], [6.0, 6.0]
            x = sample(20, lb, ub, SobolSample())
            s = AbstractGPSurrogate(
                x, f2.(x), gp = GP(SqExponentialKernel()),
                Σy = 0.05
            )
            z = only(Zygote.gradient(v -> s((v[1], v[2])), [3.0, 3.0]))
            fd = ForwardDiff.gradient(v -> s((v[1], v[2])), [3.0, 3.0])
            @test length(z) == 2 && all(isfinite, z)
            @test isapprox(collect(z), fd; rtol = 1.0e-6)
        end
    end

    @testset "PolynomialChaosSurrogate" begin
        @testset "1-D" begin
            Random.seed!(3)
            lb, ub = 1.0, 6.0
            x = sample(20, lb, ub, SobolSample())
            s = PolynomialChaosSurrogate(x, f1.(x), lb, ub)
            z = only(Zygote.gradient(t -> s(t), 3.0))
            fd = ForwardDiff.derivative(t -> s(t), 3.0)
            @test isfinite(z) && isfinite(fd)
            @test isapprox(z, fd; rtol = 1.0e-6)
            # A polynomial chaos expansion of a quadratic is essentially exact.
            @test isapprox(z, df1(3.0); atol = 0.2)
        end

        @testset "N-D" begin
            Random.seed!(3)
            lb, ub = [1.0, 1.0], [6.0, 6.0]
            x = sample(20, lb, ub, SobolSample())
            s = PolynomialChaosSurrogate(x, f2.(x), lb, ub)
            z = only(Zygote.gradient(v -> s(v), [3.0, 3.0]))
            fd = ForwardDiff.gradient(v -> s(v), [3.0, 3.0])
            @test length(z) == 2 && all(isfinite, z)
            @test isapprox(collect(z), fd; rtol = 1.0e-6)
        end
    end
end

# Multi-output AD for the surrogates that support a vector response: the
# Jacobian row per output is checked against the analytic one, not merely
# checked for not throwing.
@testset "multi-output AD" begin
    lb, ub = [1.0, 1.0], [6.0, 6.0]
    # Jacobian is [2x1 0; 0 1], so it is known exactly at any point.
    f = z -> [z[1]^2, z[2]]
    Random.seed!(4)
    x = sample(30, lb, ub, SobolSample())
    y = f.(x)
    at = [2.0, 5.0]
    expected = [2 * at[1] 0.0; 0.0 1.0]

    cases = [
        ("RadialBasis", RadialBasis(x, y, lb, ub, rad = linearRadial())),
        ("InverseDistance", InverseDistanceSurrogate(x, y, lb, ub, p = 1.4)),
        ("SecondOrderPolynomial", SecondOrderPolynomialSurrogate(x, y, lb, ub)),
    ]

    @testset "$(name)" for (name, surr) in cases
        J = Zygote.jacobian(v -> surr(v), at)[1]
        @test size(J) == (2, 2)
        @test all(isfinite, J)
        # The two backends must agree on the same fitted model, whatever the
        # model's own approximation error is.
        @test isapprox(J, ForwardDiff.jacobian(v -> surr(v), at); rtol = 1.0e-6)
    end

    @testset "an interpolant recovers the true Jacobian" begin
        # `RadialBasis` interpolates this design closely enough to check the
        # derivative values, not just that they are finite and consistent.
        surr = RadialBasis(x, y, lb, ub, rad = linearRadial())
        J = Zygote.jacobian(v -> surr(v), at)[1]
        @test isapprox(J, expected; atol = 1.5)
        # The second output is exactly z[2], so its row is [0, 1] to good accuracy.
        @test isapprox(J[2, :], [0.0, 1.0]; atol = 0.3)
    end
end

# Coverage for the extension surrogates the block above does not reach:
# `GENNSurrogate` and `MOE` under ForwardDiff as well as Zygote, both
# dimensionalities, against the analytic derivative; and the two that cannot be
# differentiated at all, pinned so the claim is checked rather than assumed.
@testset "AD for the remaining extension surrogates" begin
    f1 = t -> (t - 3.7)^2 + 1.0
    df1 = t -> 2 * (t - 3.7)
    f2 = z -> (z[1] - 2.5)^2 + (z[2] - 7.5)^2 + 1.0
    df2 = z -> [2 * (z[1] - 2.5), 2 * (z[2] - 7.5)]

    lb1, ub1 = 1.0, 6.0
    lb2, ub2 = [1.0, 1.0], [6.0, 9.0]
    Random.seed!(5)
    x1 = sample(30, lb1, ub1, SobolSample())
    x2 = sample(40, lb2, ub2, SobolSample())
    y1 = f1.(x1)
    y2 = f2.(x2)
    q1 = 3.0
    q2 = [2.0, 5.0]

    @testset "GENNSurrogate" begin
        # Trained on gradients, so its derivative is the one thing it should get
        # right. `predict_derivative` is a prediction; these are true AD through
        # the network, and the two must agree.
        @testset "1-D" begin
            dydx = reshape(df1.(x1), length(x1), 1)
            s = GENNSurrogate(x1, y1, lb1, ub1, dydx; n_epochs = 400)
            fd = ForwardDiff.derivative(s, q1)
            zy = only(Zygote.gradient(s, q1))
            @test fd isa Number && isfinite(fd)
            @test zy ≈ fd rtol = 1.0e-4
            @test fd ≈ df1(q1) atol = 1.5
            @test only(predict_derivative(s, q1)) ≈ fd atol = 1.0e-4
        end

        @testset "N-D" begin
            dydx = reduce(vcat, [reshape(df2(collect(p)), 1, 2) for p in x2])
            s = GENNSurrogate(x2, y2, lb2, ub2, dydx; n_epochs = 400)
            fd = ForwardDiff.gradient(s, q2)
            zy = only(Zygote.gradient(s, q2))
            @test length(fd) == 2 && all(isfinite, fd)
            @test zy ≈ fd rtol = 1.0e-4
            @test fd ≈ df2(q2) atol = 3.0
        end
    end

    @testset "MOE" begin
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

        @testset "1-D" begin
            s = MOE(x1, y1, experts)
            fd = ForwardDiff.derivative(s, q1)
            zy = only(Zygote.gradient(s, q1))
            @test fd isa Number && isfinite(fd)
            @test zy ≈ fd rtol = 1.0e-4
            @test fd ≈ df1(q1) atol = 0.5
        end

        @testset "N-D" begin
            s = MOE(x2, y2, experts; ndim = 2)
            fd = ForwardDiff.gradient(s, q2)
            zy = only(Zygote.gradient(s, q2))
            @test length(fd) == 2 && all(isfinite, fd)
            @test zy ≈ fd rtol = 1.0e-4
            @test fd ≈ df2(q2) atol = 0.5
        end
    end

    @testset "AbstractGPSurrogate takes a point in either representation" begin
        # `ForwardDiff.gradient` supplies a coordinate vector. The call handed
        # the point straight to the kernel, which compared it against a
        # tuple-stored design and raised `DimensionMismatch`, so this surrogate
        # could not be differentiated in more than one dimension at all.
        s = AbstractGPSurrogate(x2, y2)
        @test s(Tuple(q2)) == s(q2)
        @test std_error_at_point(s, Tuple(q2)) == std_error_at_point(s, q2)
        fd = ForwardDiff.gradient(s, q2)
        zy = only(Zygote.gradient(s, q2))
        @test length(fd) == 2 && all(isfinite, fd)
        @test zy ≈ fd rtol = 1.0e-4
    end

    @testset "the non-differentiable surrogates fail rather than answer" begin
        # Gradient-boosted trees and a LIBSVM model are piecewise constant and
        # wrap foreign calls. Neither backend can differentiate them, and both
        # must say so rather than return a plausible-looking zero.
        labels1 = round.(Int, y1) .% 2
        labels2 = round.(Int, y2) .% 2
        pairs = [
            (
                "XGBoostSurrogate", XGBoostSurrogate(x1, y1, lb1, ub1),
                XGBoostSurrogate(x2, y2, lb2, ub2),
            ),
            (
                "SVMSurrogate", SVMSurrogate(x1, labels1, lb1, ub1),
                SVMSurrogate(x2, labels2, lb2, ub2),
            ),
        ]
        @testset "$(name)" for (name, s1, s2) in pairs
            @test_throws Exception ForwardDiff.derivative(s1, q1)
            @test_throws Exception Zygote.gradient(s1, q1)
            @test_throws Exception ForwardDiff.gradient(s2, q2)
            @test_throws Exception Zygote.gradient(s2, q2)
            # They must still predict; only differentiation is unavailable.
            @test s1(q1) isa Number
            @test s2(q2) isa Number
        end
    end
end
