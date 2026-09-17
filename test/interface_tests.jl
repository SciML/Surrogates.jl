# Interface compatibility tests for Surrogates.jl
# Tests BigFloat support for SciML array/number interface compliance

using Test
using Surrogates
using SurrogatesBase

include("testutils.jl")

mutable struct InterfaceContractDeterministic{T} <: SurrogatesBase.AbstractDeterministicSurrogate
    x::Vector{T}
    y::Vector{T}
end

mutable struct InterfaceContractStochastic{T} <: SurrogatesBase.AbstractStochasticSurrogate
    x::Vector{T}
    y::Vector{T}
end

(s::InterfaceContractDeterministic)(point) = point^2
(s::InterfaceContractStochastic)(point) = point^2

function SurrogatesBase.update!(
        s::Union{
            InterfaceContractDeterministic, InterfaceContractStochastic,
        }, x_new, y_new
    )
    if x_new isa AbstractVector && y_new isa AbstractVector
        append!(s.x, x_new)
        append!(s.y, y_new)
    else
        push!(s.x, x_new)
        push!(s.y, y_new)
    end
    return nothing
end

Surrogates.std_error_at_point(::InterfaceContractStochastic, point) = abs(point)
Surrogates.logpdf_surrogate(::InterfaceContractStochastic) = -0.5

@testset "Generic surrogate contracts" begin
    @test SRBF() isa Surrogates.SurrogateOptimizationAlgorithm
    @test MeanConstantLiar() isa Surrogates.ParallelStrategy

    deterministic = InterfaceContractDeterministic([0.0], [0.0])
    @test deterministic isa Surrogates.AbstractSurrogate
    @test deterministic(2.0) == 4.0
    @test update!(deterministic, 2.0, 4.0) === nothing
    @test deterministic.x == [0.0, 2.0]
    @test deterministic.y == [0.0, 4.0]

    stochastic = InterfaceContractStochastic([0.0], [0.0])
    @test stochastic isa Surrogates.AbstractSurrogate
    @test stochastic(2.0) == 4.0
    @test std_error_at_point(stochastic, 2.0) == 2.0
    @test logpdf_surrogate(stochastic) == -0.5
    @test update!(stochastic, [1.0, 2.0], [1.0, 4.0]) === nothing
    @test stochastic.x == [0.0, 1.0, 2.0]
    @test stochastic.y == [0.0, 1.0, 4.0]
end

@testset "Interface Compatibility" begin
    @testset "BigFloat Support - 1D Surrogates" begin
        # Test data with BigFloat
        x_bf = BigFloat[1.0, 2.0, 3.0, 4.0, 5.0]
        y_bf = BigFloat[0.5, 1.2, 2.1, 2.8, 3.6]
        lb_bf = BigFloat(0.0)
        ub_bf = BigFloat(6.0)
        test_point = BigFloat(2.5)

        @testset "RadialBasis 1D" begin
            rad = RadialBasis(x_bf, y_bf, lb_bf, ub_bf)
            result = rad(test_point)
            @test result isa BigFloat
        end

        @testset "InverseDistanceSurrogate 1D" begin
            ids = InverseDistanceSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = ids(test_point)
            @test result isa BigFloat
        end

        @testset "LobachevskySurrogate 1D" begin
            lob = LobachevskySurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = lob(test_point)
            @test result isa BigFloat
        end

        @testset "SecondOrderPolynomialSurrogate 1D" begin
            sop = SecondOrderPolynomialSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = sop(test_point)
            @test result isa BigFloat
        end

        @testset "Wendland 1D" begin
            wen = Wendland(x_bf, y_bf, lb_bf, ub_bf)
            result = wen(test_point)
            @test result isa BigFloat
        end

        @testset "Kriging 1D" begin
            k = Kriging(x_bf, y_bf, lb_bf, ub_bf)
            result = k(test_point)
            @test result isa BigFloat
        end
    end

    @testset "BigFloat Support - ND Surrogates" begin
        # Test data with BigFloat for N-dimensional
        # Six points, not five: a full quadratic in two dimensions has
        # 1 + 2d + d(d - 1) / 2 = 6 coefficients, so five samples leave
        # SecondOrderPolynomialSurrogate underdetermined.
        x_bf = [
            (BigFloat(1.0), BigFloat(2.0)), (BigFloat(2.0), BigFloat(3.0)),
            (BigFloat(3.0), BigFloat(1.0)), (BigFloat(4.0), BigFloat(4.0)),
            (BigFloat(5.0), BigFloat(2.0)), (BigFloat(2.0), BigFloat(5.0)),
        ]
        y_bf = BigFloat[0.5, 1.2, 2.1, 2.8, 3.5, 2.4]
        lb_bf = (BigFloat(0.0), BigFloat(0.0))
        ub_bf = (BigFloat(6.0), BigFloat(5.0))
        test_point = (BigFloat(2.5), BigFloat(2.5))

        @testset "RadialBasis ND" begin
            rad = RadialBasis(x_bf, y_bf, lb_bf, ub_bf)
            result = rad(test_point)
            @test result isa BigFloat
        end

        @testset "InverseDistanceSurrogate ND" begin
            ids = InverseDistanceSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = ids(test_point)
            @test result isa BigFloat
        end

        @testset "SecondOrderPolynomialSurrogate ND" begin
            sop = SecondOrderPolynomialSurrogate(x_bf, y_bf, lb_bf, ub_bf)
            result = sop(test_point)
            @test result isa BigFloat
        end

        @testset "Kriging ND" begin
            k = Kriging(x_bf, y_bf, lb_bf, ub_bf)
            result = k(test_point)
            @test result isa BigFloat
        end
    end
end

@testset "SurrogatesBase parameter interface" begin
    # `parameters`, `hyperparameters` and `update_hyperparameters!` are optional
    # in `SurrogatesBase`, and none of them had a single method here, so generic
    # code written against the interface could neither inspect nor refit any
    # Surrogates.jl model.
    using Random
    f = x -> (x - 3.7)^2 + 1.0
    df = x -> 2 * (x - 3.7)
    Random.seed!(3)
    x = sample(20, 1.0, 6.0, SobolSample())
    y = f.(x)

    @testset "learned state and configuration are separated" begin
        # `theta` and `p` are configuration even when fitted; `mu`, `b`, `sigma`
        # and the coefficient vectors are outputs of the fit.
        cases = [
            (
                "Kriging", Kriging(x, y, 1.0, 6.0),
                (:mu, :b, :sigma), (:p, :theta),
            ),
            (
                "GEK", GEK(x, vcat(y, df.(x)), 1.0, 6.0; optimize_theta = false),
                (:mu, :b, :sigma), (:p, :theta),
            ),
            (
                "RadialBasis", RadialBasis(x, y, 1.0, 6.0, rad = linearRadial()),
                (:coeff,), (
                    :radial_function, :dim_poly, :scale_factor, :sparse,
                    :regularization,
                ),
            ),
            (
                "Wendland", Wendland(x, y, 1.0, 6.0),
                (:coeff,), (:eps, :maxiters, :tol),
            ),
            (
                "Lobachevsky", LobachevskySurrogate(x, y, 1.0, 6.0, alpha = 2.0, n = 4),
                (:coeff,), (:alpha, :n, :sparse),
            ),
            (
                "InverseDistance", InverseDistanceSurrogate(x, y, 1.0, 6.0, p = 2.0),
                (), (:p,),
            ),
            ("LinearSurrogate", LinearSurrogate(x, y, 1.0, 6.0), (:coeff,), ()),
            (
                "SecondOrderPolynomial", SecondOrderPolynomialSurrogate(x, y, 1.0, 6.0),
                (:beta,), (),
            ),
        ]
        @testset "$(name)" for (name, surr, want_params, want_hyper) in cases
            @test parameters(surr) isa NamedTuple
            @test hyperparameters(surr) isa NamedTuple
            @test keys(parameters(surr)) == want_params
            @test keys(hyperparameters(surr)) == want_hyper
            # Reading the interface must not disturb the model.
            before = surr(3.0)
            parameters(surr); hyperparameters(surr)
            @test surr(3.0) == before
        end
    end

    @testset "update_hyperparameters! actually refits" begin
        # Starting from a deliberately poor `theta`, the refit has to move it and
        # improve the fit — otherwise the method is a no-op that merely looks
        # implemented.
        g = t -> sin(3t) + 0.3cos(7t)
        Random.seed!(5)
        xs = sample(25, 0.0, 6.0, SobolSample())
        k = Kriging(xs, g.(xs), 0.0, 6.0; theta = 5.0, optimize_theta = false)
        grid = range(0.1, 5.9, length = 60)
        rmse(m) = sqrt(sum((m(t) - g(t))^2 for t in grid) / length(grid))

        theta_before, rmse_before = hyperparameters(k).theta, rmse(k)
        returned = update_hyperparameters!(k)
        @test returned === k                       # mutates and returns the model
        @test hyperparameters(k).theta != theta_before
        @test rmse(k) < rmse_before
        # The fitted state must be rebuilt with the new theta, not left stale.
        @test isfinite(parameters(k).mu)
        @test isfinite(parameters(k).sigma)
        @test isfinite(k(3.0))
    end

    @testset "the PLS family exposes and refits its hyperparameters" begin
        # These were missing from the first cut of the interface, and the gap hid
        # a real bug: `hyperparameters(::GEKPLS)` read `g.n_comp` and `g.delta_x`,
        # but the fields are `num_components` and `delta`. Nothing caught it
        # because nothing tested the PLS models.
        using Zygote
        h = z -> sin(3z[1]) + 0.3cos(7z[2])
        lbn, ubn = [0.0, 0.0], [3.0, 3.0]
        Random.seed!(5)
        xn = sample(60, lbn, ubn, SobolSample())
        yn = h.(xn)
        grid = [(p[1], p[2]) for p in eachrow(rand(MersenneTwister(9), 40, 2) .* 3.0)]
        rmse(m) = sqrt(sum((m(p) - h(collect(p)))^2 for p in grid) / length(grid))

        cases = [
            (
                "KPLS", () -> KPLS(
                    xn, yn, 2, lbn, ubn, [1.0, 1.0];
                    optimize_theta = false
                ), (:theta, :n_comp),
            ),
            (
                "KPLSK", () -> KPLSK(
                    xn, yn, 2, lbn, ubn, [1.0, 1.0];
                    optimize_theta = false
                ), (:theta, :theta_pls, :n_comp),
            ),
            (
                "GEKPLS", () -> GEKPLS(
                    xn, yn, Zygote.gradient.(h, xn), 2, 1.0e-4,
                    lbn, ubn, 2, [1.0e-2, 1.0e-2]; optimize_theta = false
                ),
                (:theta, :n_comp, :delta_x, :extra_points, :nugget, :noise),
            ),
        ]

        @testset "$(name)" for (name, mk, want_hyper) in cases
            m = mk()
            # Every advertised key must actually exist on the struct.
            @test keys(hyperparameters(m)) == want_hyper
            @test parameters(m) isa NamedTuple
            @test :sigma2 in keys(parameters(m))

            theta_before = collect(hyperparameters(m).theta)
            returned = update_hyperparameters!(m)
            @test returned === m
            @test collect(hyperparameters(m).theta) != theta_before
            # The refit must rebuild the model, not just move `theta`.
            @test isfinite(m((1.0, 1.0)))
            @test rmse(m) < 1.0
        end

        @testset "GEKPLS keeps its gradients through a refit" begin
            # The gradients are stored as an `n x d` matrix but the constructor
            # takes Zygote's broadcast shape, so refitting has to convert back;
            # getting that wrong loses the gradient block entirely.
            m = GEKPLS(
                xn, yn, Zygote.gradient.(h, xn), 2, 1.0e-4, lbn, ubn, 2,
                [1.0e-2, 1.0e-2]; optimize_theta = false
            )
            before = copy(m.grads)
            update_hyperparameters!(m)
            @test size(m.grads) == size(before)
            @test m.grads ≈ before
        end
    end

    @testset "surrogates with fixed configuration have no refit method" begin
        # A silent no-op would be worse than a MethodError: it would look as
        # though the configuration had been optimized.
        rb = RadialBasis(x, y, 1.0, 6.0, rad = linearRadial())
        @test !hasmethod(update_hyperparameters!, Tuple{typeof(rb)})
        @test !hasmethod(update_hyperparameters!, Tuple{typeof(LinearSurrogate(x, y, 1.0, 6.0))})
    end
end

@testset "surrogates broadcast as scalars" begin
    # A surrogate broadcast as an *argument* must be treated as a scalar.
    # `surrogate.(points)` alone does not test this: there the surrogate is in
    # function position, and broadcast never asks `broadcastable` of the
    # function, so it passes with or without the method. Only the argument
    # position exercises it, and it is what `std_error_at_point.(s, pts)` and
    # `gradient.(s, pts)` do.
    using Random
    f = t -> (t - 3.7)^2 + 1.0
    Random.seed!(3)
    x = sample(20, 1.0, 6.0, SobolSample())
    y = f.(x)
    pts = [2.0, 3.0, 4.0]

    stochastic = [
        ("Kriging", Kriging(x, y, 1.0, 6.0)),
        ("KPLS", KPLS(x, y, 1, [1.0], [6.0], [1.0]; optimize_theta = false)),
        ("KPLSK", KPLSK(x, y, 1, [1.0], [6.0], [1.0]; optimize_theta = false)),
    ]
    deterministic = [
        ("RadialBasis", RadialBasis(x, y, 1.0, 6.0, rad = linearRadial())),
        ("Wendland", Wendland(x, y, 1.0, 6.0)),
    ]

    apply(s, p) = s(p)

    @testset "$(name)" for (name, surr) in vcat(stochastic, deterministic)
        vals = surr.(pts)
        @test length(vals) == 3
        @test all(isfinite, vals)
        # Broadcasting and looping must agree.
        @test vals ≈ [surr(p) for p in pts]
        # The surrogate as a broadcast argument: this is the one that needs
        # `broadcastable`, and it must agree with the loop too.
        @test apply.(surr, pts) ≈ [surr(p) for p in pts]
    end

    @testset "$(name): std_error_at_point broadcasts over points" for (name, surr) in
        stochastic
        @test std_error_at_point.(surr, pts) ≈
            [std_error_at_point(surr, p) for p in pts]
    end

    @testset "the retyped models really are stochastic" begin
        # If this stops holding, the broadcast fix above is being tested against
        # the wrong half of the union.
        @test all(s -> s[2] isa SurrogatesBase.AbstractStochasticSurrogate, stochastic)
        @test all(
            s -> s[2] isa SurrogatesBase.AbstractDeterministicSurrogate,
            deterministic
        )
    end
end

@testset "public bindings carry their own docstring" begin
    # A comment between a docstring and the definition it precedes silently
    # detaches the docstring: Julia attaches nothing, and the `@docs` block in
    # the manual then renders a different method's docstring, or errors. Every
    # binding named in an `@docs` block has to answer with documentation.
    documented(b) = !occursin("No documentation found", string(Base.Docs.doc(b)))

    @testset "$(name)" for name in [
            :Kriging, :GEK, :GEKPLS, :KPLS, :KPLSK, :RadialBasis, :Wendland,
            :LobachevskySurrogate, :LinearSurrogate, :InverseDistanceSurrogate,
            :SecondOrderPolynomialSurrogate, :EarthSurrogate,
            :VariableFidelitySurrogate, :sample, :std_error_at_point,
            :surrogate_optimize!, :potential_optimal_points,
        ]
        @test documented(getfield(Surrogates, name))
    end

    # The struct docstring is the long one, so a detached docstring shows up as
    # a suspiciously short rendering even when a constructor method still has
    # one of its own.
    @testset "$(name) renders its full docstring" for name in
        [:Kriging, :GEK, :GEKPLS, :KPLS, :KPLSK]
        @test length(string(Base.Docs.doc(getfield(Surrogates, name)))) > 1000
    end
end

@testset "update! accepts a point in either representation" begin
    # A `d`-dimensional point may be written as a tuple or as a coordinate
    # vector, and every call overload accepts both. `update!` differed: only the
    # two surrogates carrying a private `_match_stored` took a coordinate vector
    # against a tuple-stored design, and the other fourteen raised a bare
    # `MethodError` from `vcat`. The helper now lives in `src/utils.jl`.
    lb, ub = [0.0, 0.0], [5.0, 5.0]

    check_update_representations((x, y) -> RadialBasis(x, y, lb, ub), "RadialBasis")
    check_update_representations((x, y) -> Kriging(x, y, lb, ub), "Kriging")
    check_update_representations((x, y) -> Wendland(x, y, lb, ub), "Wendland")
    check_update_representations(
        (x, y) -> LobachevskySurrogate(x, y, lb, ub), "LobachevskySurrogate"
    )
    check_update_representations(
        (x, y) -> LinearSurrogate(x, y, lb, ub), "LinearSurrogate"
    )
    check_update_representations(
        (x, y) -> InverseDistanceSurrogate(x, y, lb, ub), "InverseDistanceSurrogate"
    )
    check_update_representations(
        (x, y) -> SecondOrderPolynomialSurrogate(x, y, lb, ub),
        "SecondOrderPolynomialSurrogate"
    )
    check_update_representations(
        (x, y) -> EarthSurrogate(x, y, lb, ub), "EarthSurrogate"
    )
    check_update_representations(
        (x, y) -> VariableFidelitySurrogate(x, y, lb, ub),
        "VariableFidelitySurrogate"
    )
    check_update_representations(
        (x, y) -> KPLS(x, y, 2, lb, ub, [1.0, 1.0]; optimize_theta = false), "KPLS"
    )
    check_update_representations(
        (x, y) -> KPLSK(x, y, 2, lb, ub, [1.0, 1.0]; optimize_theta = false), "KPLSK"
    )
end

@testset "the gradient-enhanced models take a point either way too" begin
    lb, ub = [0.0, 0.0], [5.0, 5.0]
    obj(p) = p[1]^2 + p[2]^2
    x = sample(40, lb, ub, SobolSample())
    grads = [(Tuple(2 .* collect(p)),) for p in x]

    # `GEK` stores `[values; gradients]`, so its `y` is built the same way here.
    gek_y = vcat(obj.(x), reduce(vcat, [collect(2 .* collect(p)) for p in x]))
    gek() = GEK(x, gek_y, lb, ub; optimize_theta = false)
    gekpls() = GEKPLS(
        x, obj.(x), grads, 2, 1.0e-4, lb, ub, 2, [0.01, 0.01];
        optimize_theta = false
    )

    @testset "$(name)" for (name, build) in (("GEK", gek), ("GEKPLS", gekpls))
        for point in ((1.0, 2.0), [1.0, 2.0]), grad in ((2.0, 4.0), [2.0, 4.0])
            surr = build()
            n = length(surr.x)
            update!(surr, point, obj(point), grad)
            @test length(surr.x) == n + 1
        end
    end
end
