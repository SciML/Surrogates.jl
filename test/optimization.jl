using Surrogates
using LinearAlgebra
using Random
using LIBSVM
using Zygote
using Test

# Coordinates of a sample point, whatever shape the surrogate stores it in.
coords(p::Number) = [p]
coords(p) = collect(p)

"""
Assert the contract every single-objective `surrogate_optimize!` method obeys,
whatever surrogate or algorithm it was given:

  - it returns the best observation the surrogate holds, as a `(point, value)`;
  - that value is a real evaluation of the objective, not an acquisition value;
  - every sample, including the ones the run added, lies inside the box;
  - the run costs at most `maxiters` objective evaluations.

The last two are what silently broke before: acquisition values were stored as
observations, and `maxiters` bought `maxiters^2` evaluations.
"""
function check_contract(result, surr, obj, lb, ub, n_before, maxiters; atol = 1.0e-8)
    @test result isa Tuple && length(result) == 2
    xbest, ybest = result
    @test ybest ≈ minimum(surr.y)
    @test isapprox(ybest, obj(xbest); atol = atol)
    @test all(p -> all(lb .- 1.0e-8 .<= coords(p) .<= ub .+ 1.0e-8), surr.x)
    @test length(surr.x) <= n_before + maxiters
    return nothing
end

# Interior minima, so a method cannot score well by clamping to a bound. The
# earlier targets were monotone (`2x + 1`, `3‖z‖ + 1`) with the optimum at a
# corner of the box, which is why almost nothing was asserted about them.
f1d(x) = (x - 3.7)^2 + 1.0
const LB1, UB1, MIN1 = 0.0, 10.0, 1.0

f2d(z) = (z[1] - 2.5)^2 + (z[2] - 7.5)^2 + 1.0
const LB2, UB2, MIN2 = [0.0, 0.0], [10.0, 10.0], 1.0

@testset "SRBF" begin
    @testset "converges on an interior optimum" begin
        for (label, mk, tol) in (
                ("Kriging", x -> Kriging(x, f1d.(x), LB1, UB1), 1.0e-3),
                (
                    "RadialBasis",
                    x -> RadialBasis(x, f1d.(x), LB1, UB1, rad = cubicRadial()), 1.0e-3,
                ),
            )
            Random.seed!(11)
            x = sample(15, LB1, UB1, SobolSample())
            surr = mk(x)
            n_before = length(surr.x)
            result = surrogate_optimize!(
                f1d, SRBF(), LB1, UB1, surr, SobolSample(); maxiters = 30
            )
            check_contract(result, surr, f1d, LB1, UB1, n_before, 30)
            @test minimum(surr.y) - MIN1 < tol
        end

        for (label, mk, tol) in (
                ("Kriging", x -> Kriging(x, f2d.(x), LB2, UB2), 1.0e-1),
                (
                    "RadialBasis",
                    x -> RadialBasis(x, f2d.(x), LB2, UB2, rad = cubicRadial()), 1.0e-1,
                ),
            )
            Random.seed!(11)
            x = sample(25, LB2, UB2, SobolSample())
            surr = mk(x)
            n_before = length(surr.x)
            result = surrogate_optimize!(
                f2d, SRBF(), LB2, UB2, surr, SobolSample(); maxiters = 40
            )
            check_contract(result, surr, f2d, LB2, UB2, n_before, 40)
            @test minimum(surr.y) - MIN2 < tol
        end
    end

    # SRBF is the one algorithm that only needs a prediction, so it has to work
    # with every surrogate. These check the contract rather than convergence:
    # a regression model has no reason to locate an interior optimum.
    @testset "works with any surrogate" begin
        Random.seed!(3)
        x1 = sample(12, LB1, UB1, SobolSample())
        y1 = f1d.(x1)
        one_d = (
            "Kriging" => Kriging(x1, y1, LB1, UB1),
            "RadialBasis" => RadialBasis(x1, y1, LB1, UB1, rad = linearRadial()),
            "Wendland" => Wendland(x1, y1, LB1, UB1),
            "Earth" => EarthSurrogate(x1, y1, LB1, UB1),
        )
        for (label, surr) in one_d
            @testset "$label (1-D)" begin
                n_before = length(surr.x)
                result = surrogate_optimize!(
                    f1d, SRBF(), LB1, UB1, surr, HaltonSample(); maxiters = 10
                )
                check_contract(result, surr, f1d, LB1, UB1, n_before, 10)
            end
        end

        Random.seed!(3)
        xN = sample(20, LB2, UB2, SobolSample())
        yN = f2d.(xN)
        n_d = (
            "Kriging" => Kriging(xN, yN, LB2, UB2),
            "RadialBasis" => RadialBasis(xN, yN, LB2, UB2, rad = linearRadial()),
            "Lobachevsky" => LobachevskySurrogate(xN, yN, LB2, UB2),
            "InverseDistance" => InverseDistanceSurrogate(xN, yN, LB2, UB2, p = 2.5),
            "SecondOrderPolynomial" => SecondOrderPolynomialSurrogate(xN, yN, LB2, UB2),
        )
        for (label, surr) in n_d
            @testset "$label (ND)" begin
                n_before = length(surr.x)
                result = surrogate_optimize!(
                    f2d, SRBF(), LB2, UB2, surr, SobolSample(); maxiters = 10
                )
                check_contract(result, surr, f2d, LB2, UB2, n_before, 10)
                @test all(isfinite, surr.y)
                @test all(isfinite(surr(v)) for v in sample(50, LB2, UB2, HaltonSample()))
            end
        end

        # A linear surrogate needs a design it can actually fit.
        Random.seed!(3)
        xL = sample(200, LB2, UB2, SobolSample())
        linear = LinearSurrogate(xL, f2d.(xL), LB2, UB2)
        n_before = length(linear.x)
        result = surrogate_optimize!(
            f2d, SRBF(), LB2, UB2, linear, SobolSample(); maxiters = 10
        )
        check_contract(result, linear, f2d, LB2, UB2, n_before, 10)

        # `SVMSurrogate` wraps a classifier, so it is kept on a small design
        # with few distinct responses; only the contract is meaningful here.
        @testset "SVM (ND)" begin
            objective = z -> 3 * norm(z) + 1
            lb, ub = [1.0, 1.0], [6.0, 6.0]
            xS = sample(5, lb, ub, SobolSample())
            svm = SVMSurrogate(xS, objective.(xS), lb, ub)
            n_before = length(svm.x)
            result = surrogate_optimize!(
                objective, SRBF(), lb, ub, svm, SobolSample(); maxiters = 15
            )
            check_contract(result, svm, objective, lb, ub, n_before, 15)
        end
    end
end

@testset "DYCORS" begin
    @testset "1-D" begin
        for (label, mk) in (
                ("Kriging", x -> Kriging(x, f1d.(x), LB1, UB1)),
                (
                    "RadialBasis",
                    x -> RadialBasis(x, f1d.(x), LB1, UB1, rad = linearRadial()),
                ),
            )
            Random.seed!(5)
            x = sample(15, LB1, UB1, SobolSample())
            surr = mk(x)
            n_before = length(surr.x)
            result = surrogate_optimize!(
                f1d, DYCORS(), LB1, UB1, surr, SobolSample(); maxiters = 40
            )
            check_contract(result, surr, f1d, LB1, UB1, n_before, 40)
            @test minimum(surr.y) - MIN1 < 1.0e-3
        end
    end

    @testset "ND" begin
        Random.seed!(5)
        x = sample(25, LB2, UB2, SobolSample())
        surr = Kriging(x, f2d.(x), LB2, UB2)
        n_before = length(surr.x)
        result = surrogate_optimize!(
            f2d, DYCORS(), LB2, UB2, surr, SobolSample(); maxiters = 60
        )
        check_contract(result, surr, f2d, LB2, UB2, n_before, 60)
        @test minimum(surr.y) - MIN2 < 1.0e-1

        # Wendland and RadialBasis have no `std_error_at_point`, so DYCORS is
        # the interesting multidimensional path for them.
        for (label, mk) in (
                ("Wendland", () -> Wendland(x, f2d.(x), LB2, UB2)),
                (
                    "RadialBasis",
                    () -> RadialBasis(x, f2d.(x), LB2, UB2, rad = linearRadial()),
                ),
            )
            @testset "$label" begin
                Random.seed!(5)
                surr = mk()
                n_before = length(surr.x)
                result = surrogate_optimize!(
                    f2d, DYCORS(), LB2, UB2, surr, SobolSample(); maxiters = 20
                )
                check_contract(result, surr, f2d, LB2, UB2, n_before, 20)
            end
        end
    end

    @testset "step size schedule" begin
        # Doubles after `t_success` consecutive improvements, halves after
        # `t_fail` consecutive failures, and never falls below `sigma_min`.
        sigma, C_success, C_fail = Surrogates.adjust_step_size(1.0, 0.1, 3, 3, 0, 5)
        @test sigma == 2.0
        @test C_success == 0

        sigma, C_success, C_fail = Surrogates.adjust_step_size(1.0, 0.1, 0, 3, 5, 5)
        @test sigma == 0.5
        @test C_fail == 0

        # Below the floor the halving is clamped, not applied.
        sigma, _, _ = Surrogates.adjust_step_size(0.15, 0.1, 0, 3, 5, 5)
        @test sigma == 0.1

        # Neither threshold reached: nothing changes.
        sigma, C_success, C_fail = Surrogates.adjust_step_size(1.0, 0.1, 2, 3, 4, 5)
        @test (sigma, C_success, C_fail) == (1.0, 2, 4)
    end

    @testset "perturbations stay in a narrow box" begin
        # Every candidate is a Gaussian step from the incumbent, so the
        # reflection is what keeps the search inside a box narrower than the
        # step width.
        lb, ub = [0.0, 0.0], [0.5, 0.5]
        obj = z -> (z[1] - 0.2)^2 + (z[2] - 0.4)^2
        Random.seed!(12)
        x = sample(12, lb, ub, SobolSample())
        surr = Kriging(x, obj.(x), lb, ub)
        n_before = length(surr.x)
        result = surrogate_optimize!(
            obj, DYCORS(), lb, ub, surr, SobolSample(); maxiters = 25
        )
        check_contract(result, surr, obj, lb, ub, n_before, 25)
    end
end

@testset "EI" begin
    @testset "1-D" begin
        Random.seed!(2)
        x = sample(15, LB1, UB1, SobolSample())
        krig = Kriging(x, f1d.(x), LB1, UB1)
        n_before = length(krig.x)
        result = surrogate_optimize!(
            f1d, EI(), LB1, UB1, krig, SobolSample(); maxiters = 30
        )
        check_contract(result, krig, f1d, LB1, UB1, n_before, 30)
        @test minimum(krig.y) - MIN1 < 1.0e-2
    end

    @testset "ND" begin
        Random.seed!(2)
        x = sample(25, LB2, UB2, SobolSample())
        krig = Kriging(x, f2d.(x), LB2, UB2)
        n_before = length(krig.x)
        result = surrogate_optimize!(
            f2d, EI(), LB2, UB2, krig, SobolSample(); maxiters = 40
        )
        check_contract(result, krig, f2d, LB2, UB2, n_before, 40)
        @test minimum(krig.y) - MIN2 < 5.0e-1
    end

    @testset "minimizes rather than maximizes" begin
        # `theta` is used as given: three observations cannot identify a maximum
        # likelihood fit of two correlation scales plus the process variance, and
        # the fitted scales come out about five times smaller than the
        # sample-spread default, which makes the model overconfident away from
        # the data and starves EI of exploration.
        objective = z -> 3 * norm(z) + 1
        lb, ub = [-1.0, -1.0], [6.0, 6.0]
        x = [(1.2, 3.0), (3.0, 3.5), (5.2, 5.7)]
        krig = Kriging(x, objective.(x), lb, ub; optimize_theta = false)
        surrogate_optimize!(objective, EI(), lb, ub, krig, SobolSample())
        y_min, index_min = findmin(krig.y)
        @test norm(collect(krig.x[index_min]) .- [0.0, 0.0]) < 0.05 * norm(ub .- lb)
        @test abs(y_min - objective((0.0, 0.0))) < 0.05 * (objective(ub) - objective(lb))
    end

    @testset "acquisition properties" begin
        Random.seed!(1)
        x = sample(10, LB1, UB1, SobolSample())
        krig = Kriging(x, f1d.(x), LB1, UB1)
        f_min = minimum(krig.y)
        grid = collect(range(LB1, UB1, length = 200))

        # An already-evaluated point has no predictive variance left, so no
        # improvement can be expected there.
        @test all(
            Surrogates._expected_improvement(krig, xi, f_min) < 1.0e-12
                for xi in krig.x
        )
        # Improvement is an expectation of a non-negative quantity.
        @test all(Surrogates._expected_improvement(krig, g, f_min) >= 0.0 for g in grid)
        # ... and is strictly positive somewhere, or the search could not move.
        @test maximum(Surrogates._expected_improvement(krig, g, f_min) for g in grid) > 0.0
        # The offset demands a minimum improvement before it counts, so raising
        # it can only lower the score.
        best(xi) = maximum(Surrogates._expected_improvement(krig, g, f_min, xi) for g in grid)
        @test best(0.0) >= best(0.01) >= best(0.5)
    end

    @testset "respects a fixed section" begin
        # The optimum is sought on the slice x[2] == 2.0, so the constrained
        # coordinate must never move.
        objective = x -> x[1]^2 + x[2]^2 + x[3]^2
        x2 = 2.0
        sampler = SectionSample([NaN64, x2, NaN64], SobolSample())
        lb, ub = [-1.0, x2, -1.0], [6.0, x2, 6.0]
        x = sample(5, lb, ub, sampler)
        krig = Kriging(x, objective.(x), lb, ub)
        surrogate_optimize!(objective, EI(), lb, ub, krig, sampler)
        y_min, index_min = findmin(krig.y)
        @test all(p -> coords(p)[2] == x2, krig.x)
        @test norm(coords(krig.x[index_min]) .- [0.0, x2, 0.0]) < 0.05 * norm(ub .- lb)
        @test abs(y_min - objective([0.0, x2, 0.0])) <
            0.05 * (objective(ub) - objective(lb))
    end
end

@testset "LCBS" begin
    @testset "1-D" begin
        Random.seed!(4)
        x = sample(15, LB1, UB1, SobolSample())
        krig = Kriging(x, f1d.(x), LB1, UB1)
        n_before = length(krig.x)
        result = surrogate_optimize!(
            f1d, LCBS(), LB1, UB1, krig, SobolSample(); maxiters = 30
        )
        check_contract(result, krig, f1d, LB1, UB1, n_before, 30)
        @test minimum(krig.y) - MIN1 < 1.0e-2
    end

    @testset "ND" begin
        Random.seed!(4)
        x = sample(25, LB2, UB2, SobolSample())
        krig = Kriging(x, f2d.(x), LB2, UB2)
        n_before = length(krig.x)
        result = surrogate_optimize!(
            f2d, LCBS(), LB2, UB2, krig, SobolSample(); maxiters = 40
        )
        check_contract(result, krig, f2d, LB2, UB2, n_before, 40)
        @test minimum(krig.y) - MIN2 < 5.0e-1
    end

    @testset "k controls exploration" begin
        # A rough target on a sparse design, so the surrogate has real
        # predictive spread to trade against its mean.
        rough(x) = sin(3x) + 0.3 * cos(7x) + 0.05 * (x - 6)^2
        Random.seed!(1)
        x = sample(5, LB1, UB1, SobolSample())
        krig = Kriging(x, rough.(x), LB1, UB1)
        grid = collect(range(LB1, UB1, length = 200))
        mu = [krig(g) for g in grid]
        sd = [std_error_at_point(krig, g) for g in grid]

        # The bound sits below the mean, further for larger k.
        t = 7.3
        @test krig(t) - 4.0 * std_error_at_point(krig, t) <
            krig(t) - 2.0 * std_error_at_point(krig, t) <
            krig(t)

        # Raising k has to move the selected candidate towards more uncertain
        # ground. Minimizing `mu + k*sigma` instead would move it the other way,
        # which is the defect this pins.
        selected_sd = [sd[argmin(mu .- k .* sd)] for k in (0.0, 1.0, 3.0, 10.0, 30.0)]
        @test issorted(selected_sd)
        @test selected_sd[end] > selected_sd[1]

        # The same ordering has to survive a full run: with a larger k the
        # points the run adds sit further from the initial design.
        function mean_gap(k)
            Random.seed!(1)
            xx = sample(5, LB1, UB1, SobolSample())
            kr = Kriging(xx, rough.(xx), LB1, UB1)
            surrogate_optimize!(
                rough, LCBS(), LB1, UB1, kr, SobolSample(); maxiters = 12, k = k
            )
            added = kr.x[6:end]
            return isempty(added) ? 0.0 :
                sum(minimum(abs(a - x) for x in xx) for a in added) / length(added)
        end
        @test mean_gap(0.0) < mean_gap(2.0) < mean_gap(10.0)
    end

    @testset "the bound tightens where there is data" begin
        Random.seed!(4)
        x = sample(10, LB1, UB1, SobolSample())
        krig = Kriging(x, f1d.(x), LB1, UB1)
        grid = collect(range(LB1, UB1, length = 200))

        # An observed point leaves little to be optimistic about, so the bound
        # sits close to the observation there. The gap is not zero: this design
        # needs a conditioning nugget, so the model regularizes rather than
        # interpolating exactly. It is still three orders below the response
        # range, which is what makes the bound informative.
        gaps = [
            abs(krig(xi) - 2.0 * std_error_at_point(krig, xi) - yi)
                for (xi, yi) in zip(krig.x, krig.y)
        ]
        @test maximum(gaps) < 0.01 * (maximum(krig.y) - minimum(krig.y))

        # ... and the model is genuinely more certain on the data than away
        # from it, or `k` would have nothing to trade against.
        at_data = maximum(std_error_at_point(krig, xi) for xi in krig.x)
        off_data = maximum(std_error_at_point(krig, g) for g in grid)
        @test at_data < off_data / 2
    end
end

@testset "SOP" begin
    @testset "1-D" begin
        Random.seed!(6)
        x = sample(20, LB1, UB1, SobolSample())
        krig = Kriging(x, f1d.(x), LB1, UB1)
        n_before = length(krig.x)
        result = surrogate_optimize!(
            f1d, SOP(2), LB1, UB1, krig, SobolSample(); maxiters = 30
        )
        check_contract(result, krig, f1d, LB1, UB1, n_before, 2 * 30)
        @test minimum(krig.y) - MIN1 < 1.0e-2
    end

    @testset "ND" begin
        Random.seed!(6)
        x = sample(20, LB2, UB2, SobolSample())
        krig = Kriging(x, f2d.(x), LB2, UB2)
        n_before = length(krig.x)
        result = surrogate_optimize!(
            f2d, SOP(2), LB2, UB2, krig, SobolSample(); maxiters = 20
        )
        check_contract(result, krig, f2d, LB2, UB2, n_before, 2 * 20)
        @test minimum(krig.y) - MIN2 < 1.0
    end

    @testset "one evaluation per center per iteration" begin
        # `SOP(p)` carries `p` search centers and proposes one candidate from
        # each, so the budget is `p * maxiters`, not `maxiters`.
        for p in (1, 2, 3)
            Random.seed!(4)
            x = sample(20, LB1, UB1, SobolSample())
            krig = Kriging(x, f1d.(x), LB1, UB1)
            surrogate_optimize!(
                f1d, SOP(p), LB1, UB1, krig, SobolSample(); maxiters = 10
            )
            @test length(krig.x) - 20 == p * 10
        end
    end

    @testset "fronts partition the centers" begin
        # Non-dominated sorting assigns every point to exactly one front.
        Random.seed!(6)
        x = sample(8, LB1, UB1, SobolSample())
        krig = Kriging(x, f1d.(x), LB1, UB1)
        fronts = Surrogates.I_tier_ranking_1D(copy(krig.x), krig)
        @test sum(length, values(fronts)) == length(krig.x)
        @test sort(reduce(vcat, values(fronts))) ≈ sort(krig.x)
        # Front 1 is non-dominated, so nothing in a later front can dominate it.
        @test !isempty(fronts[1])
    end
end

@testset "trust region schedule" begin
    # The width only has to be inside its usable range once it has actually
    # moved. Testing it every iteration stops the search after a single
    # evaluation on any box narrower than `0.2 / 0.8`, since the *starting*
    # width already exceeds `0.8 * diameter` there.
    for (lb, ub) in (([0.0, 0.0], [0.15, 0.15]), (0.0, 0.2))
        obj = lb isa Number ? (x -> (x - 0.05)^2) : (z -> (z[1] - 0.05)^2 + (z[2] - 0.05)^2)
        @test 0.8 * norm(ub .- lb) < 0.2   # the starting scale is out of range here
        Random.seed!(1)
        x = sample(12, lb, ub, SobolSample())
        surr = Kriging(x, obj.(x), lb, ub)
        n_before = length(surr.x)
        result = surrogate_optimize!(
            obj, SRBF(), lb, ub, surr, SobolSample(); maxiters = 20
        )
        check_contract(result, surr, obj, lb, ub, n_before, 20)
        @test length(surr.x) - n_before > 1
    end

    # A resize is reported only when a counter crosses its threshold.
    #                                    scale success failure improved
    @test Surrogates._adjust_trust_region(1.0, 1, 0, true) == (1.0, 2, 0, false)
    @test Surrogates._adjust_trust_region(1.0, 2, 0, true) == (2.0, 0, 0, true)
    @test Surrogates._adjust_trust_region(1.0, 0, 3, false) == (1.0, 0, 4, false)
    @test Surrogates._adjust_trust_region(1.0, 0, 4, false) == (0.5, 0, 0, true)
    # An improvement clears the failure count and vice versa.
    @test Surrogates._adjust_trust_region(1.0, 0, 2, true) == (1.0, 1, 0, false)
    @test Surrogates._adjust_trust_region(1.0, 2, 0, false) == (1.0, 0, 1, false)
end

@testset "parallel ask-tell" begin
    strategies = (
        MinimumConstantLiar(), MaximumConstantLiar(),
        MeanConstantLiar(), KrigingBeliever(),
    )

    @testset "batch contract" begin
        for (dim, lb, ub, obj, n0) in (
                ("1-D", LB1, UB1, f1d, 20),
                ("ND", LB2, UB2, f2d, 20),
            )
            dtol = 1.0e-3 * norm(ub .- lb)
            for alg in (SRBF(), EI()), strategy in strategies
                @testset "$dim $(typeof(alg).name.name) $(typeof(strategy).name.name)" begin
                    Random.seed!(3)
                    x = sample(n0, lb, ub, SobolSample())
                    krig = Kriging(x, obj.(x), lb, ub)
                    before_x, before_y = deepcopy(krig.x), deepcopy(krig.y)

                    points, merits = potential_optimal_points(
                        alg, strategy, lb, ub, krig, SobolSample(), 5;
                        num_new_samples = 200
                    )

                    @test length(points) == 5
                    @test length(merits) == 5
                    @test all(isfinite, merits)
                    @test all(p -> all(lb .- 1.0e-8 .<= coords(p) .<= ub .+ 1.0e-8), points)
                    # A batch is evaluated in parallel, so repeating a point
                    # wastes one of the evaluations it was asked for.
                    @test length(unique(points)) == 5
                    @test minimum(
                        norm(coords(points[i]) .- coords(points[j]))
                            for i in 1:5 for j in (i + 1):5
                    ) > dtol
                    # Virtual points belong to a temporary copy: asking must
                    # never change the surrogate the caller holds.
                    @test krig.x == before_x
                    @test krig.y == before_y
                end
            end
        end
    end

    @testset "believers accumulate within a batch" begin
        # The believed value has to come from the model that already carries
        # the batch's earlier beliefs, not from the untouched original.
        lb, ub = [0.0, 0.0], [10.0, 10.0]
        Random.seed!(3)
        x = sample(20, lb, ub, SobolSample())
        krig = Kriging(x, f2d.(x), lb, ub)
        tmp = deepcopy(krig)
        believed = (5.0, 5.0)
        Surrogates.calculate_liars(KrigingBeliever(), tmp, krig, believed)

        # The belief landed in the temporary surrogate only.
        @test length(tmp.x) == length(krig.x) + 1
        @test believed in tmp.x
        @test !(believed in krig.x)
        # ... and it is the temporary model's own prediction there.
        @test tmp.y[end] ≈ krig(believed) atol = 1.0e-6

        # A second belief nearby must read the updated model, so the two
        # predictions the strategies see are no longer the same.
        near = (5.05, 5.05)
        @test tmp(near) != krig(near)
        @test std_error_at_point(tmp, near) < std_error_at_point(krig, near)
    end
end

@testset "multi-objective" begin
    # `SMB` and `RTEA` return a Pareto set and its front rather than a single
    # best point, so they get their own contract: the two line up, every front
    # entry is a real evaluation, and no member dominates another.
    function check_pareto(pareto_set, pareto_front, obj, lb, ub)
        @test length(pareto_set) == length(pareto_front)
        @test !isempty(pareto_set)
        @test all(p -> all(lb .- 1.0e-8 .<= coords(p) .<= ub .+ 1.0e-8), pareto_set)
        @test all(collect(fp) ≈ collect(obj(p)) for (p, fp) in zip(pareto_set, pareto_front))
        # No member of a Pareto front may dominate another. Aggregated into one
        # assertion: the pairwise sweep is quadratic in the front size.
        front = [collect(fp) for fp in pareto_front]
        dominates(a, b) = all(a .<= b) && any(a .< b)
        @test !any(
            dominates(front[i], front[j])
                for i in eachindex(front) for j in eachindex(front) if i != j
        )
        return nothing
    end

    @testset "SMB 1-D" begin
        Random.seed!(8)
        f = x -> [x^2, x]
        lb, ub = 1.0, 10.0
        x = sample(100, lb, ub, SobolSample())
        surr = RadialBasis(x, f.(x), lb, ub, rad = linearRadial())
        pareto_set, pareto_front = surrogate_optimize!(
            f, SMB(), lb, ub, surr, SobolSample()
        )
        check_pareto(pareto_set, pareto_front, f, lb, ub)
        # The front is drawn from the samples, so it cannot outnumber them.
        @test length(pareto_set) <= length(surr.x)
        @test all(p -> p in surr.x, pareto_set)
    end

    @testset "SMB ND" begin
        Random.seed!(8)
        f = z -> [z[1]^2 + z[2]^2, (z[1] - 2.0)^2 + z[2]^2]
        lb, ub = [0.0, 0.0], [3.0, 3.0]
        x = sample(60, lb, ub, SobolSample())
        surr = RadialBasis(x, f.(x), lb, ub, rad = linearRadial())
        pareto_set, pareto_front = surrogate_optimize!(
            f, SMB(), lb, ub, surr, SobolSample(); maxiters = 10
        )
        check_pareto(pareto_set, pareto_front, f, lb, ub)
        # The two objectives are minimized at (0,0) and (2,0), so a genuine
        # trade-off front has to spread along the segment between them.
        @test length(pareto_set) > 1
        first_coords = [coords(p)[1] for p in pareto_set]
        @test maximum(first_coords) - minimum(first_coords) > 0.1
    end

    @testset "RTEA" begin
        f = x -> [x, sin(x)]
        lb, ub = 1.0, 10.0
        # `k` is the number of re-evaluations and `z` the retained percentage;
        # neither may break the front contract.
        for (k, z) in ((2, 0.8), (1, 0.5), (3, 0.9))
            @testset "k = $k, z = $z" begin
                Random.seed!(8)
                x = sample(500, lb, ub, RandomSample())
                surr = RadialBasis(x, f.(x), lb, ub, rad = linearRadial())
                pareto_set, pareto_front = surrogate_optimize!(
                    f, RTEA(k, z, 0.5, 1.0, 1.5), lb, ub, surr, SobolSample();
                    maxiters = 10
                )
                check_pareto(pareto_set, pareto_front, f, lb, ub)
            end
        end
    end

    @testset "RTEA ND" begin
        # Multidimensional bounds: the samples are stored as `Tuple`s, which
        # both the mutation and the crossover have to handle.
        f = z -> [z[1]^2 + z[2]^2, (z[1] - 2.0)^2 + z[2]^2]
        lb, ub = [0.0, 0.0], [3.0, 3.0]
        Random.seed!(8)
        x = sample(200, lb, ub, RandomSample())
        surr = RadialBasis(x, f.(x), lb, ub, rad = linearRadial())
        pareto_set, pareto_front = surrogate_optimize!(
            f, RTEA(2, 0.5, 0.5, 1.0, 1.5), lb, ub, surr, SobolSample();
            maxiters = 10
        )
        check_pareto(pareto_set, pareto_front, f, lb, ub)
        @test all(p -> length(coords(p)) == 2, pareto_set)
    end

    @testset "RTEA improves the front it was given" begin
        # The Pareto bookkeeping read a tally that was never incremented, so
        # `new_to_pareto` was always false and `still_in_pareto` always false:
        # the front could only ever shrink. The old test could not see it,
        # because the survivors of the attrition are still a valid front.
        f = x -> [x, sin(x)]
        lb, ub = 1.0, 10.0
        Random.seed!(11)
        x = sample(80, lb, ub, RandomSample())
        surr = RadialBasis(x, f.(x), lb, ub, rad = linearRadial())
        pareto_set, pareto_front = surrogate_optimize!(
            f, RTEA(1, 0.2, 0.9, 1.0, 0.8), lb, ub, surr, SobolSample();
            maxiters = 40
        )
        check_pareto(pareto_set, pareto_front, f, lb, ub)
        # At least one member has to be a point the search produced rather than
        # one of the initial samples. This is the assertion that fails outright
        # on the unfixed method, which adds nothing at all.
        @test any(p -> !(p in x), pareto_set)
    end

    @testset "RTEA keeps its children in the box" begin
        # Crossover and mutation were applied with no projection onto the
        # domain. `sigma` here is a third of the domain width, so an unclamped
        # child escapes almost immediately; the surrogate would then be
        # extrapolating and the reported set would violate its own bounds.
        f = x -> [x, (x - 5.0)^2]
        lb, ub = 1.0, 10.0
        Random.seed!(5)
        x = sample(60, lb, ub, RandomSample())
        surr = RadialBasis(x, f.(x), lb, ub, rad = linearRadial())
        pareto_set, _ = surrogate_optimize!(
            f, RTEA(1, 0.1, 1.0, 1.0, 3.0), lb, ub, surr, SobolSample();
            maxiters = 30
        )
        @test all(p -> all(lb .<= coords(p) .<= ub), pareto_set)
        # Everything handed to the surrogate must be inside too, not just what
        # survived into the returned front.
        @test all(p -> all(lb .<= coords(p) .<= ub), surr.x)
    end

    @testset "SMB screens with the surrogate, not the objective" begin
        # The screening sweep called `obj` on every one of `n_new_look`
        # candidates in order to choose the single point at which to call `obj`.
        # At these settings that was 9965 evaluations instead of 10 — the exact
        # cost a surrogate-based method exists to avoid.
        calls = Ref(0)
        base = x -> [x^2, x]
        counted = x -> (calls[] += 1; base(x))
        lb, ub = 1.0, 10.0
        Random.seed!(8)
        x = sample(100, lb, ub, SobolSample())
        surr = RadialBasis(x, base.(x), lb, ub, rad = linearRadial())
        calls[] = 0
        pareto_set, pareto_front = surrogate_optimize!(
            counted, SMB(), lb, ub, surr, SobolSample();
            maxiters = 10, n_new_look = 1000
        )
        @test calls[] <= 10
        check_pareto(pareto_set, pareto_front, base, lb, ub)
    end

    @testset "SMB rejects a candidate pool it would exhaust" begin
        # One candidate is consumed per iteration, so `maxiters > n_new_look`
        # ran `x_to_look` empty and indexed past its end.
        f = x -> [x^2, x]
        lb, ub = 1.0, 10.0
        Random.seed!(8)
        x = sample(40, lb, ub, SobolSample())
        surr = RadialBasis(x, f.(x), lb, ub, rad = linearRadial())
        @test_throws ArgumentError surrogate_optimize!(
            f, SMB(), lb, ub, surr, SobolSample(); maxiters = 20, n_new_look = 5
        )
    end

    # Every surrogate that accepts a vector-valued response has to work with
    # both multi-objective optimizers, in both dimensionalities. Before this,
    # `SMB` and `RTEA` were only ever run against `RadialBasis`, so a surrogate
    # could accept a multi-output design, predict from it correctly, and still
    # be unusable for optimization without anything noticing.
    #
    # The set is exactly the surrogates whose construction accepts
    # `y::Vector{<:AbstractVector}`. `Kriging`, `GEK`, `Wendland`, the PLS
    # models, `EarthSurrogate` and the extension surrogates all reject it, which
    # is the multi-output gap tracked separately — not something these tests can
    # assert around.
    @testset "multi-output surrogates under $(alg_name)" for (alg_name, alg) in (
            ("SMB", SMB()), ("RTEA", RTEA(1, 0.5, 0.5, 1.0, 0.4)),
        )
        @testset "1-D" begin
            g = x -> [x, sin(x)]
            lb, ub = 1.0, 10.0
            Random.seed!(8)
            x = sample(60, lb, ub, RandomSample())
            y = g.(x)
            builders = (
                ("RadialBasis",
                    () -> RadialBasis(x, y, lb, ub, rad = linearRadial())),
                ("Lobachevsky",
                    () -> LobachevskySurrogate(x, y, lb, ub, alpha = 2.0, n = 4)),
                ("SecondOrderPolynomial",
                    () -> SecondOrderPolynomialSurrogate(x, y, lb, ub)),
                ("LinearSurrogate", () -> LinearSurrogate(x, y, lb, ub)),
                ("InverseDistance",
                    () -> InverseDistanceSurrogate(x, y, lb, ub, p = 2.0)),
            )
            @testset "$(name)" for (name, mk) in builders
                surr = mk()
                # The surrogate must predict a vector before it is worth asking
                # an optimizer to rank its outputs; a model that quietly
                # collapsed to a scalar would still "pass" the run below.
                @test surr(x[1]) isa AbstractVector
                @test length(surr(x[1])) == 2

                pareto_set, pareto_front = alg_name == "SMB" ?
                    surrogate_optimize!(g, alg, lb, ub, surr, SobolSample();
                        maxiters = 6, n_new_look = 50) :
                    surrogate_optimize!(g, alg, lb, ub, surr, SobolSample();
                        maxiters = 8)
                check_pareto(pareto_set, pareto_front, g, lb, ub)
            end
        end

        @testset "N-D" begin
            # Two objectives minimized at (0,0) and (2,0), so the front is a
            # genuine trade-off rather than a single point.
            g = z -> [z[1]^2 + z[2]^2, (z[1] - 2.0)^2 + z[2]^2]
            lb, ub = [0.0, 0.0], [3.0, 3.0]
            Random.seed!(8)
            x = sample(60, lb, ub, RandomSample())
            y = g.(x)
            builders = (
                ("RadialBasis",
                    () -> RadialBasis(x, y, lb, ub, rad = linearRadial())),
                ("Lobachevsky",
                    () -> LobachevskySurrogate(x, y, lb, ub,
                        alpha = [2.0, 2.0], n = 4)),
                ("SecondOrderPolynomial",
                    () -> SecondOrderPolynomialSurrogate(x, y, lb, ub)),
                ("LinearSurrogate", () -> LinearSurrogate(x, y, lb, ub)),
                ("InverseDistance",
                    () -> InverseDistanceSurrogate(x, y, lb, ub, p = 2.0)),
            )
            @testset "$(name)" for (name, mk) in builders
                surr = mk()
                @test surr(x[1]) isa AbstractVector
                @test length(surr(x[1])) == 2

                pareto_set, pareto_front = alg_name == "SMB" ?
                    surrogate_optimize!(g, alg, lb, ub, surr, SobolSample();
                        maxiters = 6, n_new_look = 50) :
                    surrogate_optimize!(g, alg, lb, ub, surr, SobolSample();
                        maxiters = 8)
                check_pareto(pareto_set, pareto_front, g, lb, ub)
                @test all(p -> length(coords(p)) == 2, pareto_set)
            end
        end
    end
end

@testset "SectionSample keeps every method on its section" begin
    # `SectionSample` pins a subset of coordinates and searches only the rest.
    # `sample` enforces that itself, so `SRBF`, `EI` and `LCBS` — which draw
    # every candidate through it — are covered by construction. `DYCORS` and
    # `SOP` perturb coordinates *after* sampling, so the pin is theirs to keep,
    # and a violation is a silent wrong answer rather than a crash.
    f = z -> (z[1] - 2.5)^2 + (z[2] - 7.5)^2 + 1.0
    lb, ub = [0.0, 0.0], [10.0, 10.0]
    pinned = 3.0
    make_sampler() = SectionSample([NaN64, pinned], SobolSample())

    @testset "$(name)" for (name, alg) in (
            ("SRBF", SRBF()), ("LCBS", LCBS()), ("EI", EI()),
            ("DYCORS", DYCORS()), ("SOP", SOP(2)),
        )
        Random.seed!(4)
        x = sample(20, lb, ub, make_sampler())
        # The design itself is on the section, so anything off it was put there
        # by the run.
        @test all(p -> coords(p)[2] ≈ pinned, x)

        krig = Kriging(x, f.(x), lb, ub)
        n_before = length(krig.x)
        surrogate_optimize!(f, alg, lb, ub, krig, make_sampler(); maxiters = 6)

        added = krig.x[(n_before + 1):end]
        @test !isempty(added)
        @test all(p -> coords(p)[2] ≈ pinned, added)
        # The free coordinate must still be free: a method that pinned
        # everything would also pass the assertion above.
        @test any(p -> coords(p)[1] != coords(x[1])[1], added)
        @test all(p -> all(lb .- 1.0e-8 .<= coords(p) .<= ub .+ 1.0e-8), added)
    end

    @testset "free dimensions helper" begin
        # Without a section every dimension is perturbable, so the two
        # coordinate-perturbation methods are unchanged for ordinary samplers.
        @test Surrogates._free_dimensions(SobolSample(), 3) == 1:3
        @test Surrogates._free_dimensions(RandomSample(), 1) == 1:1
        @test Surrogates._free_dimensions(
            SectionSample([NaN64, 2.0, NaN64], SobolSample()), 3) == [1, 3]
        @test Surrogates._free_dimensions(
            SectionSample([1.0, 2.0], SobolSample()), 2) == Int[]
    end
end

@testset "Multi-objective invariants" begin
    # Unit-level cover for the pieces the end-to-end runs above exercise only
    # statistically. Each one was silently wrong.

    @testset "SBX spread factor uses 1/(n_c + 1)" begin
        # `(2mu)^(1 / n_c + 1)` parses as `(2mu)^((1/n_c) + 1)`, a different
        # exponent for every `n_c`.
        for n_c in (1.0, 2.0, 5.0, 20.0)
            e = 1 / (n_c + 1)
            for mu in (0.0, 0.1, 0.25, 0.5)
                @test Surrogates._sbx_beta(mu, n_c) ≈ (2 * mu)^e
            end
            for mu in (0.6, 0.75, 0.9, 0.99)
                @test Surrogates._sbx_beta(mu, n_c) ≈ (1 / (2 * (1 - mu)))^e
            end
            # `mu = 0.5` is the contact point: the child coincides with a
            # parent, so beta is 1 for every distribution index.
            @test Surrogates._sbx_beta(0.5, n_c) ≈ 1.0
        end
        # A larger distribution index concentrates offspring nearer the parents,
        # i.e. beta closer to 1. The old expression had this backwards.
        @test abs(Surrogates._sbx_beta(0.1, 20.0) - 1.0) <
            abs(Surrogates._sbx_beta(0.1, 1.0) - 1.0)
    end

    @testset "SBX child works for scalar and tuple parents" begin
        # `p_cross = 0.0` takes the no-crossover branch, `1.0` forces crossover.
        @test Surrogates._sbx_child(1.0, 2.0, 0.0, 1.0) == 2.0
        @test Surrogates._sbx_child((1.0, 1.0), (2.0, 2.0), 0.0, 1.0) == (2.0, 2.0)
        Random.seed!(3)
        child = Surrogates._sbx_child((0.0, 0.0), (2.0, 4.0), 1.0, 2.0)
        @test child isa Tuple && length(child) == 2
        @test all(isfinite, child)
        # Crossing a parent with itself reproduces it, whatever beta is drawn.
        for _ in 1:20
            @test all(Surrogates._sbx_child((1.5, -2.0), (1.5, -2.0), 1.0, 3.0) .≈
                (1.5, -2.0))
        end
    end

    @testset "Pareto insertion accepts and evicts by dominance" begin
        # Minimizing both objectives.
        set = Any[1.0, 2.0]
        front = Any[[0.0, 4.0], [4.0, 0.0]]
        n_re = Int[3, 7]

        # Dominated by [0,4]: rejected, and nothing is disturbed.
        @test Surrogates._offer_to_pareto!(set, front, n_re, 9.0, [1.0, 5.0]) == false
        @test length(set) == 2 && n_re == [3, 7]

        # Non-dominated and dominating neither: accepted, front grows.
        @test Surrogates._offer_to_pareto!(set, front, n_re, 3.0, [2.0, 2.0]) == true
        @test set == Any[1.0, 2.0, 3.0]
        @test n_re == [3, 7, 0]

        # Dominates [4,0] and [2,2]: both evicted along with their tallies.
        @test Surrogates._offer_to_pareto!(set, front, n_re, 5.0, [1.0, 0.0]) == true
        @test set == Any[1.0, 5.0]
        @test front == Any[[0.0, 4.0], [1.0, 0.0]]
        @test n_re == [3, 0]

        # A duplicate of an existing member is not dominated by it (dominance is
        # strict in at least one objective), so it is admitted rather than
        # silently dropped.
        @test Surrogates._offer_to_pareto!(set, front, n_re, 1.0, [0.0, 4.0]) == true
    end

    @testset "re-evaluation is judged against the other members" begin
        front = Any[[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]]
        # A member compared against the whole front would meet itself; the test
        # must skip its own index.
        @test Surrogates._survives_revaluation(front, 2, [2.0, 2.0]) == true
        # Re-evaluated to something [0,4] dominates: it must go.
        @test Surrogates._survives_revaluation(front, 2, [1.0, 5.0]) == false
        # Re-evaluated to something better than before: it stays.
        @test Surrogates._survives_revaluation(front, 2, [1.0, 1.0]) == true
        # Single-member front: nothing can dominate it.
        @test Surrogates._survives_revaluation(Any[[5.0, 5.0]], 1, [9.0, 9.0]) == true
    end

    @testset "tier-II ranking orders by response, not by coordinate" begin
        # `sortperm(D[i])` sorted the points themselves — by coordinate, and
        # lexicographically over the tuple in ND — which says nothing about
        # which center is more promising.
        lb, ub = [0.0, 0.0], [4.0, 4.0]
        pts = [(3.0, 3.0), (1.0, 1.0), (2.0, 2.0)]
        # Response decreasing in the coordinate, so ranking by value reverses
        # the coordinate order and the two criteria cannot be confused.
        vals = [1.0, 9.0, 5.0]
        surr = RadialBasis(pts, vals, lb, ub, rad = linearRadial())
        D = Dict(1 => copy(pts))
        ranked = Surrogates.II_tier_ranking_ND(D, surr)
        @test ranked[1] == [(3.0, 3.0), (2.0, 2.0), (1.0, 1.0)]

        # Same in one dimension.
        surr1 = RadialBasis([3.0, 1.0, 2.0], [1.0, 9.0, 5.0], 0.0, 4.0,
            rad = linearRadial())
        D1 = Dict(1 => [3.0, 1.0, 2.0])
        @test Surrogates.II_tier_ranking_1D(D1, surr1)[1] == [3.0, 2.0, 1.0]
    end
end

# Each of these pins an invariant that the convergence checks above cannot see:
# every one of them passed while the behaviour it names was broken.
@testset "Optimization invariants" begin
    @testset "merit function rewards distance" begin
        lb, ub = 0.0, 15.0
        x = sample(20, lb, ub, SobolSample())
        k = Kriging(x, (t -> (t - 4.0)^2).(x), lb, ub)
        grid = collect(range(lb, ub, length = 50))
        s = k.(grid)
        s_max, s_min, d_max, d_min = Surrogates._merit_ranges(k, grid)
        w = 0.3
        merit = [
            Surrogates.merit_function(g, w, k, s_max, s_min, d_max, d_min) for g in grid
        ]
        # Subtracting the response term leaves the distance term, which has to
        # vary across the pool or the merit ranks candidates on the surrogate
        # alone and the search never explores.
        exploration = merit .- w .* (s .- s_min) ./ (s_max - s_min)
        @test maximum(exploration) - minimum(exploration) > 1.0e-3
        # Both criteria are rescaled onto [0, 1], so the score is too.
        @test all(0.0 .<= merit .<= 1.0)
        # A pool with no spread scores 1 on both criteria (Regis and Shoemaker).
        flat = fill(x[1], 5)
        @test all(Surrogates._merit_ranges(k, flat) .>= 0.0)
    end

    @testset "SRBF keeps accepted points apart" begin
        lb, ub = [0.0, 0.0], [10.0, 10.0]
        f = z -> (z[1] - 2.5)^2 + (z[2] - 7.5)^2
        x = sample(20, lb, ub, SobolSample())
        k = Kriging(x, f.(x), lb, ub)
        surrogate_optimize!(f, SRBF(), lb, ub, k, SobolSample(); maxiters = 20)
        dtol = 1.0e-3 * norm(ub - lb)
        closest = minimum(
            norm(collect(k.x[i]) .- collect(k.x[j]))
                for i in eachindex(k.x) for j in (i + 1):length(k.x)
        )
        # The proximity filter has to see the points the run itself adds.
        @test closest > dtol
        # ... and one iteration must cost one evaluation, not `maxiters` of them.
        @test length(k.x) <= 20 + 20
    end

    @testset "SRBF works with a surrogate that has no duplicate guard" begin
        lb, ub = [0.0, 0.0], [10.0, 10.0]
        f = z -> (z[1] - 2.5)^2 + (z[2] - 7.5)^2
        x = sample(25, lb, ub, SobolSample())
        r = RadialBasis(x, f.(x), lb, ub, rad = cubicRadial())
        # Duplicate points make the RBF interpolation matrix singular; this
        # threw `SingularException` on every seed.
        @test surrogate_optimize!(
            f, SRBF(), lb, ub, r, SobolSample(); maxiters = 12
        ) isa Tuple
    end

    @testset "LCBS is a lower confidence bound" begin
        lb, ub = 0.0, 10.0
        f = t -> (t - 4.0)^2
        x = sample(10, lb, ub, SobolSample())
        k = Kriging(x, f.(x), lb, ub)
        # Away from the data the bound must sit *below* the mean; the upper
        # bound was being minimised instead, which penalises exploration.
        t = 7.3
        @test k(t) - 2 * std_error_at_point(k, t) < k(t)

        k1 = Kriging(x, f.(x), lb, ub)
        surrogate_optimize!(f, LCBS(), lb, ub, k1, SobolSample(); maxiters = 6)
        # The 1-D method used to store the acquisition value as if it were an
        # observation, so the surrogate no longer interpolated the objective.
        @test all(isapprox(k1.y[i], f(k1.x[i]); atol = 1.0e-8) for i in eachindex(k1.x))
    end

    @testset "DYCORS perturbs around the incumbent" begin
        lb, ub = 0.0, 10.0
        f = t -> (t - 4.0)^2
        x = sample(12, lb, ub, SobolSample())
        k = Kriging(x, f.(x), lb, ub)
        surrogate_optimize!(f, DYCORS(), lb, ub, k, SobolSample(); maxiters = 20)
        # `x_best` was `argmin(surr.y)` -- an index -- so candidates were drawn
        # around a sample number. Every point must stay inside the box.
        @test all(lb <= xi <= ub for xi in k.x)
        @test minimum(k.y) <= minimum(f.(x))
    end

    @testset "DYCORS perturbation probability stays positive" begin
        # `(1 - log(k)) / log(maxiters - 1)` went negative from k = 3, after
        # which no coordinate was ever selected.
        maxiters, d = 100, 5
        p = [min(20 / d, 1) * (1 - log(k) / log(max(maxiters, 2))) for k in 1:maxiters]
        @test all(p .>= 0)
        @test issorted(p, rev = true)
    end

    @testset "DYCORS weights cycle over iterations" begin
        # Samples clustered low, response decreasing: the candidate at 2.0 is
        # close to the data and looks good to the surrogate, the one at 9.0 is
        # far away and looks worse. Iteration 1 takes w_nR = 0.95 and must
        # exploit; iteration 2 takes w_nR = 0.3 and must explore.
        lb, ub = 0.0, 10.0
        x = collect(range(0.0, 2.0, length = 8))
        k = Kriging(x, x, lb, ub)
        candidates = [2.0, 9.0]
        exploit = Surrogates.select_evaluation_point(candidates, k, 1)
        explore = Surrogates.select_evaluation_point(candidates, k, 2)
        @test exploit == 2.0
        @test explore == 9.0
        # Reading the weight off `maxiters` made both calls return the same point.
        @test exploit != explore
    end

    @testset "Pareto dominance is antisymmetric" begin
        lb, ub = 0.0, 10.0
        f = t -> (t - 4.0)^2
        x = sample(8, lb, ub, SobolSample())
        k = Kriging(x, f.(x), lb, ub)
        fronts = Surrogates.I_tier_ranking_1D(copy(k.x), k)
        # `q_dominates_p` was a verbatim copy of `p_dominates_q`, so the front
        # order was inverted and front 1 collapsed to a single point.
        @test !isempty(fronts)
        @test sum(length, values(fronts)) == length(k.x)
    end

    @testset "hypervolume improvement" begin
        P = [0.5 0.5; 0.7 0.3]
        # More than one row used to throw `DimensionMismatch`.
        @test Surrogates.Hypervolume_Pareto_improving(1.0, 2.0, P) == 0.0
        @test Surrogates.Hypervolume_Pareto_improving(0.9, 0.6, P) == 0.0
        @test Surrogates.Hypervolume_Pareto_improving(0.6, 0.4, P) > 0.0
        # A point that dominates the whole set must gain more than one that
        # merely extends it.
        @test Surrogates.Hypervolume_Pareto_improving(0.1, 0.1, P) >
            Surrogates.Hypervolume_Pareto_improving(0.6, 0.4, P)
    end

    @testset "parallel batches contain no repeats" begin
        lb, ub = [0.0, 0.0], [10.0, 10.0]
        f = z -> (z[1] - 2.5)^2 + (z[2] - 7.5)^2
        x = sample(20, lb, ub, SobolSample())
        strategies = (
            MinimumConstantLiar(), MaximumConstantLiar(),
            MeanConstantLiar(), KrigingBeliever(),
        )
        # Points already chosen for the batch live in the temporary surrogate,
        # so the proximity filter has to run against that -- as the 1-D SRBF
        # and the EI methods already did. The ND SRBF method filtered against
        # the original surrogate and returned the same point twice for three
        # of these four strategies.
        for alg in (SRBF(), EI()), strategy in strategies
            k = Kriging(x, f.(x), lb, ub)
            pts, merits = potential_optimal_points(
                alg, strategy, lb, ub, k, SobolSample(), 6; num_new_samples = 200
            )
            @test length(unique(pts)) == 6
            @test length(merits) == 6
        end
    end

    @testset "EI never re-selects a sampled point" begin
        lb, ub = 0.0, 10.0
        f = t -> (t - 4.0)^2
        x = sample(10, lb, ub, SobolSample())
        k = Kriging(x, f.(x), lb, ub)
        surrogate_optimize!(f, EI(), lb, ub, k, SobolSample(); maxiters = 8)
        dtol = 1.0e-3 * norm(ub - lb)
        closest = minimum(
            abs(k.x[i] - k.x[j])
                for i in eachindex(k.x) for j in (i + 1):length(k.x)
        )
        # At a sampled point the predictive variance is zero, so EI must be
        # zero there. Falling back to `z = 0` returned half the remaining gap
        # instead, which keeps already-evaluated points competitive.
        @test closest > dtol
        @test all(isapprox(k.y[i], f(k.x[i]); atol = 1.0e-8) for i in eachindex(k.x))
    end
end

# Responses that correspond one-to-one with the stored samples. `GEK` appends the
# observed gradients to its response vector, so reducing over the whole of `y`
# there compares a directional derivative against a function value.
sample_responses(surr) = view(surr.y, 1:length(surr.x))

@testset "gradient-enhanced surrogates" begin
    # `GEK` holds `[values; gradients]`, so `argmin(y)` used to land in the
    # gradient block: `_best_point` indexed past the samples and every method
    # that asks for the incumbent raised a `BoundsError`.
    @testset "GEK y is twice as long as x" begin
        x = collect(range(0.4, 9.6, length = 20))
        grads = Zygote.gradient.(f1d, x)
        gek = GEK(
            x, vcat(f1d.(x), [g[1] for g in grads]), LB1, UB1;
            optimize_theta = false
        )
        @test length(gek.y) == 2 * length(gek.x)
        # The whole vector reaches into the gradients, which are negative here.
        @test minimum(gek.y) < MIN1
        # The response block does not.
        @test minimum(sample_responses(gek)) >= MIN1
    end

    @testset "every method reaches the minimum with needs_gradient" begin
        for (name, method) in [
                ("SRBF", SRBF()), ("LCBS", LCBS()), ("EI", EI()),
                ("DYCORS", DYCORS()), ("SOP", SOP(3)),
            ]
            Random.seed!(7)
            x = collect(range(0.4, 9.6, length = 20))
            grads = Zygote.gradient.(f1d, x)
            gek = GEK(
                x, vcat(f1d.(x), [g[1] for g in grads]), LB1, UB1;
                optimize_theta = false
            )
            xbest, ybest = surrogate_optimize!(
                f1d, method, LB1, UB1, gek, SobolSample();
                maxiters = 15, num_new_samples = 20, needs_gradient = true
            )
            # A gradient can never be returned as the incumbent value.
            @test ybest ≈ minimum(sample_responses(gek))
            @test isapprox(ybest, f1d(xbest); atol = 1.0e-8)
            @test LB1 - 1.0e-8 <= xbest <= UB1 + 1.0e-8
            @test ybest >= MIN1 - 1.0e-8
        end
    end

    # `Zygote.gradient` returns one entry per argument of the objective, so the
    # gradient is that lone entry. Passing the wrapper through gave one
    # "derivative" per point, which is the right count only in one dimension.
    @testset "gradients are unwrapped in ND too" begin
        Random.seed!(7)
        x = [
            (a, b) for a in range(0.4, 9.6, length = 6)
                for b in range(0.4, 9.6, length = 6)
        ]
        grads = Zygote.gradient.(f2d, x)
        gek = GEK(
            x, vcat(f2d.(x), reduce(vcat, [collect(g[1]) for g in grads])),
            LB2, UB2; optimize_theta = false
        )
        xbest, ybest = surrogate_optimize!(
            f2d, SRBF(), LB2, UB2, gek, SobolSample();
            maxiters = 10, num_new_samples = 20, needs_gradient = true
        )
        @test isapprox(ybest, f2d(xbest); atol = 1.0e-8)
        @test all(LB2 .- 1.0e-8 .<= coords(xbest) .<= UB2 .+ 1.0e-8)
    end
end

@testset "predictive standard deviation across the Kriging family" begin
    # A design with a dense cluster and a wide gap, so the variance has to have
    # real structure rather than being numerically flat.
    xs = [0.5, 1.0, 1.5, 2.0, 8.5, 9.0]
    ys = f1d.(xs)
    grads = Zygote.gradient.(f1d, xs)
    models = [
        ("Kriging", Kriging(xs, ys, LB1, UB1)),
        ("KPLS", KPLS(xs, ys, 1, [LB1], [UB1], [0.1]; optimize_theta = false)),
        ("KPLSK", KPLSK(xs, ys, 1, [LB1], [UB1], [0.1]; optimize_theta = false)),
        ("GEKPLS", GEKPLS(xs, ys, grads, 1, 1.0e-4, [LB1], [UB1], 1, [0.1])),
        (
            "GEK",
            GEK(
                xs, vcat(ys, [g[1] for g in grads]), LB1, UB1;
                optimize_theta = false
            ),
        ),
    ]
    for (name, model) in models
        @testset "$name" begin
            grid = range(LB1, UB1, length = 100)
            values = [std_error_at_point(model, t) for t in grid]
            @test all(isfinite, values)
            @test all(>=(0), values)
            # Least known in the middle of the gap, best known inside the cluster.
            @test std_error_at_point(model, 5.25) >
                std_error_at_point(model, 1.25)
        end
    end
end

@testset "variance-based methods span the stochastic surrogates" begin
    xs = collect(range(0.4, 9.6, length = 20))
    ys = f1d.(xs)
    grads = Zygote.gradient.(f1d, xs)
    builders = [
        "Kriging" => () -> Kriging(xs, ys, LB1, UB1),
        "KPLS" => () -> KPLS(xs, ys, 1, [LB1], [UB1], [1.0]; optimize_theta = false),
        "KPLSK" => () -> KPLSK(xs, ys, 1, [LB1], [UB1], [1.0]; optimize_theta = false),
        "GEKPLS" => () -> GEKPLS(xs, ys, grads, 1, 1.0e-4, [LB1], [UB1], 1, [0.01]),
    ]
    for (name, build) in builders, (mname, method) in [("EI", EI()), ("LCBS", LCBS())]
        @testset "$mname with $name" begin
            Random.seed!(11)
            surr = build()
            n_before = length(surr.x)
            result = surrogate_optimize!(
                f1d, method, LB1, UB1, surr, SobolSample();
                maxiters = 10, num_new_samples = 20
            )
            check_contract(result, surr, f1d, LB1, UB1, n_before, 10)
        end
    end

    # A surrogate with no variance model cannot score these acquisitions. The
    # failure used to surface as a `MethodError` on `std_error_at_point` from
    # deep inside the loop.
    @testset "deterministic surrogates are rejected up front" begin
        for (name, build) in [
                "RadialBasis" => () -> RadialBasis(xs, ys, LB1, UB1),
                "LinearSurrogate" => () -> LinearSurrogate(xs, ys, LB1, UB1),
                "Wendland" => () -> Wendland(xs, ys, LB1, UB1),
            ]
            for method in (EI(), LCBS())
                @test_throws ArgumentError surrogate_optimize!(
                    f1d, method, LB1, UB1, build(), SobolSample(); maxiters = 5
                )
            end
        end
    end
end

@testset "ask-tell coverage" begin
    xs = collect(range(0.4, 9.6, length = 20))
    ys = f1d.(xs)

    @testset "EI batches honour the caller's bounds" begin
        # `potential_optimal_points(::EI, ...)` opened with `lb = krig.lb;
        # ub = krig.ub`, throwing away the bounds the caller passed. Restricting
        # the batch to a sub-box is the whole point of passing them.
        Random.seed!(13)
        krig = Kriging(xs, ys, LB1, UB1)
        sub_lb, sub_ub = 6.0, 9.0
        points, _ = potential_optimal_points(
            EI(), MinimumConstantLiar(), sub_lb, sub_ub, krig,
            SobolSample(), 3; num_new_samples = 40
        )
        @test !isempty(points)
        @test all(p -> sub_lb - 1.0e-8 <= p <= sub_ub + 1.0e-8, points)
        # The sub-box excludes the global incumbent, so this also pins that the
        # restriction actually bit rather than coinciding with the full box.
        @test all(p -> p > LB1 + 1.0e-8, points)
    end

    @testset "EI batches work for surrogates with no lb/ub fields" begin
        # The same two lines assumed a field layout that the PLS-based models do
        # not have: they hold their design in standardized units and expose no
        # `lb`, so every batch call died with `type KPLS has no field lb`.
        for (name, mk) in (
                ("KPLS", () -> KPLS(xs, ys, 1, [LB1], [UB1], [1.0];
                    optimize_theta = false)),
                ("KPLSK", () -> KPLSK(xs, ys, 1, [LB1], [UB1], [1.0];
                    optimize_theta = false)),
            )
            @testset "$name" begin
                Random.seed!(13)
                surr = mk()
                points, scores = potential_optimal_points(
                    EI(), MinimumConstantLiar(), LB1, UB1, surr,
                    SobolSample(), 3; num_new_samples = 40
                )
                @test length(points) == length(scores)
                @test all(p -> LB1 - 1.0e-8 <= p <= UB1 + 1.0e-8, points)
                @test length(surr.x) == length(xs)
            end
        end
    end

    @testset "EI batches reject a surrogate with no variance" begin
        # `surrogate_optimize!(::EI, ...)` and the `LCBS` batch method both
        # check up front; the `EI` batch method did not, and failed several
        # candidates deep with a raw `MethodError` on `std_error_at_point`.
        for mk in (
                () -> RadialBasis(xs, ys, LB1, UB1, rad = linearRadial()),
                () -> Wendland(xs, ys, LB1, UB1),
            )
            @test_throws ArgumentError potential_optimal_points(
                EI(), MinimumConstantLiar(), LB1, UB1, mk(), SobolSample(), 2
            )
        end
    end

    @testset "supported algorithms return a batch" begin
        for (name, method) in [("SRBF", SRBF()), ("EI", EI()), ("LCBS", LCBS())]
            @testset "$name" begin
                Random.seed!(13)
                krig = Kriging(xs, ys, LB1, UB1)
                points, scores = potential_optimal_points(
                    method, MinimumConstantLiar(), LB1, UB1, krig,
                    SobolSample(), 3; num_new_samples = 40
                )
                @test length(points) == length(scores)
                @test all(p -> LB1 - 1.0e-8 <= p <= UB1 + 1.0e-8, points)
                @test allunique(points)
                # The surrogate the caller passed in is never modified.
                @test length(krig.x) == length(xs)
            end
        end
    end

    # `DYCORS` schedules its perturbation probability on the iteration index, and
    # `SOP` already evaluates several centers per iteration, so neither has an
    # ask-tell form. Saying so beats a `MethodError`.
    @testset "unsupported algorithms say so" begin
        krig = Kriging(xs, ys, LB1, UB1)
        for method in (DYCORS(), SOP(3))
            @test_throws ArgumentError potential_optimal_points(
                method, MinimumConstantLiar(), LB1, UB1, krig, SobolSample(), 2
            )
        end
    end

    # The believers need a predictive standard deviation, not a `Kriging`
    # specifically; they used to be typed to `Kriging` and so refused `GEK`.
    @testset "believer strategies accept any stochastic surrogate" begin
        grads = Zygote.gradient.(f1d, xs)
        for strategy in (
                KrigingBeliever(), KrigingBelieverUpperBound(),
                KrigingBelieverLowerBound(),
            )
            Random.seed!(17)
            gek = GEK(
                xs, vcat(ys, [g[1] for g in grads]), LB1, UB1;
                optimize_theta = false
            )
            points, _ = potential_optimal_points(
                SRBF(), strategy, LB1, UB1, gek, SobolSample(), 2;
                num_new_samples = 40
            )
            @test length(points) == 2
            @test all(p -> LB1 - 1.0e-8 <= p <= UB1 + 1.0e-8, points)
        end
    end
end
