"""
    SurrogateOptimizationAlgorithm

Abstract interface for surrogate optimization strategies used by
[`surrogate_optimize!`](@ref).

Concrete subtypes select how new candidate points are generated and evaluated
against an existing surrogate. A subtype is used as a dispatch token:

```julia
surrogate_optimize!(objective, SRBF(), lb, ub, surrogate, sample_type)
```

# Interface

A concrete `alg <: SurrogateOptimizationAlgorithm` is valid when
`surrogate_optimize!(objective, alg, lb, ub, surrogate, sample_type; kwargs...)`
is implemented for the surrogate and sampling types it supports. Implementations
may mutate `surrogate` by calling `update!` with newly evaluated points.

# Arguments

  - `objective::Function`: expensive objective function to minimize.
  - `lb`: lower bound of the search domain.
  - `ub`: upper bound of the search domain.
  - `surrogate`: fitted surrogate satisfying the [`AbstractSurrogate`](@ref)
    evaluation and `update!` interface.
  - `sample_type::SamplingAlgorithm`: sampling strategy used to generate
    candidate points.
"""
abstract type SurrogateOptimizationAlgorithm end

"""
    ParallelStrategy

Abstract interface for virtual-point strategies used by
[`potential_optimal_points`](@ref).

Concrete subtypes define how a temporary surrogate is updated while selecting a
batch of parallel candidate points. A strategy is used only through
`potential_optimal_points(alg, strategy, lb, ub, surrogate, sample_type, n)`.

# Interface

A subtype `strategy <: ParallelStrategy` must be supported by a
`calculate_liars(strategy, tmp_surrogate, surrogate, x_new)` method. The method
updates `tmp_surrogate` with a virtual objective value at `x_new` without
evaluating the true objective.
"""
abstract type ParallelStrategy end

"""
    AbstractSurrogate

Union alias for deterministic and stochastic surrogate models accepted by the
generic Surrogates.jl interfaces.

Concrete surrogates are expected to satisfy the SurrogatesBase interface:

  - `surrogate(x)` evaluates the fitted approximation at `x`.
  - `update!(surrogate, x_new, y_new)` incorporates one or more new observations.
  - sample storage is available through `surrogate.x` and `surrogate.y` for the
    optimization routines in this package.

This alias is a convenience type for method signatures. New surrogate
implementations should subtype `SurrogatesBase.AbstractDeterministicSurrogate`
or `SurrogatesBase.AbstractStochasticSurrogate`, not this union alias.
"""
const AbstractSurrogate = Union{AbstractDeterministicSurrogate, AbstractStochasticSurrogate}

"""
    KrigingBeliever()

Virtual-point strategy that uses the Kriging surrogate prediction as the
temporary objective value for each selected parallel point.

# Interface

`KrigingBeliever` is passed to [`potential_optimal_points`](@ref). It requires a
Kriging surrogate with callable prediction and `update!` support.
"""
struct KrigingBeliever <: ParallelStrategy end

"""
    KrigingBelieverUpperBound()

Virtual-point strategy that updates the temporary Kriging surrogate with an
upper-confidence value at each selected parallel point.

# Interface

Use this strategy with [`potential_optimal_points`](@ref) when optimistic
batching should account for Kriging uncertainty through
`prediction + std_error_at_point(...)`.
"""
struct KrigingBelieverUpperBound <: ParallelStrategy end

"""
    KrigingBelieverLowerBound()

Virtual-point strategy that updates the temporary Kriging surrogate with a
lower-confidence value at each selected parallel point.

# Interface

Use this strategy with [`potential_optimal_points`](@ref) when batching should
favor exploitation through `prediction - std_error_at_point(...)`.
"""
struct KrigingBelieverLowerBound <: ParallelStrategy end

"""
    MinimumConstantLiar()

Virtual-point strategy that inserts the current minimum observed surrogate value
for each selected parallel point.

# Interface

`MinimumConstantLiar` is used with [`potential_optimal_points`](@ref) and
requires the surrogate to expose observed responses through `surrogate.y`.
"""
struct MinimumConstantLiar <: ParallelStrategy end

"""
    MaximumConstantLiar()

Virtual-point strategy that inserts the current maximum observed surrogate value
for each selected parallel point.

# Interface

`MaximumConstantLiar` is used with [`potential_optimal_points`](@ref) and
requires the surrogate to expose observed responses through `surrogate.y`.
"""
struct MaximumConstantLiar <: ParallelStrategy end

"""
    MeanConstantLiar()

Virtual-point strategy that inserts the mean observed surrogate value for each
selected parallel point.

# Interface

`MeanConstantLiar` is used with [`potential_optimal_points`](@ref) and requires
the surrogate to expose observed responses through `surrogate.y`.
"""
struct MeanConstantLiar <: ParallelStrategy end

#single objective optimization
"""
    SRBF()

Surrogate optimization marker for the stochastic radial-basis-function search
strategy.

# Usage

```julia
surrogate_optimize!(objective, SRBF(), lb, ub, surrogate, sample_type)
```

# Interface

The surrogate must implement `surrogate(x)` and `update!(surrogate, x, y)`, and
must store existing samples in `surrogate.x` and `surrogate.y`.
"""
struct SRBF <: SurrogateOptimizationAlgorithm end

"""
    LCBS()

Surrogate optimization marker for lower-confidence-bound search.

# Usage

```julia
surrogate_optimize!(objective, LCBS(), lb, ub, kriging_surrogate, sample_type)
```

# Interface

The surrogate must provide `std_error_at_point(surrogate, x)` in addition to the
generic surrogate evaluation and `update!` interface.
"""
struct LCBS <: SurrogateOptimizationAlgorithm end

"""
    EI()

Surrogate optimization marker for expected-improvement search.

# Usage

```julia
surrogate_optimize!(objective, EI(), lb, ub, kriging_surrogate, sample_type)
```

# Interface

The surrogate must provide `std_error_at_point(surrogate, x)` and the generic
surrogate evaluation and `update!` interface.
"""
struct EI <: SurrogateOptimizationAlgorithm end

"""
    DYCORS()

Surrogate optimization marker for dynamic coordinate search.

# Usage

```julia
surrogate_optimize!(objective, DYCORS(), lb, ub, surrogate, sample_type)
```

# Interface

The surrogate must implement `surrogate(x)`, `update!(surrogate, x, y)`, and
sample storage through `surrogate.x` and `surrogate.y`.
"""
struct DYCORS <: SurrogateOptimizationAlgorithm end

"""
    SOP(p)

Surrogate optimization marker for the candidate-ranking strategy used by the
second-order polynomial optimizer.

# Fields

  - `p`: number of search centers carried, and so the number of candidate
    points proposed per iteration.

# Usage

```julia
surrogate_optimize!(objective, SOP(2), lb, ub, surrogate, sample_type)
```
"""
struct SOP{P} <: SurrogateOptimizationAlgorithm
    p::P
end

#multi objective optimization
"""
    SMB()

Surrogate optimization marker for the surrogate-model-based multi-objective
optimizer.

# Usage

```julia
surrogate_optimize!(objective, SMB(), lb, ub, surrogate, sample_type)
```
"""
struct SMB <: SurrogateOptimizationAlgorithm end

"""
    RTEA(k, z, p, n_c, sigma)

Surrogate optimization marker for the radial basis trust-region evolutionary
algorithm.

# Fields

  - `k`: trust-region or candidate-count parameter used by the RTEA method.
  - `z`: target or reference value used by the method.
  - `p`: polynomial or distance parameter used by the method.
  - `n_c`: candidate count or population size.
  - `sigma`: perturbation scale.

# Usage

```julia
surrogate_optimize!(objective, RTEA(k, z, p, n_c, sigma), lb, ub, surrogate, sample_type)
```
"""
struct RTEA{K, Z, P, N, S} <: SurrogateOptimizationAlgorithm
    k::K
    z::Z
    p::P
    n_c::N
    sigma::S
end

# Smallest separation two evaluated points are allowed to have, as a fraction
# of the domain diameter. Candidates closer than this to an existing sample are
# discarded: they buy almost no information and, for an interpolating
# surrogate, make the interpolation matrix singular.
_candidate_tolerance(lb, ub) = 1.0e-3 * norm(ub .- lb)

# The best observation held by a surrogate, as the `(point, value)` pair every
# single-objective method returns.
function _best_point(surr::AbstractSurrogate)
    index = argmin(surr.y)
    return (surr.x[index], surr.y[index])
end

# Distance from `point` to the nearest sample of `surr`.
_nearest_sample_distance(surr::AbstractSurrogate, point) =
    minimum(norm(x .- point) for x in surr.x)

# Rescale `value` onto `[0, 1]` across `[lo, hi]`. A range too narrow to divide
# by scores `1`, the convention both criteria use in Regis and Shoemaker (2007).
function _unit_score(value, lo, hi)
    span = hi - lo
    return span <= 1.0e-6 ? one(float(value)) : (value - lo) / span
end

"""
    merit_function(point, w, surr, s_max, s_min, d_max, d_min)

Weighted score of a candidate point, to be minimized.

Following Regis and Shoemaker (2007), the score combines two criteria, each
rescaled onto `[0, 1]` over the candidate pool: the surrogate prediction
`surr(point)` against the pool's range `[s_min, s_max]`, and the distance from
`point` to the nearest evaluated sample against the pool's range
`[d_min, d_max]`, inverted so that an isolated candidate scores low. The weight
`w` trades one against the other, and cycling `w` over successive iterations is
what alternates the search between exploitation and exploration.

## References

Regis, R.G. and Shoemaker, C.A. (2007). A stochastic radial basis function
method for the global optimization of expensive functions. *INFORMS Journal on
Computing*, 19(4), 497-509.
"""
function merit_function(point, w, surr::AbstractSurrogate, s_max, s_min, d_max, d_min)
    D_x = _nearest_sample_distance(surr, point)
    return w * _unit_score(surr(point), s_min, s_max) +
        (1 - w) * _unit_score(d_max - D_x, zero(d_max - D_x), d_max - d_min)
end

# Cyclic weight pattern of Regis and Shoemaker (2007), shared by SRBF and
# DYCORS: mostly exploratory at 0.3, almost purely greedy at 0.95.
const _SRBF_WEIGHTS = (0.3, 0.5, 0.8, 0.95)

# Trust region of half-width `3 * scale * ||incumbent - bound||` around the
# incumbent, clipped to the domain.
function _trust_region(incumbent_x, lb::Number, ub::Number, scale)
    new_lb = max(lb, incumbent_x - 3 * scale * norm(incumbent_x - lb))
    new_ub = min(ub, incumbent_x + 3 * scale * norm(incumbent_x - ub))
    return new_lb, new_ub
end

function _trust_region(incumbent_x, lb, ub, scale)
    new_lb = incumbent_x .- 3 * scale * norm(incumbent_x .- lb)
    new_ub = incumbent_x .+ 3 * scale * norm(incumbent_x .- ub)
    return vec(max.(new_lb, lb)), vec(min.(new_ub, ub))
end

# Trust-region schedule: double the region after three consecutive improvements,
# halve it after five consecutive failures, and reset the counters either way.
# The fourth return value says whether the width moved, so a caller tests it
# against its usable range only when there is something new to test -- the
# starting width is not itself a reason to stop.
function _adjust_trust_region(scale, success, failure, improved)
    if improved
        success, failure = success + 1, 0
    else
        success, failure = 0, failure + 1
    end
    if success == 3
        return 2 * scale, 0, 0, true
    elseif failure == 5
        return scale / 2, 0, 0, true
    end
    return scale, success, failure, false
end

# Expected improvement of Jones, Schonlau and Welch (1998) at `point`, with the
# exploration offset `xi`. Exactly zero where the surrogate carries no
# predictive variance: the value there is already known, so nothing can be
# gained by evaluating it again.
function _expected_improvement(krig, point, f_min, xi = 0.01)
    sigma = std_error_at_point(krig, point)
    abs(sigma) <= 1.0e-6 && return zero(float(f_min))
    improvement = f_min - krig(point) - xi
    z = improvement / sigma
    return improvement * cdf(Normal(), z) + sigma * pdf(Normal(), z)
end

# A sampled candidate in the form `update!` and the objective expect: a scalar
# stays a scalar, a coordinate vector becomes a tuple.
_as_new_sample(x::Number) = x
_as_new_sample(x) = Tuple(x)

# Pick the candidate that optimizes `scores` among those at least `dtol` away
# from every point in `xs`, dropping the ones that are too close as it goes.
# Returns `(candidate, score)`, or `nothing` once the pool is exhausted.
#
# `scores` and `candidates` are consumed in step, so a caller that needs them
# afterwards must pass copies.
function _select_distant_candidate!(scores, candidates, xs, dtol; pick = argmin)
    while !isempty(candidates)
        i = pick(scores)
        candidate = candidates[i]
        if all(norm(x .- candidate) > dtol for x in xs)
            return (candidate, scores[i])
        end
        deleteat!(scores, i)
        deleteat!(candidates, i)
    end
    return nothing
end

# Range of surrogate predictions and of nearest-sample distances over a
# candidate pool -- the four normalizing constants `merit_function` needs.
function _merit_ranges(surr::AbstractSurrogate, candidates)
    s = [surr(c) for c in candidates]
    d = [_nearest_sample_distance(surr, c) for c in candidates]
    s_min, s_max = extrema(s)
    d_min, d_max = extrema(d)
    return s_max, s_min, d_max, d_min
end

"""
    surrogate_optimize!(objective, algorithm, lb, ub, surrogate, sample_type;
        maxiters = 100, num_new_samples = 100, needs_gradient = false)

Minimize `objective` with a surrogate-assisted optimization algorithm.

The algorithm generates candidate points with `sample`, scores them using the
surrogate, evaluates the selected point with `objective`, and updates the
surrogate with the new observation. The available methods differ in their
algorithm token and in the surrogate capabilities they require; the common
call/update contract is described under [`AbstractSurrogate`](@ref).

# Arguments

  - `objective::Function`: objective function to minimize. It must accept one
    point in the representation used by `surrogate` and return a scalar or
    objective vector supported by `algorithm`.
  - `algorithm::SurrogateOptimizationAlgorithm`: optimization strategy, such as
    [`SRBF`](@ref), [`LCBS`](@ref), [`EI`](@ref), or [`DYCORS`](@ref).
  - `lb`: lower bound of the search domain.
  - `ub`: upper bound of the search domain, with the same dimensionality as
    `lb`.
  - `surrogate::AbstractSurrogate`: fitted surrogate whose call overload
    predicts an objective value and whose `update!` method accepts new data.
  - `sample_type::SamplingAlgorithm`: sampling strategy used to generate
    candidate points.

# Keywords

  - `maxiters::Integer = 100`: maximum number of optimization iterations. One
    iteration costs one objective evaluation, so this also caps how many times
    `objective` is called -- except for [`SOP`](@ref), which proposes one
    candidate per search center and so costs `SOP.p` evaluations per iteration.
  - `num_new_samples::Integer = 100`: number of candidate points considered at
    each iteration.
  - `needs_gradient::Bool = false`: whether the selected method should evaluate
    an objective gradient and pass it to a gradient-aware `update!` method.
    This keyword is supported by the multidimensional `SRBF` method.

# Returns

`(point, value)`: the best observation the surrogate holds when the search
stops. The value is always a measured objective value, never an acquisition
score. Multi-objective methods ([`SMB`](@ref), [`RTEA`](@ref)) return a Pareto
set and its front instead.

A search can stop before `maxiters` is reached: when every remaining candidate
falls within the minimum separation of an already-evaluated point ("Out of
sampling points"), or when the trust region grows past the domain or shrinks
below a usable width.

# Example

```julia
using Surrogates

objective(x) = (x - 0.25)^2
x = [0.0, 0.5, 1.0]
y = objective.(x)
surrogate = Kriging(x, y, 0.0, 1.0)
best_point, best_value = surrogate_optimize!(
    objective, SRBF(), 0.0, 1.0, surrogate, RandomSample();
    maxiters = 2, num_new_samples = 8)
```
"""
function surrogate_optimize!(
        obj::Function, ::SRBF, lb, ub, surr::AbstractSurrogate,
        sample_type::SamplingAlgorithm; maxiters = 100,
        num_new_samples = 100, needs_gradient = false
    )
    scale = 0.2
    success = 0
    failure = 0
    dtol = _candidate_tolerance(lb, ub)
    num_of_iterations = 0
    for w in Iterators.cycle(_SRBF_WEIGHTS)
        num_of_iterations += 1
        num_of_iterations > maxiters && return _best_point(surr)

        #1) Sample near the incumbent
        incumbent_value = minimum(surr.y)
        incumbent_x = surr.x[argmin(surr.y)]
        new_lb, new_ub = _trust_region(incumbent_x, lb, ub, scale)
        new_sample = sample(num_new_samples, new_lb, new_ub, sample_type)

        #2) Score the candidates and take the best one far enough from the
        #   samples already evaluated
        s_max, s_min, d_max, d_min = _merit_ranges(surr, new_sample)
        merits = [
            merit_function(c, w, surr, s_max, s_min, d_max, d_min) for c in new_sample
        ]
        selection = _select_distant_candidate!(merits, new_sample, surr.x, dtol)
        if selection === nothing
            println("Out of sampling points")
            return _best_point(surr)
        end
        adaptive_point_x = _as_new_sample(first(selection))

        #3) Evaluate the objective there and refit
        adaptive_point_y = obj(adaptive_point_x)
        if needs_gradient
            adaptive_grad = Zygote.gradient(obj, adaptive_point_x)
            update!(surr, adaptive_point_x, adaptive_point_y, adaptive_grad)
        else
            update!(surr, adaptive_point_x, adaptive_point_y)
        end

        #4) Widen or narrow the trust region, judged on the measured value
        #   rather than the refitted surrogate's prediction
        scale, success, failure,
            resized = _adjust_trust_region(
            scale, success, failure, adaptive_point_y < incumbent_value
        )
        if resized && scale > 0.8 * norm(ub - lb)
            println("Exiting, scale too big")
            return _best_point(surr)
        elseif resized && scale < 1.0e-5
            println("Exiting, too narrow")
            return _best_point(surr)
        end
    end
    return
end


"""
    potential_optimal_points(alg, strategy, lb, ub, surrogate, sample_type, n_parallel;
        num_new_samples = 500)

Return a batch of candidate points selected from the current surrogate without
evaluating the true objective.

This is the generic interface used for parallel surrogate optimization. The
method deep-copies the surrogate, selects one candidate at a time, and calls the
virtual-point strategy to update the temporary surrogate between selections.

# Arguments

  - `alg::SurrogateOptimizationAlgorithm`: optimization strategy, currently
    implemented for [`SRBF`](@ref).
  - `strategy::ParallelStrategy`: virtual-point update strategy such as
    [`MinimumConstantLiar`](@ref) or [`KrigingBeliever`](@ref).
  - `lb`: lower bound of the search domain.
  - `ub`: upper bound of the search domain.
  - `surrogate`: fitted surrogate satisfying the [`AbstractSurrogate`](@ref)
    evaluation and `update!` interface.
  - `sample_type::SamplingAlgorithm`: sampling strategy used to generate the
    candidate pool.
  - `n_parallel::Integer`: number of candidate points to return.

# Keywords

  - `num_new_samples`: number of sampled candidate points considered before
    selecting the batch.

# Returns

A tuple `(points, merits)`, where `points` contains the selected candidate
locations and `merits` contains their merit-function values.
"""
function potential_optimal_points(
        ::SRBF, strategy, lb, ub, surr::AbstractSurrogate,
        sample_type::SamplingAlgorithm, n_parallel;
        num_new_samples = 500
    )
    scale = 0.2
    w_cycle = Iterators.cycle(_SRBF_WEIGHTS)
    w, state = iterate(w_cycle)
    dtol = _candidate_tolerance(lb, ub)

    incumbent_x = surr.x[argmin(surr.y)]
    new_lb, new_ub = _trust_region(incumbent_x, lb, ub, scale)
    new_sample = sample(num_new_samples, new_lb, new_ub, sample_type)

    # Virtual points accumulate here; the true surrogate is left untouched.
    tmp_surr = deepcopy(surr)
    proposed_points_x = Vector{typeof(surr.x[1])}(undef, n_parallel)
    merit_of_proposed_points = zeros(float(eltype(surr.y)), n_parallel)

    new_addition = 0
    while new_addition < n_parallel
        # Scored against `tmp_surr`, so the liars placed at the points already
        # chosen for this batch steer the next choice away from them, and the
        # separation filter sees them too.
        s_max, s_min, d_max, d_min = _merit_ranges(tmp_surr, new_sample)
        merits = [
            merit_function(c, w, tmp_surr, s_max, s_min, d_max, d_min)
                for c in new_sample
        ]
        selection = _select_distant_candidate!(merits, new_sample, tmp_surr.x, dtol)
        if selection === nothing
            println("Out of sampling points")
            return (
                proposed_points_x[1:new_addition],
                merit_of_proposed_points[1:new_addition],
            )
        end

        new_addition += 1
        proposed_points_x[new_addition], merit_of_proposed_points[new_addition] = selection
        calculate_liars(strategy, tmp_surr, surr, proposed_points_x[new_addition])
        w, state = iterate(w_cycle, state)
    end

    return (proposed_points_x, merit_of_proposed_points)
end

"""
    surrogate_optimize!(obj, ::LCBS, lb, ub, krig, sample_type;
        maxiters = 100, num_new_samples = 100, k = 2.0)

Minimize `obj` with the lower confidence bound acquisition function.

Under a Gaussian process prior the acquisition is

``LCB(x) = E[x] - k\\sqrt{V[x]}``

which is minimized over a fresh candidate pool at each iteration. Larger `k`
weights the predictive standard deviation more heavily and so explores more.
The search stops once no candidate's bound improves on the best observation,
meaning none of them can plausibly beat the incumbent.

`krig` must provide `std_error_at_point`, so this method needs a surrogate with
a predictive variance such as [`Kriging`](@ref).

## References

Cox, D.D. and John, S. (1992). A statistical method for global optimization.
*IEEE International Conference on Systems, Man, and Cybernetics*, 1241-1246.

Srinivas, N., Krause, A., Kakade, S.M. and Seeger, M. (2010). Gaussian process
optimization in the bandit setting: no regret and experimental design.
*ICML*, 1015-1022.
"""
function surrogate_optimize!(
        obj::Function, ::LCBS, lb, ub, krig,
        sample_type::SamplingAlgorithm; maxiters = 100,
        num_new_samples = 100, k = 2.0
    )
    dtol = _candidate_tolerance(lb, ub)
    for _ in 1:maxiters
        new_sample = sample(num_new_samples, lb, ub, sample_type)
        bounds = [
            krig(c) - k * std_error_at_point(krig, c) for c in new_sample
        ]

        selection = _select_distant_candidate!(bounds, new_sample, krig.x, dtol)
        if selection === nothing
            println("Out of sampling points")
            return _best_point(krig)
        end
        min_add_x, min_add_bound = selection

        # Nothing left that could beat the incumbent even at its optimistic
        # bound, so there is no point evaluating further.
        min_add_bound >= minimum(krig.y) && return _best_point(krig)

        min_add_y = obj(min_add_x)
        if isinf(min_add_y) || isnan(min_add_y)
            println("New point being added is +Inf or NaN, skipping.")
        else
            update!(krig, _as_new_sample(min_add_x), min_add_y)
        end
    end
    return _best_point(krig)
end

# Ask EI, 1-D and ND
function potential_optimal_points(
        ::EI, strategy, lb, ub, krig,
        sample_type::SamplingAlgorithm, n_parallel::Number;
        num_new_samples = 100
    )
    lb = krig.lb
    ub = krig.ub
    dtol = _candidate_tolerance(lb, ub)

    # Virtual points accumulate here; the true surrogate is left untouched.
    tmp_krig = deepcopy(krig)
    new_x_max = Vector{typeof(tmp_krig.x[1])}(undef, n_parallel)
    new_EI_max = zeros(float(eltype(tmp_krig.y)), n_parallel)

    for i in 1:n_parallel
        new_sample = sample(num_new_samples, lb, ub, sample_type)
        f_min = minimum(tmp_krig.y)
        improvements = [_expected_improvement(tmp_krig, c, f_min) for c in new_sample]

        # Filtered against `tmp_krig`, which holds a liar at every point already
        # chosen for this batch, so the batch cannot repeat a point.
        selection = _select_distant_candidate!(
            improvements, new_sample, tmp_krig.x, dtol; pick = argmax
        )
        if selection === nothing
            println("Out of sampling points")
            return (new_x_max[1:(i - 1)], new_EI_max[1:(i - 1)])
        end

        new_x_max[i], new_EI_max[i] = selection
        calculate_liars(strategy, tmp_krig, krig, new_x_max[i])
    end

    return (new_x_max, new_EI_max)
end

"""
    surrogate_optimize!(obj, ::EI, lb, ub, krig, sample_type;
        maxiters = 100, num_new_samples = 100)

Minimize `obj` with the expected improvement acquisition function.

At each iteration a fresh candidate pool is scored by

``EI(x) = (f_{min} - \\mu(x) - \\xi)\\Phi(z) + \\sigma(x)\\phi(z),
\\qquad z = \\frac{f_{min} - \\mu(x) - \\xi}{\\sigma(x)}``

the candidate maximizing it is evaluated, and the surrogate is refitted. The
offset ``\\xi`` biases the search towards exploration. The search stops once the
best expected improvement is negligible against the spread of the observations.

`krig` must provide `std_error_at_point`, so this method needs a surrogate with
a predictive variance such as [`Kriging`](@ref).

## References

Jones, D.R., Schonlau, M. and Welch, W.J. (1998). Efficient global optimization
of expensive black-box functions. *Journal of Global Optimization*, 13,
455-492.
"""
function surrogate_optimize!(
        obj::Function, ::EI, lb, ub, krig,
        sample_type::SamplingAlgorithm; maxiters = 100,
        num_new_samples = 100
    )
    dtol = _candidate_tolerance(lb, ub)
    for _ in 1:maxiters
        new_sample = sample(num_new_samples, lb, ub, sample_type)
        f_min = minimum(krig.y)
        improvements = [_expected_improvement(krig, c, f_min) for c in new_sample]

        selection = _select_distant_candidate!(
            improvements, new_sample, krig.x, dtol; pick = argmax
        )
        if selection === nothing
            println("Out of sampling points")
            return _best_point(krig)
        end
        new_x_max, new_EI_max = selection

        if new_EI_max < 1.0e-6 * norm(maximum(krig.y) - minimum(krig.y))
            println("Termination tolerance reached.")
            return _best_point(krig)
        end
        update!(krig, _as_new_sample(new_x_max), obj(new_x_max))
    end
    println("Completed maximum number of iterations.")
    return _best_point(krig)
end

function adjust_step_size(sigma_n, sigma_min, C_success, t_success, C_fail, t_fail)
    if C_success >= t_success
        sigma_n = 2 * sigma_n
        C_success = 0
    end
    if C_fail >= t_fail
        sigma_n = max(sigma_n / 2, sigma_min)
        C_fail = 0
    end
    return sigma_n, C_success, C_fail
end

"""
    select_evaluation_point(candidates, surr, numb_iters)

Pick the candidate with the best weighted score, cycling the weight.

The score is `merit_function`; the weight comes from the cyclic pattern
of Regis and Shoemaker at iteration `numb_iters`, so successive iterations
alternate between refining near the incumbent and probing unexplored regions.
This is the selection step DYCORS shares with SRBF -- the two differ in how
candidates are generated, not in how they are ranked.
"""
function select_evaluation_point(candidates, surr::AbstractSurrogate, numb_iters)
    w = _SRBF_WEIGHTS[mod1(numb_iters - 1, length(_SRBF_WEIGHTS))]
    s_max, s_min, d_max, d_min = _merit_ranges(surr, candidates)
    scores = [merit_function(c, w, surr, s_max, s_min, d_max, d_min) for c in candidates]
    return candidates[argmin(scores)]
end

"""
    surrogate_optimize!(obj, ::DYCORS, lb::Number, ub::Number, surr1, sample_type;
        maxiters = 100, num_new_samples = 100)

One-dimensional DYCORS. With a single coordinate there is nothing to choose
between, so this reduces to perturbing the incumbent by a Gaussian step whose
width follows the same success/failure schedule as the multidimensional method.
See the multidimensional method for the algorithm and its reference.
"""
function surrogate_optimize!(
        obj::Function, ::DYCORS, lb::Number, ub::Number,
        surr1::AbstractSurrogate, sample_type::SamplingAlgorithm;
        maxiters = 100, num_new_samples = 100
    )
    x_best = surr1.x[argmin(surr1.y)]
    y_best = minimum(surr1.y)
    sigma_n = 0.2 * norm(ub - lb)
    d = length(lb)
    sigma_min = 0.2 * (0.5)^6 * norm(ub - lb)
    t_success = 3
    t_fail = max(d, 5)
    C_success = 0
    C_fail = 0
    for k in 1:maxiters
        # Falls from the full perturbation probability to zero over the run,
        # so later iterations perturb fewer coordinates.
        p_select = min(20 / d, 1) * (1 - log(k) / log(max(maxiters, 2)))
        # In 1D I_perturb is always equal to one, no need to sample
        d = 1
        I_perturb = d
        new_points = zeros(eltype(surr1.x[1]), num_new_samples)
        for i in 1:num_new_samples
            new_points[i] = x_best + rand(Normal(0, sigma_n))
            # Reflect a perturbation that leaves the box back about the
            # bound it crossed, clamping if it overshoots the far side.
            while new_points[i] < lb || new_points[i] > ub
                if new_points[i] > ub
                    new_points[i] = max(lb, 2 * ub - new_points[i])
                end
                if new_points[i] < lb
                    new_points[i] = min(ub, 2 * lb - new_points[i])
                end
            end
        end

        x_new = select_evaluation_point(new_points, surr1, k)
        f_new = obj(x_new)

        if f_new < y_best
            C_success = C_success + 1
            C_fail = 0
        else
            C_fail = C_fail + 1
            C_success = 0
        end

        sigma_n, C_success,
            C_fail = adjust_step_size(
            sigma_n, sigma_min, C_success,
            t_success, C_fail, t_fail
        )

        if f_new < y_best
            x_best = x_new
            y_best = f_new
        end
        update!(surr1, x_new, f_new)
    end
    index = argmin(surr1.y)
    return (surr1.x[index], surr1.y[index])
end


"""
    surrogate_optimize!(obj, ::DYCORS, lb, ub, surrn, sample_type;
        maxiters = 100, num_new_samples = 100)

Minimize `obj` with dynamic coordinate search.

DYCORS extends SRBF by changing how candidates are generated, not how they are
ranked: candidates are Gaussian perturbations of the incumbent in a random
subset of the coordinates rather than a design over a trust region. Each
coordinate is perturbed with probability

``p_{select}(k) = \\min(20/d, 1)\\left(1 - \\frac{\\ln k}{\\ln k_{max}}\\right)``

which falls to zero over the run, so late iterations move along very few
directions -- the useful behaviour when the objective depends on only a handful
of them. At least one coordinate is always perturbed. The perturbation width
doubles after three consecutive improvements and halves after `max(d, 5)`
consecutive failures. Candidates are ranked by the same weighted score as
SRBF, so the two methods differ only in how candidates are generated.

## References

Regis, R.G. and Shoemaker, C.A. (2013). Combining radial basis function
surrogates and dynamic coordinate search in high-dimensional expensive
black-box optimization. *Engineering Optimization*, 45(5), 529-555.
"""
function surrogate_optimize!(
        obj::Function, ::DYCORS, lb, ub, surrn::AbstractSurrogate,
        sample_type::SamplingAlgorithm; maxiters = 100,
        num_new_samples = 100
    )
    x_best = collect(surrn.x[argmin(surrn.y)])
    y_best = minimum(surrn.y)
    sigma_n = 0.2 * norm(ub - lb)
    d = length(lb)
    sigma_min = 0.2 * (0.5)^6 * norm(ub - lb)
    t_success = 3
    t_fail = max(d, 5)
    C_success = 0
    C_fail = 0
    for k in 1:maxiters
        # Falls from the full perturbation probability to zero over the run,
        # so later iterations perturb fewer coordinates.
        p_select = min(20 / d, 1) * (1 - log(k) / log(max(maxiters, 2)))
        new_points = zeros(eltype(surrn.x[1]), num_new_samples, d)
        for j in 1:num_new_samples
            # A fresh draw per candidate: a design sequence would restart and
            # hand every candidate the same mask.
            w = rand(d)
            I_perturb = w .< p_select
            if ~(true in I_perturb)
                val = rand(1:d)
                I_perturb = vcat(zeros(Int, val - 1), 1, zeros(Int, d - val))
            end
            I_perturb = Int.(I_perturb)
            for i in 1:d
                if I_perturb[i] == 1
                    new_points[j, i] = x_best[i] + rand(Normal(0, sigma_n))
                else
                    new_points[j, i] = x_best[i]
                end
            end
        end

        for i in 1:num_new_samples
            for j in 1:d
                while new_points[i, j] < lb[j] || new_points[i, j] > ub[j]
                    if new_points[i, j] > ub[j]
                        new_points[i, j] = max(lb[j], 2 * ub[j] - new_points[i, j])
                    end
                    if new_points[i, j] < lb[j]
                        new_points[i, j] = min(ub[j], 2 * lb[j] - new_points[i, j])
                    end
                end
            end
        end

        #ND version
        # `new_points` is a `num_new_samples x d` matrix; the surrogate is
        # queried with coordinate tuples.
        candidates = [Tuple(new_points[i, :]) for i in axes(new_points, 1)]
        x_new = collect(select_evaluation_point(candidates, surrn, k))
        f_new = obj(x_new)

        if f_new < y_best
            C_success = C_success + 1
            C_fail = 0
        else
            C_fail = C_fail + 1
            C_success = 0
        end

        sigma_n, C_success,
            C_fail = adjust_step_size(
            sigma_n, sigma_min, C_success,
            t_success, C_fail, t_fail
        )

        if f_new < y_best
            x_best = x_new
            y_best = f_new
        end
        update!(surrn, Tuple(x_new), f_new)
    end
    index = argmin(surrn.y)
    return (surrn.x[index], surrn.y[index])
end

function obj2_1D(value, points)
    min = +Inf
    my_p = filter(x -> abs(x - value) > 10^-6, points)
    for i in 1:length(my_p)
        new_val = norm(my_p[i] - value)
        if new_val < min
            min = new_val
        end
    end
    return min
end

function I_tier_ranking_1D(P, surrSOP::AbstractSurrogate)
    #obj1 = objective_function
    #obj2 = obj2_1D
    Fronts = Dict{Int, Array{eltype(surrSOP.x[1]), 1}}()
    i = 1
    while true
        F = []
        j = 1
        for p in P
            n_p = 0
            k = 1
            for q in P
                #I use equality with floats because p and q are in surrSOP.x
                #for sure at this stage
                p_index = j
                q_index = k
                val1_p = surrSOP.y[p_index]
                val2_p = obj2_1D(p, P)
                val1_q = surrSOP.y[q_index]
                val2_q = obj2_1D(q, P)
                # `q` dominates `p` when it is no worse in both objectives and
                # strictly better in one.
                q_dominates_p = (val1_q < val1_p || abs(val1_q - val1_p) <= 10^-5) &&
                    (val2_q < val2_p || abs(val2_q - val2_p) <= 10^-5) &&
                    ((val1_q < val1_p) || (val2_q < val2_p))
                if q_dominates_p
                    n_p += 1
                end
                k = k + 1
            end
            if n_p == 0
                # no individual dominates p
                push!(F, p)
            end
            j = j + 1
        end
        if length(F) > 0
            Fronts[i] = F
            P = setdiff(P, F)
            i = i + 1
        else
            return Fronts
        end
    end
    return F
end

function II_tier_ranking_1D(D::Dict, srg::AbstractSurrogate)
    for i in 1:length(D)
        D[i] = D[i][sortperm(D[i])]
    end
    return D
end

# Hypervolume of the region dominated by `points` and bounded by `v_ref`, for
# two objectives, both minimised.
#
# Sweeps the rows in order of the first objective and accumulates a strip only
# where the second objective improves on every row seen so far. Dominated rows
# never improve on it and so contribute nothing, which is what lets the caller
# pass a set that is not a Pareto front.
function _dominated_hypervolume(points, v_ref)
    order = sortperm(points[:, 1])
    area = zero(eltype(points))
    best_f2 = v_ref[2]
    for i in order
        f1, f2 = points[i, 1], points[i, 2]
        if f2 < best_f2 && f1 < v_ref[1]
            area += (v_ref[1] - f1) * (best_f2 - f2)
            best_f2 = f2
        end
    end
    return area
end

# Hypervolume gained by adding `(f1_new, f2_new)` to `Pareto_set`.
#
# Both areas are measured against the same reference point, taken over the union
# of the old set and the new one; measuring each against its own maximum would
# compare areas in two different frames.
function Hypervolume_Pareto_improving(f1_new, f2_new, Pareto_set)
    Pareto_after = vcat(Pareto_set, [f1_new f2_new])
    # The reference has to *strictly* dominate every row, or whichever row
    # attains the maximum contributes a zero-width strip and is invisible. Its
    # offset is a fraction of each objective's own spread, so the measure stays
    # scale free; a degenerate objective falls back to one.
    v_ref = map(1:2) do j
        col = @view Pareto_after[:, j]
        spread = maximum(col) - minimum(col)
        maximum(col) + (spread > 0 ? 0.1 * spread : one(spread))
    end
    v_ref = reshape(v_ref, 1, 2)
    return _dominated_hypervolume(Pareto_after, v_ref) -
        _dominated_hypervolume(Pareto_set, v_ref)
end

"""
    surrogate_optimize!(obj, sop::SOP, lb, ub, surr, sample_type;
        maxiters = 100, num_new_samples = min(500d, 5000))

Minimize `obj` with surrogate optimization using Pareto center selection.

SOP maintains several search centers at once and picks them by non-dominated
sorting on two criteria: the observed objective value, and the distance to the
nearest evaluated point. Ranking centers this way spreads them between the
promising and the unexplored parts of the domain, which is what makes the
method parallel-friendly -- `sop.p` centers are carried, and one candidate is
proposed from each per iteration. A center whose proposal fails to improve the dominated hypervolume
has its radius halved, and after enough failures it is placed on a tabu list.

`num_new_samples` is best set to `min(500d, 5000)` for a `d`-dimensional
problem.

## References

Krityakierne, T., Akhtar, T. and Shoemaker, C.A. (2016). SOP: parallel
surrogate global optimization with Pareto center selection for computationally
expensive single objective problems. *Journal of Global Optimization*, 64,
421-445.

Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*.
Wiley.
"""
function surrogate_optimize!(
        obj::Function, sop1::SOP, lb::Number, ub::Number,
        surrSOP::AbstractSurrogate, sample_type::SamplingAlgorithm;
        maxiters = 100, num_new_samples = min(500 * 1, 5000)
    )
    d = length(lb)
    N_fail = 3
    N_tenure = 5
    tau = 10^-5
    num_P = sop1.p
    centers_global = surrSOP.x
    r_centers_global = 0.2 * norm(ub - lb) * ones(length(surrSOP.x))
    N_failures_global = zeros(length(surrSOP.x))
    tabu = []
    N_tenures_tabu = []
    for k in 1:maxiters
        N_tenures_tabu .+= 1
        #deleting points that have been in tabu for too long
        del = N_tenures_tabu .> N_tenure

        if length(del) > 0
            for i in 1:length(del)
                if del[i]
                    del[i] = i
                end
            end
            deleteat!(N_tenures_tabu, del)
            deleteat!(tabu, del)
        end

        ##### P CENTERS ######
        C = []

        #S(x) set of points already evaluated
        #Rank points in S with:
        #1) Non dominated sorting
        Fronts_I = I_tier_ranking_1D(centers_global, surrSOP)
        #2) Second tier ranking
        Fronts = II_tier_ranking_1D(Fronts_I, surrSOP)
        ranked_list = []
        for i in 1:length(Fronts)
            for j in 1:length(Fronts[i])
                push!(ranked_list, Fronts[i][j])
            end
        end
        ranked_list = eltype(surrSOP.x[1]).(ranked_list)

        centers_full = 0
        i = 1
        while i <= length(ranked_list) && centers_full == 0
            flag = 0
            for j in 1:length(ranked_list)
                for m in 1:length(tabu)
                    if abs(ranked_list[j] - tabu[m]) < tau
                        flag = 1
                    end
                end
                for l in 1:length(centers_global)
                    if abs(ranked_list[j] - centers_global[l]) < tau
                        flag = 1
                    end
                end
            end
            if flag == 1
                skip
            else
                push!(C, ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
            end
            i = i + 1
        end

        #I examined all the points in the ranked list but num_selected < num_p
        #I just iterate again using only radius rule
        if length(C) < num_P
            i = 1
            while i <= length(ranked_list) && centers_full == 0
                flag = 0
                for j in 1:length(ranked_list)
                    for m in 1:length(centers_global)
                        if abs(centers_global[j] - ranked_list[m]) < tau
                            flag = 1
                        end
                    end
                end
                if flag == 1
                    skip
                else
                    push!(C, ranked_list[i])
                    if length(C) == num_P
                        centers_full = 1
                    end
                end
                i = i + 1
            end
        end

        #If I still have num_selected < num_P, I double down on some centers iteratively
        if length(C) < num_P
            i = 1
            while i <= length(ranked_list)
                push!(C, ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
                i = i + 1
            end
        end

        #Here I have selected C = [] containing the centers
        r_centers = 0.2 * norm(ub - lb) * ones(num_P)
        N_failures = zeros(num_P)
        #2.3 Candidate search
        new_points = zeros(eltype(surrSOP.x[1]), num_P, 2)
        for i in 1:num_P
            N_candidates = zeros(eltype(surrSOP.x[1]), num_new_samples)
            #Using phi(n) just like DYCORS, merit function = surrogate
            #Like in DYCORS, I_perturb = 1 always
            evaluations = zeros(eltype(surrSOP.y[1]), num_new_samples)
            for j in 1:num_new_samples
                a = lb - C[i]
                b = ub - C[i]
                N_candidates[j] = C[i] + rand(truncated(Normal(0, r_centers[i]), a, b))
                evaluations[j] = surrSOP(N_candidates[j])
            end
            x_best = N_candidates[argmin(evaluations)]
            y_best = minimum(evaluations)
            new_points[i, 1] = x_best
            new_points[i, 2] = y_best
        end

        #new_points[i] now contains:
        #[x_1,y_1; x_2,y_2,...,x_{num_new_samples},y_{num_new_samples}]

        #2.4 Adaptive learning and tabu archive
        for i in 1:num_P
            if new_points[i, 1] in centers_global
                r_centers[i] = r_centers_global[i]
                N_failures[i] = N_failures_global[i]
            end

            f_1 = obj(new_points[i, 1])
            # Second objective: distance from the candidate to the nearest
            # evaluated point.
            f_2 = obj2_1D(new_points[i, 1], surrSOP.x)

            l = length(Fronts[1])
            Pareto_set = zeros(eltype(surrSOP.x[1]), l, 2)

            for j in 1:l
                val = obj2_1D(Fronts[1][j], surrSOP.x)
                Pareto_set[j, 1] = obj(Fronts[1][j])
                Pareto_set[j, 2] = val
            end
            if (Hypervolume_Pareto_improving(f_1, f_2, Pareto_set) < tau)
                #failure
                r_centers[i] = r_centers[i] / 2
                N_failures[i] += 1
                if N_failures[i] > N_fail
                    push!(tabu, C[i])
                    push!(N_tenures_tabu, 0)
                end
            else
                #P_i is success
                #Adaptive_learning
                # `new_points[i, 2]` is the surrogate's own prediction at the
                # candidate, used to rank candidates. The surrogate is fitted to
                # observations, so store the measured value instead.
                update!(surrSOP, new_points[i, 1], f_1)
                push!(r_centers_global, r_centers[i])
                push!(N_failures_global, N_failures[i])
            end
        end
    end
    index = argmin(surrSOP.y)
    return (surrSOP.x[index], surrSOP.y[index])
end

function obj2_ND(value, points)
    min = +Inf
    my_p = filter(x -> norm(x .- value) > 10^-6, points)
    for i in 1:length(my_p)
        new_val = norm(my_p[i] .- value)
        if new_val < min
            min = new_val
        end
    end
    return min
end

function I_tier_ranking_ND(P, surrSOPD::AbstractSurrogate)
    #obj1 = objective_function
    #obj2 = obj2_1D
    Fronts = Dict{Int, Array{eltype(surrSOPD.x), 1}}()
    i = 1
    while true
        F = Array{eltype(surrSOPD.x), 1}()
        j = 1
        for p in P
            n_p = 0
            k = 1
            for q in P
                #I use equality with floats because p and q are in surrSOP.x
                #for sure at this stage
                p_index = j
                q_index = k
                val1_p = surrSOPD.y[p_index]
                val2_p = obj2_ND(p, P)
                val1_q = surrSOPD.y[q_index]
                val2_q = obj2_ND(q, P)
                # `q` dominates `p` when it is no worse in both objectives and
                # strictly better in one.
                q_dominates_p = (val1_q < val1_p || abs(val1_q - val1_p) <= 10^-5) &&
                    (val2_q < val2_p || abs(val2_q - val2_p) <= 10^-5) &&
                    ((val1_q < val1_p) || (val2_q < val2_p))
                if q_dominates_p
                    n_p += 1
                end
                k = k + 1
            end
            if n_p == 0
                # no individual dominates p
                push!(F, p)
            end
            j = j + 1
        end
        if length(F) > 0
            Fronts[i] = F
            P = setdiff(P, F)
            i = i + 1
        else
            return Fronts
        end
    end
    return F
end

function II_tier_ranking_ND(D::Dict, srgD::AbstractSurrogate)
    for i in 1:length(D)
        pos = []
        yn = []
        for j in 1:length(D[i])
            push!(pos, findall(e -> e == D[i][j], srgD.x))
            push!(yn, srgD.y[pos[j]])
        end
        D[i] = D[i][sortperm(D[i])]
    end
    return D
end

function surrogate_optimize!(
        obj::Function, sopd::SOP, lb, ub, surrSOPD::AbstractSurrogate,
        sample_type::SamplingAlgorithm; maxiters = 100,
        num_new_samples = min(500 * length(lb), 5000)
    )
    d = length(lb)
    N_fail = 3
    N_tenure = 5
    tau = 10^-5
    num_P = sopd.p
    centers_global = surrSOPD.x
    r_centers_global = 0.2 * norm(ub .- lb) * ones(length(surrSOPD.x))
    N_failures_global = zeros(length(surrSOPD.x))
    tabu = []
    N_tenures_tabu = []
    for k in 1:maxiters
        N_tenures_tabu .+= 1
        #deleting points that have been in tabu for too long
        del = N_tenures_tabu .> N_tenure

        if length(del) > 0
            for i in 1:length(del)
                if del[i]
                    del[i] = i
                end
            end
            deleteat!(N_tenures_tabu, del)
            deleteat!(tabu, del)
        end

        ##### P CENTERS ######
        C = Array{eltype(surrSOPD.x), 1}()

        #S(x) set of points already evaluated
        #Rank points in S with:
        #1) Non dominated sorting
        Fronts_I = I_tier_ranking_ND(centers_global, surrSOPD)
        #2) Second tier ranking
        Fronts = II_tier_ranking_ND(Fronts_I, surrSOPD)
        ranked_list = Array{eltype(surrSOPD.x), 1}()
        for i in 1:length(Fronts)
            for j in 1:length(Fronts[i])
                push!(ranked_list, Fronts[i][j])
            end
        end

        centers_full = 0
        i = 1
        while i <= length(ranked_list) && centers_full == 0
            flag = 0
            for j in 1:length(ranked_list)
                for m in 1:length(tabu)
                    if norm(ranked_list[j] .- tabu[m]) < tau
                        flag = 1
                    end
                end
                for l in 1:length(centers_global)
                    if norm(ranked_list[j] .- centers_global[l]) < tau
                        flag = 1
                    end
                end
            end
            if flag == 1
                skip
            else
                push!(C, ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
            end
            i = i + 1
        end

        #I examined all the points in the ranked list but num_selected < num_p
        #I just iterate again using only radius rule
        if length(C) < num_P
            i = 1
            while i <= length(ranked_list) && centers_full == 0
                flag = 0
                for j in 1:length(ranked_list)
                    for m in 1:length(centers_global)
                        if norm(centers_global[j] .- ranked_list[m]) < tau
                            flag = 1
                        end
                    end
                end
                if flag == 1
                    skip
                else
                    push!(C, ranked_list[i])
                    if length(C) == num_P
                        centers_full = 1
                    end
                end
                i = i + 1
            end
        end

        #If I still have num_selected < num_P, I double down on some centers iteratively
        if length(C) < num_P
            i = 1
            while i <= length(ranked_list)
                push!(C, ranked_list[i])
                if length(C) == num_P
                    centers_full = 1
                end
                i = i + 1
            end
        end

        #Here I have selected C = [(1.0,2.0),(3.0,4.0),.....] containing the centers
        r_centers = 0.2 * norm(ub .- lb) * ones(num_P)
        N_failures = zeros(num_P)
        #2.3 Candidate search
        new_points_x = Array{eltype(surrSOPD.x), 1}()
        new_points_y = zeros(eltype(surrSOPD.y[1]), num_P)
        for i in 1:num_P
            N_candidates = zeros(eltype(surrSOPD.x[1]), num_new_samples, d)
            #Using phi(n) just like DYCORS, merit function = surrogate
            #Like in DYCORS, I_perturb = 1 always
            evaluations = zeros(eltype(surrSOPD.y[1]), num_new_samples)
            for j in 1:num_new_samples
                for k in 1:d
                    a = lb[k] - C[i][k]
                    b = ub[k] - C[i][k]
                    N_candidates[j, k] = C[i][k] +
                        rand(truncated(Normal(0, r_centers[i]), a, b))
                end
                evaluations[j] = surrSOPD(Tuple(N_candidates[j, :]))
            end
            x_best = Tuple(N_candidates[argmin(evaluations), :])
            y_best = minimum(evaluations)
            push!(new_points_x, x_best)
            new_points_y[i] = y_best
        end

        #new_points[i] is split in new_points_x and new_points_y now contains:
        #[x_1,y_1; x_2,y_2,...,x_{num_new_samples},y_{num_new_samples}]

        #2.4 Adaptive learning and tabu archive
        for i in 1:num_P
            if new_points_x[i] in centers_global
                r_centers[i] = r_centers_global[i]
                N_failures[i] = N_failures_global[i]
            end

            f_1 = obj(Tuple(new_points_x[i]))
            f_2 = obj2_ND(new_points_x[i], surrSOPD.x)

            l = length(Fronts[1])
            Pareto_set = zeros(eltype(surrSOPD.x[1]), l, 2)
            for j in 1:l
                val = obj2_ND(Fronts[1][j], surrSOPD.x)
                Pareto_set[j, 1] = obj(Tuple(Fronts[1][j]))
                Pareto_set[j, 2] = val
            end
            if (Hypervolume_Pareto_improving(f_1, f_2, Pareto_set) < tau) #check this
                #failure
                r_centers[i] = r_centers[i] / 2
                N_failures[i] += 1
                if N_failures[i] > N_fail
                    push!(tabu, C[i])
                    push!(N_tenures_tabu, 0)
                end
            else
                #P_i is success
                #Adaptive_learning
                update!(surrSOPD, new_points_x[i], f_1)
                push!(r_centers_global, r_centers[i])
                push!(N_failures_global, N_failures[i])
            end
        end
    end
    index = argmin(surrSOPD.y)
    return (surrSOPD.x[index], surrSOPD.y[index])
end

#EGO

_dominates(x, y) = all(x .<= y) && any(x .< y)
function _nonDominatedSorting(arr::Array{Float64, 2})
    fronts::Array{Array, 1} = Array[]
    ind::Array{Int64, 1} = collect(1:size(arr, 1))
    while !isempty(arr)
        s = size(arr, 1)
        red = dropdims(
            sum(
                [_dominates(arr[i, :], arr[j, :]) for i in 1:s, j in 1:s],
                dims = 1
            ) .== 0,
            dims = 1
        )
        a = 1:s
        sel::Array{Int64, 1} = a[red]
        push!(fronts, ind[sel])
        da::Array{Int64, 1} = deleteat!(collect(1:s), sel)
        ind = deleteat!(ind, sel)
        arr = arr[da, :]
    end
    return fronts
end

function surrogate_optimize!(
        obj::Function, sbm::SMB, lb::Number, ub::Number,
        surrSMB::AbstractSurrogate, sample_type::SamplingAlgorithm;
        maxiters = 100, n_new_look = 1000
    )
    #obj contains a function for each output dimension
    dim_out = length(surrSMB.y[1])
    d = 1
    x_to_look = sample(n_new_look, lb, ub, sample_type)
    for iter in 1:maxiters
        index_min = 0
        min_mean = +Inf
        for i in 1:n_new_look
            new_mean = sum(obj(x_to_look[i])) / dim_out
            if new_mean < min_mean
                min_mean = new_mean
                index_min = i
            end
        end

        x_new = x_to_look[index_min]
        deleteat!(x_to_look, index_min)
        n_new_look = n_new_look - 1
        # evaluate the true function at that point
        y_new = obj(x_new)
        #update the surrogate
        update!(surrSMB, x_new, y_new)
    end
    #Find and return Pareto
    y = surrSMB.y
    y = permutedims(reshape(hcat(y...), (length(y[1]), length(y)))) #2d matrix
    Fronts = _nonDominatedSorting(y) #this returns the indexes
    pareto_front_index = Fronts[1]
    pareto_set = []
    pareto_front = []
    for i in 1:length(pareto_front_index)
        push!(pareto_set, surrSMB.x[pareto_front_index[i]])
        push!(pareto_front, surrSMB.y[pareto_front_index[i]])
    end
    return pareto_set, pareto_front
end

function surrogate_optimize!(
        obj::Function, smb::SMB, lb, ub, surrSMBND::AbstractSurrogate,
        sample_type::SamplingAlgorithm; maxiters = 100,
        n_new_look = 1000
    )
    #obj contains a function for each output dimension
    dim_out = length(surrSMBND.y[1])
    d = length(lb)
    x_to_look = sample(n_new_look, lb, ub, sample_type)
    for iter in 1:maxiters
        index_min = 0
        min_mean = +Inf
        for i in 1:n_new_look
            new_mean = sum(obj(x_to_look[i])) / dim_out
            if new_mean < min_mean
                min_mean = new_mean
                index_min = i
            end
        end
        x_new = x_to_look[index_min]
        deleteat!(x_to_look, index_min)
        n_new_look = n_new_look - 1
        # evaluate the true function at that point
        y_new = obj(x_new)
        #update the surrogate
        update!(surrSMBND, x_new, y_new)
    end
    #Find and return Pareto
    y = surrSMBND.y
    y = permutedims(reshape(hcat(y...), (length(y[1]), length(y)))) #2d matrix
    Fronts = _nonDominatedSorting(y) #this returns the indexes
    pareto_front_index = Fronts[1]
    pareto_set = []
    pareto_front = []
    for i in 1:length(pareto_front_index)
        push!(pareto_set, surrSMBND.x[pareto_front_index[i]])
        push!(pareto_front, surrSMBND.y[pareto_front_index[i]])
    end
    return pareto_set, pareto_front
end

# RTEA (Noisy model based multi objective optimization + standard rtea by fieldsen), use this for very noisy objective functions because there are a lot of re-evaluations

function surrogate_optimize!(
        obj, rtea::RTEA, lb::Number, ub::Number,
        surrRTEA::AbstractSurrogate, sample_type::SamplingAlgorithm;
        maxiters = 100, n_new_look = 1000
    )
    Z = rtea.z
    K = rtea.k
    p_cross = rtea.p
    n_c = rtea.n_c
    sigma = rtea.sigma
    #find pareto set of the first evaluations: (estimated pareto)
    y = surrRTEA.y
    y = permutedims(reshape(hcat(y...), (length(y[1]), length(y)))) #2d matrix
    Fronts = _nonDominatedSorting(y) #this returns the indexes
    pareto_front_index = Fronts[1]
    pareto_set = []
    pareto_front = []
    for i in 1:length(pareto_front_index)
        push!(pareto_set, surrRTEA.x[pareto_front_index[i]])
        push!(pareto_front, surrRTEA.y[pareto_front_index[i]])
    end
    number_of_revaluations = zeros(Int, length(pareto_set))
    iter = 1
    d = 1
    dim_out = length(surrRTEA.y[1])
    while iter < maxiters
        if iter < (1 - Z) * maxiters
            #1) propose new point x_new

            #sample randomly from (estimated) pareto v and u
            if length(pareto_set) < 2
                throw(ArgumentError("Starting pareto set is too small, increase the number of sampling points of the surrogate."))
            end
            u = pareto_set[rand(1:length(pareto_set))]
            v = pareto_set[rand(1:length(pareto_set))]

            #children
            if rand() < p_cross
                mu = rand()
                if mu <= 0.5
                    beta = (2 * mu)^(1 / n_c + 1)
                else
                    beta = (1 / (2 * (1 - mu)))^(1 / n_c + 1)
                end
                x = 0.5 * ((1 + beta) * v + (1 - beta) * u)
            else
                x = v
            end

            #mutation
            x_new = x + rand(Normal(0, sigma))
            y_new = obj(x_new)

            #update pareto
            new_to_pareto = false
            counter = zeros(Int, dim_out)
            for i in 1:length(pareto_set)
                #compare the y_new values to pareto, if there is at least one entry where it dominates all the others, then it can be in pareto
                for l in 1:dim_out
                    if y_new[l] < pareto_front[i][l]
                        counter[l]
                    end
                end
            end
            for j in 1:dim_out
                if counter[j] == dim_out
                    new_to_pareto = true
                end
            end
            if new_to_pareto == true
                push!(pareto_set, x_new)
                push!(pareto_front, y_new)
                push!(number_of_revaluations, 0)
            end
            update!(surrRTEA, x_new, y_new)
        end
        for k in 1:K
            val, pos = findmin(number_of_revaluations)
            x_r = pareto_set[pos]
            y_r = obj(x_r)
            number_of_revaluations[pos] = number_of_revaluations[pos] + 1
            #check if it is again in the pareto set or not, if not eliminate it from pareto
            still_in_pareto = false
            for i in 1:length(pareto_set)
                counter = zeros(Int, dim_out)
                for l in 1:dim_out
                    if y_r[l] < pareto_front[i][l]
                        counter[l]
                    end
                end
            end
            for j in 1:dim_out
                if counter[j] == dim_out
                    still_in_pareto = true
                end
            end
            if still_in_pareto == false
                #remove from pareto
                deleteat!(pareto_set, pos)
                deleteat!(pareto_front, pos)
                deleteat!(number_of_revaluations, pos)
            end
        end
        iter = iter + 1
    end
    return pareto_set, pareto_front
end

function surrogate_optimize!(
        obj, rtea::RTEA, lb, ub, surrRTEAND::AbstractSurrogate,
        sample_type::SamplingAlgorithm; maxiters = 100,
        n_new_look = 1000
    )
    Z = rtea.z
    K = rtea.k
    p_cross = rtea.p
    n_c = rtea.n_c
    sigma = rtea.sigma
    #find pareto set of the first evaluations: (estimated pareto)
    y = surrRTEAND.y
    y = permutedims(reshape(hcat(y...), (length(y[1]), length(y)))) #2d matrix
    Fronts = _nonDominatedSorting(y) #this returns the indexes
    pareto_front_index = Fronts[1]
    pareto_set = []
    pareto_front = []
    for i in 1:length(pareto_front_index)
        push!(pareto_set, surrRTEAND.x[pareto_front_index[i]])
        push!(pareto_front, surrRTEAND.y[pareto_front_index[i]])
    end
    number_of_revaluations = zeros(Int, length(pareto_set))
    iter = 1
    d = length(lb)
    dim_out = length(surrRTEAND.y[1])
    while iter < maxiters
        if iter < (1 - Z) * maxiters

            #sample pareto_set
            if length(pareto_set) < 2
                throw(ArgumentError("Starting pareto set is too small, increase the number of sampling points of the surrogate."))
            end
            u = pareto_set[rand(1:length(pareto_set))]
            v = pareto_set[rand(1:length(pareto_set))]

            #children
            if rand() < p_cross
                mu = rand()
                if mu <= 0.5
                    beta = (2 * mu)^(1 / n_c + 1)
                else
                    beta = (1 / (2 * (1 - mu)))^(1 / n_c + 1)
                end
                x = 0.5 * ((1 + beta) * v + (1 - beta) * u)
            else
                x = v
            end

            #mutation
            for i in 1:d
                x_new[i] = x[i] + rand(Normal(0, sigma))
            end
            y_new = obj(x_new)

            #update pareto
            new_to_pareto = false
            counter = zeros(Int, dim_out)
            for i in 1:length(pareto_set)
                #compare the y_new values to pareto, if there is at least one entry where it dominates all the others, then it can be in pareto
                for l in 1:dim_out
                    if y_new[l] < pareto_front[i][l]
                        counter[l]
                    end
                end
            end
            for j in 1:dim_out
                if counter[j] == dim_out
                    new_to_pareto = true
                end
            end
            if new_to_pareto == true
                push!(pareto_set, x_new)
                push!(pareto_front, y_new)
                push!(number_of_revaluations, 0)
            end
            update!(surrRTEAND, x_new, y_new)
        end
        for k in 1:K
            val, pos = findmin(number_of_revaluations)
            x_r = pareto_set[pos]
            y_r = obj(x_r)
            number_of_revaluations[pos] = number_of_revaluations[pos] + 1
            #check if it is again in the pareto set or not, if not eliminate it from pareto
            still_in_pareto = false
            for i in 1:length(pareto_set)
                counter = zeros(Int, dim_out)
                for l in 1:dim_out
                    if y_r[l] < pareto_front[i][l]
                        counter[l]
                    end
                end
            end
            for j in 1:dim_out
                if counter[j] == dim_out
                    still_in_pareto = true
                end
            end
            if still_in_pareto == false
                #remove from pareto
                deleteat!(pareto_set, pos)
                deleteat!(pareto_front, pos)
                deleteat!(number_of_revaluations, pos)
            end
        end
        iter = iter + 1
    end
    return pareto_set, pareto_front
end

function surrogate_optimize!(
        obj::Function, ::EI, lb::AbstractArray, ub::AbstractArray, krig,
        sample_type::SectionSample;
        maxiters = 100, num_new_samples = 100
    )
    dtol = 1.0e-3 * norm(ub - lb)
    eps = 0.01
    for i in 1:maxiters
        d = length(krig.x)
        # Sample lots of points from the design space -- we will evaluate the EI function at these points
        new_sample = sample(num_new_samples, lb, ub, sample_type)

        # Find the best point so far
        f_min = minimum(krig.y)

        # Allocate some arrays
        evaluations = zeros(eltype(krig.x[1]), num_new_samples)  # Holds EI function evaluations
        point_found = false                                     # Whether we have found a new point to test
        new_x_max = zero(eltype(krig.x[1]))                     # New x point
        new_EI_max = zero(eltype(krig.x[1]))                    # EI at new x point
        diff_x = zeros(eltype(krig.x[1]), d)

        # For each point in the sample set, evaluate the Expected Improvement function
        while point_found == false
            for j in 1:length(new_sample)
                evaluations[j] = _expected_improvement(krig, new_sample[j], f_min, eps)
            end
            # find the sample which maximizes the EI function
            index_max = argmax(evaluations)
            x_new = new_sample[index_max]   # x point which maximized EI
            EI_new = maximum(evaluations)   # EI at the new point
            for l in 1:d
                diff_x[l] = norm(krig.x[l] .- x_new)
            end
            bit_x = diff_x .> dtol
            #new_min_x has to have some distance from krig.x
            if false in bit_x
                #The new_point is not actually that new, discard it!
                deleteat!(evaluations, index_max)
                deleteat!(new_sample, index_max)
                if length(new_sample) == 0
                    println("Out of sampling points.")
                    return section_sampler_returner(
                        sample_type, krig.x, krig.y, lb, ub,
                        krig
                    )
                end
            else
                point_found = true
                new_x_max = x_new
                new_EI_max = EI_new
            end
        end
        # if the EI is less than some tolerance times the difference between the maximum and minimum points
        # in the surrogate, then we terminate the optimizer.
        if new_EI_max < 1.0e-6 * norm(maximum(krig.y) - minimum(krig.y))
            println("Termination tolerance reached.")
            return section_sampler_returner(sample_type, krig.x, krig.y, lb, ub, krig)
        end
        update!(krig, Tuple(new_x_max), obj(new_x_max))
    end
    return println("Completed maximum number of iterations.")
end

function section_sampler_returner(
        sample_type::SectionSample, surrn_x, surrn_y,
        lb, ub, surrn
    )
    d_fixed = fixed_dimensions(sample_type)
    @assert length(surrn_y) == size(surrn_x)[1]
    surrn_xy = [(surrn_x[y], surrn_y[y]) for y in 1:length(surrn_y)]
    section_surr1_xy = filter(
        xyz -> xyz[1][d_fixed] == Tuple(sample_type.x0[d_fixed]),
        surrn_xy
    )
    section_surr1_x = [xy[1] for xy in section_surr1_xy]
    section_surr1_y = [xy[2] for xy in section_surr1_xy]
    if length(section_surr1_xy) == 0
        @debug "No new point added - surrogate locally stable"
        N_NEW_POINTS = 100
        section_surr1_x = sample(N_NEW_POINTS, lb, ub, sample_type)
        section_surr1_y = zeros(N_NEW_POINTS)
        for i in 1:size(section_surr1_x, 1)
            xi = Tuple([section_surr1_x[i, :]...])[1]
            section_surr1_y[i] = surrn(xi)
        end
    end
    index = argmin(section_surr1_y)
    return (section_surr1_x[index, :][1], section_surr1_y[index])
end
