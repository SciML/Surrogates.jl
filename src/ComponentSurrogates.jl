# Building a component surrogate from a descriptor.
#
# Descriptors carry the surrogate's type and construction is ordinary method
# dispatch, so an unsupported component is a `MethodError` at the call site, a
# mistyped keyword fails at precompile time, and adding a surrogate is one
# method here rather than a string branch in each composite.

"""
    _build_component(descriptor, x, y, lb, ub)

Build the surrogate a composite descriptor asks for, on the design `(x, y)`.

Dispatches on `descriptor.type`. Composites that exclude a component for a
reason of their own — `VariableFidelitySurrogate` and `GEK`, say — check for it
before calling this.
"""
function _build_component(s, x, y, lb, ub)
    # An `ArgumentError` naming the problem, rather than the bare
    # `ErrorException` about a missing field that `s.type` would raise.
    hasproperty(s, :type) || throw(
        ArgumentError(
            "Surrogate descriptor has no `type` field, so there is nothing to " *
                "build. Construct it with one of the `*Structure` helpers, for " *
                "example `RadialBasisStructure(radial_function = linearRadial(), " *
                "scale_factor = 1.0, sparse = false)`."
        )
    )
    return _build_component(s.type, s, x, y, lb, ub)
end

function _build_component(::Type{T}, s, x, y, lb, ub) where {T}
    throw(
        ArgumentError(
            "No component builder for $(T). Add a `_build_component` method " *
                "for it in src/ComponentSurrogates.jl to make it usable as a " *
                "component of MOE or VariableFidelitySurrogate."
        )
    )
end

_build_component(::Type{RadialBasis}, s, x, y, lb, ub) = RadialBasis(
    x, y, lb, ub; rad = s.radial_function, scale_factor = s.scale_factor,
    sparse = s.sparse
)

_build_component(::Type{Kriging}, s, x, y, lb, ub) =
    Kriging(x, y, lb, ub; p = s.p, theta = s.theta)

_build_component(::Type{GEK}, s, x, y, lb, ub) =
    GEK(x, y, lb, ub; p = s.p, theta = s.theta)

_build_component(::Type{LinearSurrogate}, s, x, y, lb, ub) =
    LinearSurrogate(x, y, lb, ub)

# Keyword form: the positional one matches the struct's own constructor and so
# skips the outer constructor's `p > 0` check.
_build_component(::Type{InverseDistanceSurrogate}, s, x, y, lb, ub) =
    InverseDistanceSurrogate(x, y, lb, ub; p = s.p)

_build_component(::Type{LobachevskySurrogate}, s, x, y, lb, ub) =
    LobachevskySurrogate(x, y, lb, ub; alpha = s.alpha, n = s.n, sparse = s.sparse)

_build_component(::Type{SecondOrderPolynomialSurrogate}, s, x, y, lb, ub) =
    SecondOrderPolynomialSurrogate(x, y, lb, ub)

_build_component(::Type{Wendland}, s, x, y, lb, ub) =
    Wendland(x, y, lb, ub; eps = s.eps, maxiters = s.maxiters, tol = s.tol)

_build_component(::Type{NeuralSurrogate}, s, x, y, lb, ub) = NeuralSurrogate(
    x, y, lb, ub; model = s.model, loss = s.loss, opt = s.opt,
    n_epochs = s.n_epochs
)

_build_component(::Type{XGBoostSurrogate}, s, x, y, lb, ub) =
    XGBoostSurrogate(x, y, lb, ub; num_round = s.num_round)

# `PolyChaosStructure` names its field `op`; the constructor keyword is
# `orthopolys`.
_build_component(::Type{PolynomialChaosSurrogate}, s, x, y, lb, ub) =
    PolynomialChaosSurrogate(x, y, lb, ub; orthopolys = s.op)

# Not an oversight: `GENNSurrogate` is gradient-enhanced and needs a per-sample
# gradient array, and a composite carries only function values.
function _build_component(::Type{GENNSurrogate}, s, x, y, lb, ub)
    throw(
        ArgumentError(
            "GENNSurrogate cannot be a component of a composite surrogate: it " *
                "is gradient-enhanced and needs a per-sample gradient array, " *
                "which MOE and VariableFidelitySurrogate do not carry."
        )
    )
end
