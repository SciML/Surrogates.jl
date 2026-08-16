"""
    VariableFidelitySurrogate(x, y, lb, ub; num_high_fidel = floor(Int, length(x) / 2),
        low_fid_structure = RadialBasisStructure(...),
        high_fid_structure = RadialBasisStructure(...))

Surrogate that combines low-fidelity observations with a correction surrogate
fit to the high-fidelity residuals.

The first `num_high_fidel` samples are treated as high-fidelity data. The
remaining samples are treated as low-fidelity data. Evaluation returns the sum
of the fitted low-fidelity surrogate and the high-fidelity residual surrogate.

# Fields

  - `x`: all training inputs.
  - `y`: all training responses.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.
  - `num_high_fidel`: number of leading samples treated as high fidelity.
  - `low_fid_surr`: surrogate fitted to low-fidelity data.
  - `eps_surr`: surrogate fitted to high-fidelity residuals.
  - `high_fid_structure`: configuration of `eps_surr`, retained so `update!`
    can refit the residual model against the updated low-fidelity surrogate.

# Arguments

  - `x`: sample locations, ordered with high-fidelity samples first.
  - `y`: observed values corresponding to `x`.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Keywords

  - `num_high_fidel`: number of leading samples treated as high fidelity.
  - `low_fid_structure`: named-tuple surrogate configuration for low-fidelity
    data, e.g. [`RadialBasisStructure`](@ref) or [`KrigingStructure`](@ref).
  - `high_fid_structure`: named-tuple surrogate configuration for the residual
    model.

# Returns

A `VariableFidelitySurrogate` satisfying the generic surrogate interface.
"""
mutable struct VariableFidelitySurrogate{X, Y, L, U, N, F, E, H} <:
    AbstractDeterministicSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    num_high_fidel::N
    low_fid_surr::F
    eps_surr::E
    high_fid_structure::H
end

# Build the surrogate described by a `*Structure` named tuple (see the
# `XYZStructure` constructors in Surrogates.jl) on the given data slice.
function _build_fidelity_surrogate(structure, x, y, lb, ub)
    name = structure.name
    return if name == "RadialBasis"
        RadialBasis(
            x, y, lb, ub,
            rad = structure.radial_function,
            scale_factor = structure.scale_factor,
            sparse = structure.sparse
        )
    elseif name == "Kriging"
        Kriging(x, y, lb, ub, p = structure.p, theta = structure.theta)
    elseif name == "GEK"
        GEK(x, y, lb, ub, p = structure.p, theta = structure.theta)
    elseif name == "LinearSurrogate"
        LinearSurrogate(x, y, lb, ub)
    elseif name == "InverseDistanceSurrogate"
        InverseDistanceSurrogate(x, y, lb, ub, p = structure.p)
    elseif name == "LobachevskySurrogate"
        LobachevskySurrogate(
            x, y, lb, ub,
            alpha = structure.alpha,
            n = structure.n,
            sparse = structure.sparse
        )
    elseif name == "NeuralSurrogate"
        NeuralSurrogate(
            x, y, lb, ub,
            model = structure.model,
            loss = structure.loss,
            opt = structure.opt,
            n_epochs = structure.n_epochs
        )
    elseif name == "XGBoostSurrogate"
        XGBoostSurrogate(x, y, lb, ub, num_round = structure.num_round)
    elseif name == "SecondOrderPolynomialSurrogate"
        SecondOrderPolynomialSurrogate(x, y, lb, ub)
    elseif name == "Wendland"
        Wendland(
            x, y, lb, ub, eps = structure.eps,
            maxiters = structure.maxiters, tol = structure.tol
        )
    else
        throw(ArgumentError("A surrogate named \"$name\" does not exist or is not currently supported with VariableFidelitySurrogate."))
    end
end

function VariableFidelitySurrogate(
        x, y, lb, ub;
        num_high_fidel = Int(floor(length(x) / 2)),
        low_fid_structure = RadialBasisStructure(
            radial_function = linearRadial(),
            scale_factor = 1.0,
            sparse = false
        ),
        high_fid_structure = RadialBasisStructure(
            radial_function = cubicRadial(),
            scale_factor = 1.0,
            sparse = false
        )
    )
    if num_high_fidel < 1 || num_high_fidel >= length(x)
        throw(ArgumentError("num_high_fidel must be between 1 and length(x) - 1 so that both fidelity levels have data! Got: $num_high_fidel for $(length(x)) samples."))
    end
    x_high = x[1:num_high_fidel]
    x_low = x[(num_high_fidel + 1):end]
    y_high = y[1:num_high_fidel]
    y_low = y[(num_high_fidel + 1):end]
    #Fit low fidelity surrogate:
    low_fid_surr = _build_fidelity_surrogate(low_fid_structure, x_low, y_low, lb, ub)
    #Fit high fidelity residual surrogate on y_high - low_fid_surr(x_high):
    y_eps = _fidelity_residuals(x_high, y_high, low_fid_surr)
    eps_surr = _build_fidelity_surrogate(high_fid_structure, x_high, y_eps, lb, ub)
    return VariableFidelitySurrogate(
        x, y, lb, ub, num_high_fidel, low_fid_surr, eps_surr, high_fid_structure
    )
end

# A comprehension rather than `zeros(eltype(y), …)`: it also covers vector-valued
# responses, for which `zero(::Type{Vector})` is undefined.
function _fidelity_residuals(x_high, y_high, low_fid_surr)
    return [y_high[i] - low_fid_surr(x_high[i]) for i in eachindex(x_high)]
end

function (varfid::VariableFidelitySurrogate)(val)
    return varfid.eps_surr(val) + varfid.low_fid_surr(val)
end

"""
    update!(varfid::VariableFidelitySurrogate, x_new, y_new)

Add new *low-fidelity* observations to the surrogate. Adding high-fidelity data
is not supported.

Both models are refit: the low-fidelity surrogate absorbs the new samples, and
the residual surrogate is rebuilt against it. Refitting the residual model is
required, not cosmetic — it is fitted to `y_high - low_fid_surr(x_high)`, so
leaving it stale after the low-fidelity surrogate moves would silently break
the surrogate's agreement with the high-fidelity data.
"""
function SurrogatesBase.update!(varfid::VariableFidelitySurrogate, x_new, y_new)
    varfid.x = vcat(varfid.x, x_new)
    varfid.y = vcat(varfid.y, y_new)
    update!(varfid.low_fid_surr, x_new, y_new)
    # New samples are appended, so the leading num_high_fidel entries are still
    # the high-fidelity block.
    x_high = varfid.x[1:(varfid.num_high_fidel)]
    y_high = varfid.y[1:(varfid.num_high_fidel)]
    y_eps = _fidelity_residuals(x_high, y_high, varfid.low_fid_surr)
    varfid.eps_surr = _build_fidelity_surrogate(
        varfid.high_fid_structure, x_high, y_eps, varfid.lb, varfid.ub
    )
    return nothing
end
