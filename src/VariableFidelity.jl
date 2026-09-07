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
  - `eps_structure`: the configuration `eps_surr` was built from, kept so that
    `update!` can refit it against the corrected residuals.

# Arguments

  - `x`: sample locations, ordered with high-fidelity samples first.
  - `y`: observed values corresponding to `x`.
  - `lb`: lower bound of the input domain.
  - `ub`: upper bound of the input domain.

# Keywords

  - `num_high_fidel`: number of leading samples treated as high fidelity. It
    must leave at least one sample on each side of the split.
  - `low_fid_structure`: named-tuple surrogate configuration for low-fidelity
    data.
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
    eps_structure::H
end

# Build the surrogate a `*Structure` named tuple describes. Both fidelity levels
# and `update!` need exactly this dispatch, so it lives in one place rather than
# being spelled out once per call site.
#
# `GEKStructure` is absent deliberately: `GEK` needs `n(1 + d)` observations,
# values followed by gradients, and a variable-fidelity design carries only
# function values — the split by sample count would slice a gradient block in
# half. It is rejected below with the other unsupported names.
function _variable_fidelity_surrogate(structure, x, y, lb, ub)
    name = structure.name
    return if name == "RadialBasis"
        RadialBasis(
            x, y, lb, ub, rad = structure.radial_function,
            scale_factor = structure.scale_factor, sparse = structure.sparse
        )
    elseif name == "Kriging"
        Kriging(x, y, lb, ub, p = structure.p, theta = structure.theta)
    elseif name == "LinearSurrogate"
        LinearSurrogate(x, y, lb, ub)
    elseif name == "InverseDistanceSurrogate"
        InverseDistanceSurrogate(x, y, lb, ub, p = structure.p)
    elseif name == "LobachevskySurrogate"
        LobachevskySurrogate(
            x, y, lb, ub, alpha = structure.alpha, n = structure.n,
            sparse = structure.sparse
        )
    elseif name == "NeuralSurrogate"
        NeuralSurrogate(
            x, y, lb, ub, model = structure.model, loss = structure.loss,
            opt = structure.opt, n_epochs = structure.n_epochs
        )
    elseif name == "XGBoostSurrogate"
        XGBoostSurrogate(x, y, lb, ub, num_round = structure.num_round)
    elseif name == "SecondOrderPolynomialSurrogate"
        SecondOrderPolynomialSurrogate(x, y, lb, ub)
    elseif name == "Wendland"
        Wendland(
            x, y, lb, ub, eps = structure.eps, maxiters = structure.maxiters,
            tol = structure.tol
        )
    else
        throw(
            ArgumentError(
                "VariableFidelitySurrogate does not support a $(name) component. " *
                    "Supported: RadialBasis, Kriging, LinearSurrogate, " *
                    "InverseDistanceSurrogate, LobachevskySurrogate, NeuralSurrogate, " *
                    "XGBoostSurrogate, SecondOrderPolynomialSurrogate, Wendland."
            )
        )
    end
end

# The residuals the correction surrogate is fitted to. Recomputed by `update!`,
# since every change to `low_fid_surr` changes what is left for it to explain.
function _variable_fidelity_residuals(low_fid_surr, x_high, y_high)
    return [y_high[i] - low_fid_surr(x_high[i]) for i in eachindex(x_high)]
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
    # Both surrogates need samples of their own; an empty split otherwise
    # reaches the inner constructor as a `BoundsError` naming neither side.
    if num_high_fidel < 1 || num_high_fidel >= length(x)
        throw(
            ArgumentError(
                "num_high_fidel must leave samples on both sides of the split: " *
                    "expected 1 to $(length(x) - 1) for $(length(x)) samples, got " *
                    "$(num_high_fidel)."
            )
        )
    end

    x_high, x_low = x[1:num_high_fidel], x[(num_high_fidel + 1):end]
    y_high, y_low = y[1:num_high_fidel], y[(num_high_fidel + 1):end]

    low_fid_surr = _variable_fidelity_surrogate(low_fid_structure, x_low, y_low, lb, ub)
    y_eps = _variable_fidelity_residuals(low_fid_surr, x_high, y_high)
    eps_surr = _variable_fidelity_surrogate(high_fid_structure, x_high, y_eps, lb, ub)

    return VariableFidelitySurrogate(
        x, y, lb, ub, num_high_fidel, low_fid_surr, eps_surr, high_fid_structure
    )
end

function (varfid::VariableFidelitySurrogate)(val)
    return varfid.eps_surr(val) + varfid.low_fid_surr(val)
end

"""
    update!(varfid::VariableFidelitySurrogate, x_new, y_new)

Add low-fidelity samples and refit.

The new observations extend the low-fidelity surrogate. The correction surrogate
is then refitted as well: it was fitted to `y_high - low_fid_surr(x_high)`, so
once `low_fid_surr` moves, those residuals describe a low-fidelity surrogate that
no longer exists and the sum of the two stops reproducing the high-fidelity data.

# Returns

Returns `nothing`.
"""
function SurrogatesBase.update!(varfid::VariableFidelitySurrogate, x_new, y_new)
    varfid.x, varfid.y = _append_samples(varfid.x, varfid.y, x_new, y_new)
    update!(varfid.low_fid_surr, x_new, y_new)

    nhf = varfid.num_high_fidel
    x_high, y_high = varfid.x[1:nhf], varfid.y[1:nhf]
    y_eps = _variable_fidelity_residuals(varfid.low_fid_surr, x_high, y_high)
    varfid.eps_surr = _variable_fidelity_surrogate(
        varfid.eps_structure, x_high, y_eps, varfid.lb, varfid.ub
    )
    return nothing
end
