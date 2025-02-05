mutable struct VariableFidelitySurrogate{X, Y, L, U, N, F, E} <:
               AbstractDeterministicSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    num_high_fidel::N
    low_fid_surr::F
    eps_surr::E
end

function VariableFidelitySurrogate(x, y, lb, ub;
        num_high_fidel = Int(floor(length(x) / 2)),
        low_fid_structure = RadialBasisStructure(radial_function = linearRadial(),
            scale_factor = 1.0,
            sparse = false),
        high_fid_structure = RadialBasisStructure(radial_function = cubicRadial(),
            scale_factor = 1.0,
            sparse = false))
    x_high = x[1:num_high_fidel]
    x_low = x[(num_high_fidel + 1):end]
    y_high = y[1:num_high_fidel]
    y_low = y[(num_high_fidel + 1):end]

    #Fit low fidelity surrogate:
    if low_fid_structure[1] == "RadialBasis"
        #fit and append to local_surr
        low_fid_surr = RadialBasis(x_low, y_low, lb, ub,
            rad = low_fid_structure.radial_function,
            scale_factor = low_fid_structure.scale_factor,
            sparse = low_fid_structure.sparse)

    elseif low_fid_structure[1] == "Kriging"
        low_fid_surr = Kriging(x_low, y_low, lb, ub, p = low_fid_structure.p,
            theta = low_fid_structure.theta)

    elseif low_fid_structure[1] == "GEK"
        low_fid_surr = GEK(x_low, y_low, lb, ub, p = low_fid_structure.p,
            theta = low_fid_structure.theta)

    elseif low_fid_structure == "LinearSurrogate"
        low_fid_surr = LinearSurrogate(x_low, y_low, lb, ub)

    elseif low_fid_structure[1] == "InverseDistanceSurrogate"
        low_fid_surr = InverseDistanceSurrogate(x_low, y_low, lb, ub,
            p = low_fid_structure.p)

    elseif low_fid_structure[1] == "LobachevskySurrogate"
        low_fid_surr = LobachevskySurrogate(x_low, y_low, lb, ub,
            alpha = low_fid_structure.alpha,
            n = low_fid_structure.n,
            sparse = low_fid_structure.sparse)

    elseif low_fid_structure[1] == "NeuralSurrogate"
        low_fid_surr = NeuralSurrogate(x_low, y_low, lb, ub,
            model = low_fid_structure.model,
            loss = low_fid_structure.loss,
            opt = low_fid_structure.opt,
            n_epochs = low_fid_structure.n_epochs)

    elseif low_fid_structure[1] == "RandomForestSurrogate"
        low_fid_surr = RandomForestSurrogate(x_low, y_low, lb, ub,
            num_round = low_fid_structure.num_round)

    elseif low_fid_structure == "SecondOrderPolynomialSurrogate"
        low_fid_surr = SecondOrderPolynomialSurrogate(x_low, y_low, lb, ub)

    elseif low_fid_structure[1] == "Wendland"
        low_fid_surr = Wendand(x_low, y_low, lb, ub, eps = low_fid_surr.eps,
            maxiters = low_fid_surr.maxiters, tol = low_fid_surr.tol)
    else
        throw("A surrogate with the name provided does not exist or is not currently supported with VariableFidelity")
    end

    #Fit surrogate eps on high fidelity data with objective function y_high - low_find_surr
    y_eps = zeros(eltype(y), num_high_fidel)
    @inbounds for i in 1:num_high_fidel
        y_eps[i] = y_high[i] - low_fid_surr(x_high[i])
    end

    if high_fid_structure[1] == "RadialBasis"
        #fit and append to local_surr
        eps = RadialBasis(x_high, y_eps, lb, ub, rad = high_fid_structure.radial_function,
            scale_factor = high_fid_structure.scale_factor,
            sparse = high_fid_structure.sparse)

    elseif high_fid_structure[1] == "Kriging"
        eps = Kriging(x_high, y_eps, lb, ub, p = high_fid_structure.p,
            theta = high_fid_structure.theta)

    elseif high_fid_structure == "LinearSurrogate"
        eps = LinearSurrogate(x_high, y_eps, lb, ub)

    elseif high_fid_structure[1] == "InverseDistanceSurrogate"
        eps = InverseDistanceSurrogate(x_high, y_eps, lb, ub, high_fid_structure.p)

    elseif high_fid_structure[1] == "LobachevskySurrogate"
        eps = LobachevskySurrogate(x_high, y_eps, lb, ub, alpha = high_fid_structure.alpha,
            n = high_fid_structure.n,
            sparse = high_fid_structure.sparse)

    elseif high_fid_structure[1] == "NeuralSurrogate"
        eps = NeuralSurrogate(x_high, y_eps, lb, ub, model = high_fid_structure.model,
            loss = high_fid_structure.loss, opt = high_fid_structure.opt,
            n_epochs = high_fid_structure.n_epochs)

    elseif high_fid_structure[1] == "RandomForestSurrogate"
        eps = RandomForestSurrogate(x_high, y_eps, lb, ub,
            num_round = high_fid_structure.num_round)

    elseif high_fid_structure == "SecondOrderPolynomialSurrogate"
        eps = SecondOrderPolynomialSurrogate(x_high, y_eps, lb, ub)

    elseif high_fid_structure[1] == "Wendland"
        eps = Wendand(x_high, y_eps, lb, ub, eps = high_fid_structure.eps,
            maxiters = high_fid_structure.maxiters, tol = high_fid_structure.tol)
    else
        throw("A surrogate with the name provided does not exist or is not currently supported with VariableFidelity")
    end
    return VariableFidelitySurrogate(x, y, lb, ub, num_high_fidel, low_fid_surr, eps)
end

#=
function (varfid::VariableFidelitySurrogate)(val::Number)
    return varfid.eps_surr(val) + varfid.low_fid_surr(val)
end

"""
VariableFidelitySurrogate(x,y,lb,ub;
                                   num_high_fidel = Int(floor(length(x)/2))
                                   low_fid = RadialBasisStructure(radial_function = linearRadial, scale_factor=1.0, sparse = false),
                                   high_fid = RadialBasisStructure(radial_function = cubicRadial ,scale_factor=1.0,sparse=false))
First section (1:num_high_fidel) of samples are high fidelity, second section are low fidelity
"""
function VariableFidelitySurrogate(x,y,lb,ub;
                                   num_high_fidel = Int(floor(length(x)/2))
                                   low_fid = RadialBasisStructure(radial_function = linearRadial, scale_factor=1.0, sparse = false),
                                   high_fid = RadialBasisStructure(radial_function = cubicRadial ,scale_factor=1.0,sparse=false))

end
=#

function (varfid::VariableFidelitySurrogate)(val)
    return varfid.eps_surr(val) + varfid.low_fid_surr(val)
end

"""
add_point!(varfid::VariableFidelitySurrogate,x_new,y_new)

I expect to add low fidelity data to the surrogate.
"""
function SurrogatesBase.update!(varfid::VariableFidelitySurrogate, x_new, y_new)
    if length(varfid.x[1]) == 1
        #1D
        varfid.x = vcat(varfid.x, x_new)
        varfid.y = vcat(varfid.y, y_new)

        #I added a new lowfidelity datapoint, I need to update the low_fid_surr:
        update!(varfid.low_fid_surr, x_new, y_new)
    else
        #ND
        varfid.x = vcat(varfid.x, x_new)
        varfid.y = vcat(varfid.y, y_new)

        #I added a new lowfidelity datapoint, I need to update the low_fid_surr:
        update!(varfid.low_fid_surr, x_new, y_new)
    end
end
