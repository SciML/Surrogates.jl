using Clustering
using GaussianMixtures
using LinearAlgebra

mutable struct MOE{X,Y,L,U,S,K,M,V,W} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    local_surr::S
    k::K
    means::M
    varcov::V
    weights::W
end


#Radial structure:
function RadialBasisStructure(;radial_function,scale_factor,sparse)
    return (name = "RadialBasis", radial_function = radial_function, scale_factor = scale_factor, sparse = sparse)
end

#Kriging structure:
function KrigingStructure(;p,theta)
    return (name = "Kriging", p = p, theta = theta)
end

#Linear structure
function LinearStructure()
    return (name = "LinearSurrogate")
end

#InverseDistance structure
function InverseDistanceStructure(;p)
    return (name = "InverseDistanceSurrogate", p = p)
end

#Lobachesky structure
function LobacheskyStructure(;alpha,n,sparse)
    return (name = "LobacheskySurrogate", alpha = alpha, n = n, sparse = sparse)
end

function NeuralStructure(;model,loss,opt,n_echos)
    return (name ="NeuralSurrogate", model = model ,loss = loss,opt = opt,n_echos = n_echos)
end

function RandomForestStructure(;num_round)
    return (name = "RandomForestSurrogate", num_round = num_round)
end

function SecondOrderPolynomialStructure()
    return (name = "SecondOrderPolynomialSurrogate")
end

function WendlandStructure(; eps, maxiters, tol)
    return (name = "Wendland", eps = eps, maxiters = maxiters, tol = tol)
end


function MOE(x,y,lb::Number,ub::Number; k::Int = 2, local_kind = [RadialBasisStructure(radial_function = linearRadial, scale_factor=1.0,sparse = false),RadialBasisStructure(radial_function = cubicRadial, scale_factor=1.0, sparse = false)])
    n = length(x)
    # find weight, mean and variance for each mixture
    # For GaussianMixtures I need nxd matrix
    X_G = reshape(x,(n,1))
    moe_gmm = GaussianMixtures.GMM(k,x)
    weights = GaussianMixtures.weights(moe_gmm)
    means = GaussianMixtures.means(moe_gmm)
    variances = moe_gmm.Σ


    #cluster the points
    #For clustering I need dxn matrix
    X_C = reshape(x,(1,n))
    KNN = kmeans(X_C, k)
    x_c = [ [] for i = 1:k]
    y_c = [ [] for i = 1:k]
    assingnements = assignments(KNN)
    @inbounds for i = 1:n
        pos = assingnements[i]
        append!(x_c[pos],x[i])
        append!(y_c[pos],y[i])
    end

    local_surr = []
    for i = 1:k
        if local_kind[i].name == "RadialBasis"
            #fit and append to local_surr
            my_local_i = RadialBasis(x_c[i],y_c[i],lb,ub,rad = local_kind[i].radial_function, scale_factor = local_kind[i].scale_factor, sparse = local_kind[i].sparse)
            #problema qui
            append!(local_surr,my_local_i)

        elseif local_kind[i].name == "Kriging"
            my_local_i = Kriging(x_c[i], y_c[i],lb,ub, p = local_kind[i].p, theta = local_kind[i].p)
            append!(local_surr,my_local_i)

        elseif local_kind[i].name == "LinearSurrogate"
            my_local_i = LinearSurrogate(x_c[i], y_c[i],lb,ub)
            append!(local_surr,my_local_i)

        elseif local_kind[i].name == "InverseDistanceSurrogate"
            my_local_i = InverseDistanceSurrogate(x_c[i], y_c[i],lb,ub, local_kind[i].p)
            append!(local_surr,my_local_i)

        elseif local_kind[i].name == "LobacheskySurrogate"
            my_local_i = LobacheskySurrogate(x_c[i], y_c[i],lb,ub,alpha = local_kind[i].alpha , n = local_kind[i].n, sparse = local_kind[i].sparse)
            append!(local_surr,my_local_i)

        elseif local_kind[i].name == "NeuralSurrogate"
            my_local_i = NeuralSurrogate(x_c[i], y_c[i],lb,ub, model = local_kind[i].model , loss = local_kind[i].loss ,opt = local_kind[i].opt ,n_echos = local_kind[i].n_echos)
            append!(local_surr,my_local_i)

        elseif local_kind[i].name == "RandomForestSurrogate"
            my_local_i = RandomForestSurrogate(x_c[i], y_c[i],lb,ub, num_round = local_kind[i].num_round)
            append!(local_surr,my_local_i)

        elseif local_kind[i].name == "SecondOrderPolynomialSurrogate"
            my_local_i = SecondOrderPolynomialSurrogate(x_c[i], y_c[i],lb,ub)
            append!(local_surr,my_local_i)

        elseif local_kind[i].name == "Wendland"
            my_local_i = Wendand(x_c[i], y_c[i],lb,ub, eps = local_kind[i].eps, maxiters = local_kind[i].maxiters, tol = local_kind[i].tol)
            append!(local_surr,my_local_i)
        else
            throw("A surrogate with name "* local_kind[i].name *" does not exist or is not currently supported with MOE.")
        end
    end
    return MOE(x,y,lb,ub,local_surr,k,means,varcov,weights)
end

function MOE(x,y,lb,ub; k::Int = 2,
            local_kind = [RadialBasisStructure(radial_function = linearRadial, scale_factor=1.0, sparse = false),RadialBasisStructure(radial_function = cubicRadial, scale_factor=1.0, sparse = false)])


    #find varcov matrix for each mixture
    X_G = collect(reshape(collect(Base.Iterators.flatten(x)), (length(x[1]),length(x)))')

    #cluster the points
    X_C = collect(reshape(collect(Base.Iterators.flatten(x)), (length(x[1]),length(x))))


    #fit each cluster with a Surrogate


    #find varcov matrix for each mixture


    return MOE(x,y,lb,ub,local_surr,k,means,varcov,weights)
end


function _prob_x_in_i(x::Number,i,mu,varcov,alpha,k)
    num = (1/sqrt(varcov[i]))*alpha[i]*exp(-0.5(x-mu[i])*(1/varcov[i])*(x-mu[i]))
    den = sum([(1/sqrt(varcov[j]))*alpha[j]*exp(-0.5(x-mu[j])*(1/varcov[j])*(x-mu[j]))  for j = 1:k])
    return num/den
end

function _prob_x_in_i(x,i,mu,varcov,alpha,k)
    num = (1/sqrt(det(varcov[i])))*alpha[i]*exp(-0.5*(x .- mu[i,:])'*(inv(varcov[i]))*(x .- mu[i,:]))
    den = sum([(1/sqrt(det(varcov[j])))*alpha[j]*exp(-0.5*(x .- mu[j,:])'*(inv(varcov[j]))*(x .- mu[j,:])) for j = 1:k])
    return num/den
end

function (moe::MOE)(val)
    return prod([moe.local_surr[i](val)*_prob_x_in_i(val,i,moe.means,moe.varcov,moe.weights,moe.k) for i = 1:moe.k])
end


function add_point!(moe::MOE,x_new,y_new)
    #classic things




    nothing
end
