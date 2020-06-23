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
    if k != length(local_kind)
        throw("Number of mixtures = $k is not equal to length of local surrogates")
    end
    n = length(x)
    # find weight, mean and variance for each mixture
    # For GaussianMixtures I need nxd matrix
    X_G = reshape(x,(n,1))
    moe_gmm = GaussianMixtures.GMM(k,X_G)
    weights = GaussianMixtures.weights(moe_gmm)
    means = GaussianMixtures.means(moe_gmm)
    variances = moe_gmm.Σ


    #cluster the points
    #For clustering I need dxn matrix
    X_C = reshape(x,(1,n))
    KNN = kmeans(X_C, k)
    x_c = [ Array{eltype(x)}(undef,0) for i = 1:k]
    y_c = [ Array{eltype(y)}(undef,0) for i = 1:k]
    a = assignments(KNN)
    @inbounds for i = 1:n
        pos = a[i]
        append!(x_c[pos],x[i])
        append!(y_c[pos],y[i])
    end

    local_surr = Dict()
    for i = 1:k
        if local_kind[i][1] == "RadialBasis"
            #fit and append to local_surr
            my_local_i = RadialBasis(x_c[i],y_c[i],lb,ub,rad = local_kind[i].radial_function, scale_factor = local_kind[i].scale_factor, sparse = local_kind[i].sparse)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "Kriging"
            my_local_i = Kriging(x_c[i], y_c[i],lb,ub, p = local_kind[i].p, theta = local_kind[i].theta)
            local_surr[i] = my_local_i

        elseif local_kind[i] == "LinearSurrogate"
            my_local_i = LinearSurrogate(x_c[i], y_c[i],lb,ub)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "InverseDistanceSurrogate"
            my_local_i = InverseDistanceSurrogate(x_c[i], y_c[i],lb,ub, local_kind[i].p)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "LobacheskySurrogate"
            my_local_i = LobacheskySurrogate(x_c[i], y_c[i],lb,ub,alpha = local_kind[i].alpha , n = local_kind[i].n, sparse = local_kind[i].sparse)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "NeuralSurrogate"
            my_local_i = NeuralSurrogate(x_c[i], y_c[i],lb,ub, model = local_kind[i].model , loss = local_kind[i].loss ,opt = local_kind[i].opt ,n_echos = local_kind[i].n_echos)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "RandomForestSurrogate"
            my_local_i = RandomForestSurrogate(x_c[i], y_c[i],lb,ub, num_round = local_kind[i].num_round)
            local_surr[i] = my_local_i

        elseif local_kind[i] == "SecondOrderPolynomialSurrogate"
            my_local_i = SecondOrderPolynomialSurrogate(x_c[i], y_c[i],lb,ub)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "Wendland"
            my_local_i = Wendand(x_c[i], y_c[i],lb,ub, eps = local_kind[i].eps, maxiters = local_kind[i].maxiters, tol = local_kind[i].tol)
            local_surr[i] = my_local_i
        else
            throw("A surrogate with name provided does not exist or is not currently supported with MOE.")
        end
    end
    return MOE(x,y,lb,ub,local_surr,k,means,variances,weights)
end

function MOE(x,y,lb,ub; k::Int = 2,
            local_kind = [RadialBasisStructure(radial_function = linearRadial, scale_factor=1.0, sparse = false),RadialBasisStructure(radial_function = cubicRadial, scale_factor=1.0, sparse = false)])

    n = length(x)
    d = length(lb)
    #GMM parameters:
    X_G = collect(reshape(collect(Base.Iterators.flatten(x)), (d,n))')
    my_gmm = GMM(k,X_G,kind = :full)
    weights = my_gmm.w
    means = my_gmm.μ
    varcov = my_gmm.Σ

    #cluster the points
    X_C = collect(reshape(collect(Base.Iterators.flatten(x)), (d,n)))
    KNN = kmeans(X_C, k)
    x_c = [ Array{eltype(x)}(undef,0) for i = 1:k]
    y_c = [ Array{eltype(y)}(undef,0) for i = 1:k]
    a = assignments(KNN)
    @inbounds for i = 1:n
        pos = a[i]
        x_c[pos] = vcat(x_c[pos],x[i])
        append!(y_c[pos],y[i])
    end

    local_surr = Dict()
    for i = 1:k
        if local_kind[i][1] == "RadialBasis"
            #fit and append to local_surr
            my_local_i = RadialBasis(x_c[i],y_c[i],lb,ub,rad = local_kind[i].radial_function, scale_factor = local_kind[i].scale_factor, sparse = local_kind[i].sparse)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "Kriging"
            my_local_i = Kriging(x_c[i], y_c[i],lb,ub, p = local_kind[i].p, theta = local_kind[i].theta)
            local_surr[i] = my_local_i

        elseif local_kind[i] == "LinearSurrogate"
            my_local_i = LinearSurrogate(x_c[i], y_c[i],lb,ub)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "InverseDistanceSurrogate"
            my_local_i = InverseDistanceSurrogate(x_c[i], y_c[i],lb,ub, local_kind[i].p)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "LobacheskySurrogate"
            my_local_i = LobacheskySurrogate(x_c[i], y_c[i],lb,ub,alpha = local_kind[i].alpha , n = local_kind[i].n, sparse = local_kind[i].sparse)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "NeuralSurrogate"
            my_local_i = NeuralSurrogate(x_c[i], y_c[i],lb,ub, model = local_kind[i].model , loss = local_kind[i].loss ,opt = local_kind[i].opt ,n_echos = local_kind[i].n_echos)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "RandomForestSurrogate"
            my_local_i = RandomForestSurrogate(x_c[i], y_c[i],lb,ub, num_round = local_kind[i].num_round)
            local_surr[i] = my_local_i

        elseif local_kind[i] == "SecondOrderPolynomialSurrogate"
            my_local_i = SecondOrderPolynomialSurrogate(x_c[i], y_c[i],lb,ub)
            local_surr[i] = my_local_i

        elseif local_kind[i][1] == "Wendland"
            my_local_i = Wendand(x_c[i], y_c[i],lb,ub, eps = local_kind[i].eps, maxiters = local_kind[i].maxiters, tol = local_kind[i].tol)
            local_surr[i] = my_local_i
        else
            throw("A surrogate with name "* local_kind[i][1] *" does not exist or is not currently supported with MOE.")
        end
    end
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
    if length(moe.x[1]) == 1
        #1D
        moe.x = vcat(moe.x,x_new)
        moe.y = vcat(moe.y,y_new)
        n = length(moe.x)

        #New mixture parameters
        X_G = reshape(moe.x,(n,1))
        moe_gmm = GaussianMixtures.GMM(moe.k,X_G)
        moe.weights = GaussianMixtures.weights(moe_gmm)
        moe.means = GaussianMixtures.means(moe_gmm)
        moe.varcov = moe_gmm.Σ

        #Find cluster of new point(s):
        n_added = length(x_new)
        X_C = reshape(moe.x,(1,n))
        KNN = kmeans(X_C, moe.k)
        a = assignments(KNN)
        #Recalculate only relevant surrogates
        for i = 1:n_added
            pos = a[n-n_added+i]
            add_point!(moe.local_surr[i],moe.x[n-n_added+i],moe.y[n-n_added+i])
        end
    else
        #ND
        moe.x = vcat(moe.x,x_new)
        moe.y = vcat(moe.y,y_new)
        n = length(moe.x)
        d = length(moe.lb)
        #New mixture parameters
        X_G = collect(reshape(collect(Base.Iterators.flatten(moe.x)), (d,n))')
        my_gmm = GMM(moe.k,X_G,kind = :full)
        moe.weights = my_gmm.w
        moe.means = my_gmm.μ
        moe.varcov = my_gmm.Σ

        #cluster the points
        X_C = collect(reshape(collect(Base.Iterators.flatten(moe.x)), (d,n)))
        KNN = kmeans(X_C, moe.k)
        a = assignments(KNN)
        n_added = length(x_new)
        for i = 1:n_added
            pos = a[n-n_added+i]
            add_point!(moe.local_surr[i],moe.x[n-n_added+i],moe.y[n-n_added+i])
        end
    end
    nothing
end
