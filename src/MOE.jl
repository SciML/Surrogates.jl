using Clustering
using GaussianMixtures
mutable struct MOE{X,Y,L,U,S,K,V,C} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    local_surr::S
    k::K
    varcov::V
    x_clustered::C
end


#Radial structure:
function RadialBasisStructure(;radial_function,scale_factor)
    return (name = "RadialBasis", radial_function = radial_function, scale_factor = scale_factor)
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
function LobacheskyStructure(;alpha,n)
    return (name = "LobacheskySurrogate", alpha = alpha, n = n)
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


function MOE(x,y,lb::number,ub::number; k::Int = 2,
            local_kind = [RadialStructure(radial_function = linearRadial, scale_factor=1.0),RadialStructure(radial_function = cubicRadial, scale_factor=1.0)])
    #cluster the points


    #fit each cluster with a Surrogate


    #find varcov matrix for each mixture


    return MOE(..)
end

function MOE(x,y,lb,ub; k::Int = 2,
            local_kind = [RadialStructure(radial_function = linearRadial, scale_factor=1.0),RadialStructure(radial_function = cubicRadial, scale_factor=1.0)])
    #cluster the points


    #fit each cluster with a Surrogate


    #find varcov matrix for each mixture


    return MOE(..)
end


function (moe::MOE)(val::number)
    #compyte formula with var covar matrix

end

function (moe::MOE)(val)


end
function add_point!(moe::MOE,x_new,y_new)
    #classic things




    nothing
end
