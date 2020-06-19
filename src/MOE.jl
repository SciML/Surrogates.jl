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


RadialStructure(p=2,k=3) = (name = "Radial",p = p, k = k) #do this for each surrogate available

function MOE(x,y,lb::number,ub::number; k = 2, local_kind = [RadialStructure() for i = 1:k])
    #cluster the points


    #fit each cluster with a Surrogate


    #find varcov matrix for each mixture


    return MOE(..)
end

function MOE(x,y,lb,ub; k = 2, local_kind = [RadialStructure() for i = 1:k])
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
