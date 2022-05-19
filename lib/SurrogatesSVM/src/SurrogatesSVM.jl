module SurrogatesSVM

import Surrogates: AbstractSurrogate, add_point!
export SVMSurrogate

using LIBSVM

mutable struct SVMSurrogate{X,Y,M,L,U} <: AbstractSurrogate
    x::X
    y::Y
    model::M
    lb::L
    ub::U
end


function SVMSurrogate(x,y,lb::Number,ub::Number)
    xn = reshape(x,length(x),1)
    model = LIBSVM.fit!(SVC(),xn,y)
    SVMSurrogate(xn,y,model,lb,ub)
end

function (svmsurr::SVMSurrogate)(val::Number)
    return LIBSVM.predict(svmsurr.model,[val])
end

"""
SVMSurrogate(x,y,lb,ub)

Builds SVM surrogate.
"""
function SVMSurrogate(x,y,lb,ub)
    X = Array{Float64,2}(undef,length(x),length(x[1]))
    for j = 1:length(x)
        X[j,:] = vec(collect(x[j]))
    end
    model = LIBSVM.fit!(SVC(),X,y)
    SVMSurrogate(x,y,model,lb,ub)
end

function (svmsurr::SVMSurrogate)(val)
    n = length(val)
    return LIBSVM.predict(svmsurr.model,reshape(collect(val),1,n))[1]
end

function add_point!(svmsurr::SVMSurrogate,x_new,y_new)
     if length(svmsurr.lb) == 1
         #1D
         svmsurr.x = vcat(svmsurr.x,x_new)
         svmsurr.y = vcat(svmsurr.y,y_new)
         svmsurr.model = LIBSVM.fit!(SVC(),reshape(svmsurr.x,length(svmsurr.x),1),svmsurr.y)
     else
         n_previous = length(svmsurr.x)
         a = vcat(svmsurr.x,x_new)
         n_after = length(a)
         dim_new = n_after - n_previous
         n = length(svmsurr.x)
         d = length(svmsurr.x[1])
         tot_dim = n + dim_new
         X = Array{Float64,2}(undef,tot_dim,d)
         for j = 1:n
             X[j,:] = vec(collect(svmsurr.x[j]))
         end
         if dim_new == 1
             X[n+1,:] = vec(collect(x_new))
         else
             i = 1
             for j = n+1:tot_dim
                 X[j,:] = vec(collect(x_new[i]))
                 i = i + 1
             end
         end
         svmsurr.x = vcat(svmsurr.x,x_new)
         svmsurr.y = vcat(svmsurr.y,y_new)
         svmsurr.model = LIBSVM.fit!(SVC(),X,svmsurr.y)
     end
     nothing
 end
end # module
