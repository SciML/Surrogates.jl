mutable struct RandomForestSurrogate{X,Y,B,L,U,N} <: AbstractSurrogate
    x::X
    y::Y
    bst::B
    lb::L
    ub::U
    num_round::N
end

function RandomForestSurrogate(x,y,lb::Number,ub::Number,num_round)
    bst = xgboost(reshape(x,length(x),1), num_round, label = y)
    RandomForestSurrogate(x,y,bst,lb,ub,num_round)
end

function (rndfor::RandomForestSurrogate)(val::Number)
    return XGBoost.predict(rndfor.bst,reshape([val],1,1))
end

function RandomForestSurrogate(x,y,lb,ub,num_round)
    X = Array{Float64,2}(undef,length(x),length(x[1]))
    for j = 1:length(x)
        X[j,:] = vec(collect(x[j]))
    end
    bst = xgboost(X, num_round, label = y)
    RandomForestSurrogate(x,y,bst,lb,ub,num_round)
end

function (rndfor::RandomForestSurrogate)(val)
    return XGBoost.predict(rndfor.bst,reshape(collect(val),1,2))[1]
end

function add_point!(rndfor::RandomForestSurrogate,x_new,y_new)
    if length(rndfor.lb) == 1
        #1D
        rndfor.x = vcat(rndfor.x,x_new)
        rndfor.y = vcat(rndfor.y,y_new)
        rndfor.bst = xgboost(reshape(rndfor.x,length(rndfor.x),1), rndfor.num_round, label = rndfor.y)
    else
        n_previous = length(rndfor.x)
        a = vcat(rndfor.x,x_new)
        n_after = length(a)
        dim_new = n_after - n_previous
        n = length(rndfor.x)
        d = length(rndfor.x[1])
        tot_dim = n + dim_new
        X = Array{Float64,2}(undef,tot_dim,d)
        for j = 1:n
            X[j,:] = vec(collect(rndfor.x[j]))
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
        rndfor.x = vcat(rndfor.x,x_new)
        rndfor.y = vcat(rndfor.y,y_new)
        rndfor.bst = xgboost(X, rndfor.num_round, label = rndfor.y)
    end
    nothing
end
