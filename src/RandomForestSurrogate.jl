using XGBoost

mutable struct RandomForestSurrogate{X,Y,B,L,U}
    x::X
    y::Y
    bst::B
    lb::L
    ub::U
end

function RandomForestSurrogate(x,y,lb::Number,ub::Number)
    num_round = 2
    bst = xgboost(reshape(x,length(x),1), num_round, label = y)
    RandomForestSurrogate(x,y,bst,lb,ub)
end

function (rndfor::RandomForestSurrogate)(val::Number)
    return XGBoost.predict(rndfor.bst,reshape([val],1,1))
end

function RandomForestSurrogate(x,y,lb,ub)
    x = vcat(map(x->x', x)...)
    num_round = 2
    bst = xgboost(x, num_round, label = y)
    RandomForestSurrogate(x,y,bst,lb,ub)
end

function (rndfor::RandomForestSurrogate)(val)
    return XGBoost.predict(rndfor.bst,val)
end

function add_point!(rndfor::RandomForestSurrogate,x_new,y_new)
    if length(rndfor.lb) == 1
        #1D
        rndfor.x = vcat(rndfor.x,x_new)
        rndfor.y = vcat(rndfor.y,y_new)
        num_round = 2
        rndfor.bst = xgboost(reshape(rndfor.x,length(rndfor.x),1), num_round, label = rndfor.y)
    else
        #ND
        rndfor.x = vcat(rndfor.x,x_new)
        rndfor.y = vcat(rndfor.y,y_new)
        num_round = 2
        rndfor.bst = xgboost(rndfor.x, num_round, label = rndfor.y)
    end
    nothing
end
