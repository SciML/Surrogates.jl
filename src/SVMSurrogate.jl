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

function SVMSurrogate(x,y,lb,ub)
    xn = vcat(map(x->x', x)...)
    model = LIBSVM.fit!(SVC(),xn,y)
    SVMSurrogate(xn,y,model,lb,ub)
end

function (svmsurr::SVMSurrogate)(val)
    return LIBSVM.predict(svmsurr.model,val)
end

function add_point!(svmsurr::SVMSurrogate,x_new,y_new)
     if length(svmsurr.lb) == 1
         #1D
         svmsurr.x = vcat(svmsurr.x,x_new)
         svmsurr.y = vcat(svmsurr.y,y_new)
         svmsurr.model = LIBSVM.fit!(SVC(),reshape(svmsurr.x,length(svmsurr.x),1),svmsurr.y)
     else
         #ND
         svmsurr.x = vcat(svmsurr.x,x_new)
         svmsurr.y = vcat(svmsurr.y,y_new)
         svmsurr.model = LIBSVM.fit!(SVC(),svmsurr.x,svmsurr.y)
     end
     nothing
 end
