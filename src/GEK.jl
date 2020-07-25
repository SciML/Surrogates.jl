
mutable struct GEK{X,Y,L,U,P,T,M,B,S,R} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    p::P
    theta::T
    mu::M
    b::B
    sigma::S
    inverse_of_R::R
 end


function GEK(x,y,lb::number,ub::number; )



end





function GEK(x,y,lb,ub;)



end

function (gek::GEK)(val::Number)


end




function (gek::GEK)(val)


end


function add_point!(k::GEK,new_x,new_y)
    if new_x in k.x
        println("Adding a sample that already exists, cannot build Kriging.")
        return
    end
    if (length(new_x) == 1 && length(new_x[1]) == 1) || ( length(new_x) > 1 && length(new_x[1]) == 1 && length(k.theta)>1)
        push!(k.x,new_x)
        push!(k.y,new_y)
    else
        append!(k.x,new_x)
        append!(k.y,new_y)
    end
    #k.mu,k.b,k.sigma,k.inverse_of_R = _calc_kriging_coeffs(k.x,k.y,k.p,k.theta)
    nothing
end
