mutable struct EarthSurrogate{X,Y,L,U,P,C} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    penalty::P
    coeff::C

 end



function EarthSurrogate(x,y,lb::Number,ub::Number; penalty::Number = 2.0)

    return EarthSurrogate()
end




function (earth::EarthSurrogate)(val::Number)



end




function EarthSurrogate(x,y,lb,ub; penalty::Number = 2.0)


end




function (earth::EarthSurrogate)(val)


end



function add_point!(earth::EarthSurrogate,xnew,ynew)
    ... standard

end
