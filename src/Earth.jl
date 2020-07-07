mutable struct EarthSurrogate{X,Y,L,U,B,C,P,M,N,R,G,I} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    basis::B
    coeff::C
    penalty::P
    n_min_terms::M
    n_max_terms::N
    rel_sse::R
    rel_GCV::G
    intercept::I
 end


#1D
_hinge(x::Number,knot::Number) = max(0,x-knot)
_hinge_mirror(x::Number,knot::Number) = max(0,knot-x)

#ND
#inside arr_hing I have functions like g(x) = x -> _hinge(x,5.0) or g(x) = one(x)
_product_hinge(val,arr_hing) = prod([arr_hing[i](val[i]) for i = 1:length(val)])


function _forward_pass_1d(x,y,n_max_terms,rel_res_error)
    n = length(x)
    basis = Array{Function}(undef,0)
    current_sse = +Inf
    intercept = sum([y[i] for i =1:length(y)])/length(y)
    num_terms = 0
    while (num_terms < n_max_terms)
        #In 1D I just need to iterate over the knots because there are no interaction terms
        for var_i in x
            new_basis = basis
            #select best new pair
            hinge1 = x-> _hinge(x,var_i)
            hinge2 = x-> _hinge_mirror(x,var_i)
            push!(new_basis,hinge1)
            push!(new_basis,hinge2)
            #find coefficients
            d = length(new_basis)
            X = zeros(eltype(hinge1(x[1])),n,d)
            coef = coef(lm(X,y))
            new_sse = zero(y[i])
            for i = 1:n
                val_i = sum(coeff[j]*new_basis[j](x[i]) for j = 1:d) + intercept
                new_sse = new_sse + (y[i]-val_i)^2
            end
            if (new_sse < current_sse)
                better_basis = new_basis
                new_max_sse = new_sse
                new_hinge1 = hinge1
                new_hinge2 = hinge2
            end

        end
        if (abs(current_sse - new_max_sse) < rel_res_error)
            break;
        else
            push!(basis,new_hinge1)
            push!(basis,new_hinge2)
        end
        num_terms = num_terms+1
    end

    return basis
end

function _backward_pass_1d(x,y,n_min_terms,basis,penalty,rel_GCV)
    n = length(x)
    current_gcv = tobecalculated
    intercept = sum([y[i] for i =1:length(y)])/length(y)
    num_terms = length(basis)
    while (num_terms > n_min_terms)

        for (base_to_remove in basis)
            #look for best basis to remove

            if min as<d.zd<
                #select best basis to remove
            end
        end

        if abs(gcv_current-gcv_new) < rel_GCV
            break
        else
            num_terms = num_terms-1
            #delete the basis that we picked and move on
        end

    end


    return basis
end

function EarthSurrogate(x,y,lb::Number,ub::Number; penalty::Number = 2.0, n_min_terms::Int = 2, n_max_terms::Int = 10, rel_sse::Number = 1e-2, rel_GCV::Number = 1e-2)

    intercept = sum([y[i] for i =1:length(y)])/length(y)
    basis_after_forward = _forward_pass_1d(x,y,n_max_terms,rel_res_error)
    basis = _backward_pass_1d(x,y,n_min_terms,basis,penalty,rel_GCV)
    #find coef
    #build matrixes for linear model
    coef =
    return EarthSurrogate(x,y,lb,ub,basis,coeff,penalty,n_min_terms,n_max_terms,rel_sse,rel_GCV,intercept)
end




function (earth::EarthSurrogate)(val::Number)
    return sum([earth.coeff[i]*earth.basis[i](val) for i = 1:length(earth.coeff)])+earth.intercept
end




function _forward_pass_nd(x,y,n_max_terms,rel_sse)

    basis = Array{Function}(undef,0)
    current_sse = +Inf
    while (num_terms < n_max_terms)
    for existing terms
        for variables
            for values of variables




end

function _backward_pass_nd(x,y,basis,penalty)



end


function EarthSurrogate(x,y,lb,ub; penalty::Number = 2.0, n_min_terms::Int = length(lb), n_max::Int = length(lb)*10, rel_res_error::Number = 1e-2, rel_GCV::Number = 1e-2)
    intercept = sum([y[i] for i =1:length(y)])/length(y)

end



function (earth::EarthSurrogate)(val)
    return sum([earth.coeff[i]*earth.basis[i](val) for i = 1:length(earth.coeff)]) + earth.intercept

end



function add_point!(earth::EarthSurrogate,xnew,ynew)
    if length(varfid.x[1]) == 1
        #1D
        earth.x = vcat(earth.x,x_new)
        earth.y = vcat(earth.y,y_new)


    else
        #ND
        earth.x = vcat(earth.x,x_new)
        earth.y = vcat(earth.y,y_new)

    end


end
