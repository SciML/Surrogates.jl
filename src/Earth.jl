using LinearAlgebra

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
    rel_res_error::R
    rel_GCV::G
    intercept::I
 end

 #1D
 _hinge(x::Number,knot::Number) = max(0,x-knot)
 _hinge_mirror(x::Number,knot::Number) = max(0,knot-x)

 #ND
 #inside arr_hing I have functions like g(x) = x -> _hinge(x,5.0) or g(x) = one(x)
 _product_hinge(val,arr_hing) = prod([arr_hing[i](val[i]) for i = 1:length(val)])

 function _coeff_1d(x,y,basis)
     n = length(x)
     d = length(basis)
     X = zeros(eltype(x[1]),n,d)
     @inbounds for i = 1:n
         for j = 1:d
             X[i,j] = basis[j](x[i])
         end
     end
     return (X'*X)\(X'*y)
 end

 function _forward_pass_1d(x,y,n_max_terms,rel_res_error)
     n = length(x)
     basis = Array{Function}(undef,0)
     current_sse = +Inf
     intercept = sum([y[i] for i =1:length(y)])/length(y)
     num_terms = 0
     for var_i in x
        #Add or not add the knot var_i?
        new_basis = copy(basis)
        #select best new pair
        hinge1 = x-> _hinge(x,var_i)
        hinge2 = x-> _hinge_mirror(x,var_i)
        push!(new_basis,hinge1)
        push!(new_basis,hinge2)
        #find coefficients
        d = length(new_basis)
        X = zeros(eltype(x[1]),n,d)
        @inbounds for i = 1:n
            for j = 1:d
                X[i,j] = new_basis[j](x[i])
            end
        end
        if (cond(X'*X) > 1e8)
            condition_number = false
            new_sse = +Inf
        else
            condition_number = true
            coeff = (X'*X)\(X'*y)
            new_sse = zero(y[1])
            d = length(new_basis)
            for i = 1:n
                val_i = sum(coeff[j]*new_basis[j](x[i]) for j = 1:d) + intercept
                new_sse = new_sse + (y[i]-val_i)^2
            end
        end
        if ( (new_sse < current_sse) && (abs(current_sse - new_sse) >= rel_res_error) && condition_number)
            #Add the hinge function to the basis
            num_terms = num_terms+1
            push!(basis,hinge1)
            push!(basis,hinge2)
            current_sse = new_sse
        end
        if (num_terms > n_max_terms)
            break
        end
    end
    return basis
 end

 function _backward_pass_1d(x,y,n_min_terms,basis,penalty,rel_GCV)
     n = length(x)
     d = length(basis)
     intercept = sum([y[i] for i =1:length(y)])/length(y)
     coeff = _coeff_1d(x,y,basis)
     sse = zero(y[1])
     for i = 1:n
         val_i = sum(coeff[j]*basis[j](x[i]) for j = 1:d) + intercept
         sse = sse + (y[i]-val_i)^2
     end
     current_gcv = sse/(n*(1-d/n)^2)
     num_terms = d
     while (num_terms > n_min_terms)
         #Basis-> select worst performing element-> eliminate it
         if num_terms <= 1
             break
         end
         found_new_to_eliminate = false
         for i = 1:num_terms
             current_basis = copy(basis)
             #remove i-esim element from current basis
             deleteat!(current_basis,i)
             coef = _coeff_1d(x,y,current_basis)
             new_sse = zero(y[i])
             for i = 1:n
                 val_i = sum(coeff[j]*basis[j](x[i]) for j = 1:d) + intercept
                 new_sse = new_sse + (y[i]-val_i)^2
             end
             i_gcv = new_sse/(n*(1-d/n)^2)
             if i_gcv < current_gcv
                 basis_to_remove = i
                 new_gcv = i_gcv
                 found_new_to_eliminate = true
             end
         end
         if !found_new_to_eliminate
             break
         end
         if abs(current_gcv-new_gcv) < rel_GCV
             break
         else
             num_terms = num_terms-1
             deleteat!(basis,basis_to_remove)
         end
     end
     return basis
 end


 function EarthSurrogate(x,y,lb::Number,ub::Number; penalty::Number = 2.0, n_min_terms::Int = 2, n_max_terms::Int = 10, rel_res_error::Number = 1e-2, rel_GCV::Number = 1e-2)
     intercept = sum([y[i] for i =1:length(y)])/length(y)
     basis_after_forward = _forward_pass_1d(x,y,n_max_terms,rel_res_error)
     basis = _backward_pass_1d(x,y,n_min_terms,basis_after_forward,penalty,rel_GCV)
     coeff = _coeff_1d(x,y,basis)
     return EarthSurrogate(x,y,lb,ub,basis,coeff,penalty,n_min_terms,n_max_terms,rel_res_error,rel_GCV,intercept)
 end

 function (earth::EarthSurrogate)(val::Number)
     return sum([earth.coeff[i]*earth.basis[i](val) for i = 1:length(earth.coeff)])+earth.intercept
 end

  function EarthSurrogate(x,y,lb,ub; penalty::Number = 2.0, n_min_terms::Int = 2, n_max_terms::Int = 10, rel_res_error::Number = 1e-2, rel_GCV::Number = 1e-2)
      return EarthSurrogate(x,y,lb,ub,1,2,3,4,5,6,7,10)
  end


function add_point!(earth::EarthSurrogate,x_new,y_new)
      if length(earth.x[1]) == 1
          #1D
          earth.x = vcat(earth.x,x_new)
          earth.y = vcat(earth.y,y_new)
          earth.intercept = sum([earth.y[i] for i =1:length(earth.y)])/length(earth.y)
          basis_after_forward = _forward_pass_1d(earth.x,earth.y,earth.n_max_terms,earth.rel_res_error)
          earth.basis = _backward_pass_1d(earth.x,earth.y,earth.n_min_terms,basis_after_forward,earth.penalty,earth.rel_GCV)
          earth.coeff = _coeff_1d(earth.x,earth.y,earth.basis)
          nothing
      else
          #ND
          earth.x = vcat(earth.x,x_new)
          earth.y = vcat(earth.y,y_new)
          #earth.intercept =
          #earth.basis =
          #earth.coeff =

      end


  end
