mutable struct GEKPLS{X,Y,L,U,C, A, B,G} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    coeff_pls::C
    alpha::A
    beta::B
    gamma::G
  end
  
function GEKPLS(x,y,lb,ub, theta, extra_points, n_comp)
    println("GEKPLS constructor called") 
    # parameters = 25.0 #vik
    # bb = boxbehnken(3, 1)
    # println("printing bb below")
    # println(bb)
    # println("printed bb above")
    C,A,B,G = 1,2,3,4
    return GEKPLS(x,y,lb,ub,A,C,B,G)
end
  
function add_point!(GEKPLS,x_new,y_new)  
    nothing
end
  
function (s::GEKPLS)(value)
    println("GEKPLS predictor called")
    return s.coeff*value
    #return s.coeff*value + s.alpha
end

function _ge_compute_pls()
    nothing
end

function _standardization()
    nothing
end

function _cross_distances()
    nothing
end

function _componentwise_distance_PLS()
    nothing
end

function _reduced_likelihood_function()
    nothing
end

function _check_param()
    nothing
end




######start of bbdesign######

    # 
    # Adapted from 'ExperimentalDesign.jl: Design of Experiments in Julia'
    # https://github.com/phrb/ExperimentalDesign.jl

    # MIT License

    # ExperimentalDesign.jl: Design of Experiments in Julia
    # Copyright (C) 2019 Pedro Bruel <pedro.bruel@gmail.com>

    # Permission is hereby granted, free of charge,  to any person obtaining a copy of
    # this software  and associated documentation  files (the "Software"), to  deal in
    # the Software  without restriction,  including without  limitation the  rights to
    # use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    # the Software, and to permit persons to  whom the Software is furnished to do so,
    # subject to the following conditions:

    # The  above copyright  notice  and  this permission  notice  (including the  next
    # paragraph)  shall be  included  in all  copies or  substantial  portions of  the
    # Software.

    # THE  SOFTWARE IS  PROVIDED "AS  IS", WITHOUT  WARRANTY OF  ANY KIND,  EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    # FOR A PARTICULAR  PURPOSE AND NONINFRINGEMENT. IN NO EVENT  SHALL THE AUTHORS OR
    # COPYRIGHT HOLDERS BE  LIABLE FOR ANY CLAIM, DAMAGES OR  OTHER LIABILITY, WHETHER
    # IN  AN ACTION  OF  CONTRACT, TORT  OR  OTHERWISE,  ARISING FROM,  OUT  OF OR  IN
    # CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    # 

function boxbehnken(matrix_size::Int)
    boxbehnken(matrix_size, matrix_size) #we should probably call this with second param 1
end

function boxbehnken(matrix_size::Int, center::Int)
    @assert matrix_size>=3

    A_fact = explicit_fullfactorial(Tuple([-1,1] for i = 1:2))
    println("A_fact: $A_fact")
    rows = floor(Int, (0.5*matrix_size*(matrix_size-1))*size(A_fact)[1])
    println("rows: $rows")
    A = zeros(rows, matrix_size)

    l = 0
    for i in 1:matrix_size-1
        for j in i+1:matrix_size
            l = l +1
            A[max(0, (l - 1)*size(A_fact)[1])+1:l*size(A_fact)[1], i] = A_fact[:, 1]
            A[max(0, (l - 1)*size(A_fact)[1])+1:l*size(A_fact)[1], j] = A_fact[:, 2]
        end
    end

    if center == matrix_size
        if matrix_size <= 16
            points = [0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16]
            center = points[matrix_size]
        end
    end

    A = transpose(hcat(transpose(A), transpose(zeros(center, matrix_size))))
end

function explicit_fullfactorial(factors::Tuple)
    println("factors from explicit_fullfactorial: $factors")
    explicit_fullfactorial(fullfactorial(factors))
end

function explicit_fullfactorial(iterator::Base.Iterators.ProductIterator)
    println("iterator from explicit_fullfactorial: $iterator")
    hcat(vcat.(collect(iterator)...)...)
end

function fullfactorial(factors::Tuple)
    println("factors from fullfactorial: $factors")
    Base.Iterators.product(factors...)
end

#bb = boxbehnken(3, 1);
######end of bb design######

##PLS1 start
    #Adapted from PartialLeastSquresRegressor.jl 
    #https://github.com/lalvim/PartialLeastSquaresRegressor.jl
    # MIT License

    # Copyright (c) 2020 Leandro Alvim

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

## the learning algorithm: PLS1 - single target
function trainer(X, Y, nfactors) 
    ncols = size(X)[2]
    W = zeros(ncols,nfactors)
    b = zeros(1,nfactors)
    P  = zeros(ncols,nfactors)
    for i = 1:nfactors
        W[:,i] = X'Y
        W[:,i] /= norm(W[:,i])#sqrt.(W[:,i]'*W[:,i])
        R      = X*W[:,i]
        Rn     = R'/(R'R) # change to use function...
        P[:,i] = Rn*X
        b[i]   = Rn * Y
        if abs(b[i]) <= 1e-3
            print("PLS1 converged. No need learning with more than $(i) factors")
            nfactors = i
            break
         end
        X      = X - R * P[:,i]'
        Y      = Y - R * b[i]
    end
    return (W,b,P)
end
##end of PLS1