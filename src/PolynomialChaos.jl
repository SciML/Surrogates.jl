using PolyChaos

mutable struct PolynomialChaosSurrogate{X,Y,L,U,C,O,N} <: AbstractSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    coeff::C
    ortopolys::O
    num_of_multi_indexes::N
 end


function _calculatepce_coeff(x,y,num_of_multi_indexes,op::AbstractCanonicalOrthoPoly)
    n = length(x)
    A = zeros(eltype(x),n,num_of_multi_indexes)
    for i = 1:n
        A[i,:] = PolyChaos.evaluate(x[i],op)
    end
    return (A'*A)\(A'*y)
end

function PolynomialChaosSurrogate(x,y,lb::Number,ub::Number; op::AbstractCanonicalOrthoPoly = GaussOrthoPoly(2))
    n = length(x)
    poly_degree = op.deg
    num_of_multi_indexes = 1+poly_degree
    if n < 2+3*num_of_multi_indexes
        throw("To avoid numerical problems, it's strongly suggested to have at least $(2+3*num_of_multi_indexes) samples")
    end
    coeff = _calculatepce_coeff(x,y,num_of_multi_indexes,op)
    return PolynomialChaosSurrogate(x,y,lb,ub,coeff,op,num_of_multi_indexes)

end

function (pc::PolynomialChaosSurrogate)(val::Number)
    return sum([pc.coeff[i]*PolyChaos.evaluate(val,pc.ortopolys)[i] for i = 1:pc.num_of_multi_indexes])
end


function _calculatepce_coeff(x,y,num_of_multi_indexes,op::MultiOrthoPoly)
    n = length(x)
    d = length(x[1])
    A = zeros(eltype(x[1]),n,num_of_multi_indexes)
    for i = 1:n
        xi = zeros(eltype(x[1]),d)
        for j = 1:d
            xi[j] = x[i][j]
        end
        A[i,:] = PolyChaos.evaluate(xi,op)
    end
    return (A'*A)\(A'*y)
end


function PolynomialChaosSurrogate(x,y,lb,ub; op::MultiOrthoPoly = MultiOrthoPoly([GaussOrthoPoly(2) for j = 1:length(lb)],2))
    n = length(x)
    d = length(lb)
    poly_degree = op.deg
    num_of_multi_indexes = binomial(d+poly_degree,poly_degree)
    if n < 2+3*num_of_multi_indexes
        throw("To avoid numerical problems, it's strongly suggested to have at least $(2+3*num_of_multi_indexes) samples")
    end
    coeff = _calculatepce_coeff(x,y,num_of_multi_indexes,op)
    return PolynomialChaosSurrogate(x,y,lb,ub,coeff,op,num_of_multi_indexes)

end

function (pcND::PolynomialChaosSurrogate)(val)
    return sum([pcND.coeff[i]*PolyChaos.evaluate(collect(val),pcND.ortopolys)[i] for i = 1:pcND.num_of_multi_indexes])
end


function add_point!(polych::PolynomialChaosSurrogate,x_new,y_new)
    if length(polych.lb) == 1
        #1D
        polych.x = vcat(polych.x,x_new)
        polych.y = vcat(polych.y,y_new)
        polych.coeff = _calculatepce_coeff(polych.x,polych.y,polych.num_of_multi_indexes,polych.ortopolys)
    else
        polych.x = vcat(polych.x,x_new)
        polych.y = vcat(polych.y,y_new)
        polych.coeff = _calculatepce_coeff(polych.x,polych.y,polych.num_of_multi_indexes,polych.ortopolys)
    end
    nothing
end
