module SurrogatesPolyChaos

using SurrogatesBase
using PolyChaos

export PolynomialChaosSurrogate, update!

mutable struct PolynomialChaosSurrogate{X, Y, L, U, C, O, N} <:
               AbstractDeterministicSurrogate
    x::X
    y::Y
    lb::L
    ub::U
    coeff::C
    orthopolys::O
    num_of_multi_indexes::N
end

function PolynomialChaosSurrogate(x, y, lb::Number, ub::Number;
        orthopolys::AbstractCanonicalOrthoPoly = GaussOrthoPoly(2))
    n = length(x)
    poly_degree = orthopolys.deg
    num_of_multi_indexes = 1 + poly_degree
    if n < 2 + 3 * num_of_multi_indexes
        error("To avoid numerical problems, it's strongly suggested to have at least $(2+3*num_of_multi_indexes) samples")
    end
    coeff = _calculatepce_coeff(x, y, num_of_multi_indexes, orthopolys)
    return PolynomialChaosSurrogate(x, y, lb, ub, coeff, orthopolys, num_of_multi_indexes)
end

function PolynomialChaosSurrogate(x, y, lb, ub;
        orthopolys = MultiOrthoPoly([GaussOrthoPoly(2) for j in 1:length(lb)], 2))
    n = length(x)
    d = length(lb)
    poly_degree = orthopolys.deg
    num_of_multi_indexes = binomial(d + poly_degree, poly_degree)
    if n < 2 + 3 * num_of_multi_indexes
        error("To avoid numerical problems, it's strongly suggested to have at least $(2+3*num_of_multi_indexes) samples")
    end
    coeff = _calculatepce_coeff(x, y, num_of_multi_indexes, orthopolys)
    return PolynomialChaosSurrogate(x, y, lb, ub, coeff, orthopolys, num_of_multi_indexes)
end

function (pc::PolynomialChaosSurrogate)(val::Number)
    return sum([pc.coeff[i] * PolyChaos.evaluate(val, pc.orthopolys)[i]
                for i in 1:(pc.num_of_multi_indexes)])
end

function (pcND::PolynomialChaosSurrogate)(val)
    sum = zero(eltype(val))
    for i in 1:(pcND.num_of_multi_indexes)
        sum = sum +
              pcND.coeff[i] *
              first(PolyChaos.evaluate(pcND.orthopolys.ind[i, :], collect(val),
            pcND.orthopolys))
    end
    return sum
end

function _calculatepce_coeff(
        x, y, num_of_multi_indexes, orthopolys::AbstractCanonicalOrthoPoly)
    n = length(x)
    A = zeros(eltype(x), n, num_of_multi_indexes)
    for i in 1:n
        A[i, :] = PolyChaos.evaluate(x[i], orthopolys)
    end
    return (A' * A) \ (A' * y)
end

function _calculatepce_coeff(x, y, num_of_multi_indexes, orthopolys::MultiOrthoPoly)
    n = length(x)
    d = length(x[1])
    A = zeros(eltype(x[1]), n, num_of_multi_indexes)
    for i in 1:n
        xi = zeros(eltype(x[1]), d)
        for j in 1:d
            xi[j] = x[i][j]
        end
        A[i, :] = PolyChaos.evaluate(xi, orthopolys)
    end
    return (A' * A) \ (A' * y)
end

function SurrogatesBase.update!(polych::PolynomialChaosSurrogate, x_new, y_new)
    polych.x = vcat(polych.x, x_new)
    polych.y = vcat(polych.y, y_new)
    polych.coeff = _calculatepce_coeff(polych.x, polych.y, polych.num_of_multi_indexes,
        polych.orthopolys)
    nothing
end

end # module
