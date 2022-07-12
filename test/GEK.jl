using Surrogates

#1D
n = 10
lb = 0.0
ub = 5.0
x = sample(n, lb, ub, SobolSample())
f = x -> x^2
y1 = f.(x)
der = x -> 2 * x
y2 = der.(x)
y = vcat(y1, y2)

my_gek = GEK(x, y, lb, ub)
val = my_gek(2.0)
std_err = std_error_at_point(my_gek, 1.0)
add_point!(my_gek, 2.5, 2.5^2)

# Test that input dimension is properly checked for 1D GEK surrogates
@test_throws ArgumentError my_gek(Float64[])
@test_throws ArgumentError my_gek((2.0, 3.0, 4.0))


#ND
n = 10
d = 2
lb = [0.0, 0.0]
ub = [5.0, 5.0]
x = sample(n, lb, ub, SobolSample())
f = x -> x[1]^2 + x[2]^2
y1 = f.(x)
grad1 = x -> 2 * x[1]
grad2 = x -> 2 * x[2]
function create_grads(n, d, grad1, grad2, y)
    c = 0
    y2 = zeros(eltype(y[1]), n * d)
    for i in 1:n
        y2[i + c] = grad1(x[i])
        y2[i + c + 1] = grad2(x[i])
        c = c + 1
    end
    return y2
end
y2 = create_grads(n, d, grad1, grad2, y)
y = vcat(y1, y2)

my_gek_ND = GEK(x, y, lb, ub)
val = my_gek_ND((1.0, 1.0))
std_err = std_error_at_point(my_gek_ND, (1.0, 1.0))
add_point!(my_gek_ND, (2.0, 2.0), 8.0)

# Test that input dimension is properly checked for ND GEK surrogates
@test_throws ArgumentError my_gek_ND(Float64[])
@test_throws ArgumentError my_gek_ND(2.0)
@test_throws ArgumentError my_gek_ND((2.0, 3.0, 4.0))
