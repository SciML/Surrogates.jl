using Surrogates

#1D
n = 10
lb = 0.0
ub = 10.0
x = sample(n, lb, ub, SobolSample())
f = x -> 2 * x
y = f.(x)
my_varfid = VariableFidelitySurrogate(x, y, lb, ub)
val = my_varfid(3.0)
update!(my_varfid, 3.0, 6.0)
val = my_varfid(3.0)

my_varfid_change_struct = VariableFidelitySurrogate(x, y, lb, ub, num_high_fidel = 2,
    low_fid_structure = InverseDistanceStructure(p = 1.0),
    high_fid_structure = RadialBasisStructure(radial_function = linearRadial(),
        scale_factor = 1.0,
        sparse = false))
#ND
n = 10
lb = [0.0, 0.0]
ub = [5.0, 5.0]
x = sample(n, lb, ub, SobolSample())
f = x -> x[1] * x[2]
y = f.(x)
my_varfidND = VariableFidelitySurrogate(x, y, lb, ub)
val = my_varfidND((2.0, 2.0))
update!(my_varfidND, (3.0, 3.0), 9.0)
my_varfidND_change_struct = VariableFidelitySurrogate(x, y, lb, ub, num_high_fidel = 2,
    low_fid_structure = InverseDistanceStructure(p = 1.0),
    high_fid_structure = RadialBasisStructure(radial_function = linearRadial(),
        scale_factor = 1.0,
        sparse = false))
