# Water flow function

The water flow function is defined as:
``f(r_w,r,T_u,H_u,T_l,H_l,L,K_w) = \frac{2*\pi*T_u(H_u - H_l)}{log(\frac{r}{r_w})*[1 + \frac{2LT_u}{log(\frac{r}{r_w})*r_w^2*K_w}+ \frac{T_u}{T_l} ]}``

It has 8 dimension.

```@example water
using Surrogates
using Plots
using LinearAlgebra
default()
```

Define the objective function:
```@example water
function f(x)
    r_w = x[1]
    r = x[2]
    T_u = x[3]
    H_u = x[4]
    T_l = x[5]
    H_l = x[6]
    L = x[7]
    K_w = x[8]
    log_val = log(r/r_w)
    return (2*pi*T_u*(H_u - H_l))/ ( log_val*(1 + (2*L*T_u/(log_val*r_w^2*K_w)) + T_u/T_l))
end
```


```@example water
n = 180
d = 8
lb = [0.05,100,63070,990,63.1,700,1120,9855]
ub = [0.15,50000,115600,1110,116,820,1680,12045]
x = sample(n,lb,ub,SobolSample())
y = f.(x)
n_test = 1000
x_test = sample(n_test,lb,ub,GoldenSample());
y_true = f.(x_test);
```


```@example water
my_rad = RadialBasis(x,y,lb,ub)
y_rad = my_rad.(x_test)
my_poly = PolynomialChaosSurrogate(x,y,lb,ub)
y_poli = my_poli.(x_test)
mse_rad = norm(y_true - y_rad,2)/n_test
mse_poli = norm(y_true - y_poli,2)/n_test
print("MSE Radial: $mse_rad")
print("MSE Radial: $mse_poli")
```
