#Welded beam function

The welded beam function is defined as:
``f(h,l,t) = \sqrt{\frac{a^2 + b^2 + abl}{\sqrt{0.25(l^2+(h+t)^2)}}}``
With:
``a = \frac{6000}{\sqrt{2}hl}``
``b = \frac{6000(14 + 0.5l)*\sqrt{0.25(l^2+(h+t)^2)}}{2*[0.707hl(\frac{l^2}{12}+0.25*(h+t)^2)]}``

It has 3 dimension.

```@example welded
using Surrogates
using Plots
using LinearAlgebra
default()
```

Define the objective function:
```@example welded
function f(x)
    h = x[1]
    l = x[2]
    t = x[3]
    a = 6000/(sqrt(2)*h*l)
    b = (6000*(14+0.5*l)*sqrt(0.25*(l^2+(h+t)^2)))/(2*(0.707*h*l*(l^2/12 + 0.25*(h+t)^2)))
    return (sqrt(a^2+b^2 + l*a*b))/(sqrt(0.25*(l^2+(h+t)^2)))
end
```


```@example welded
n = 300
d = 3
lb = [0.125,5.0,5.0]
ub = [1.,10.,10.]
x = sample(n,lb,ub,SobolSample())
y = f.(x)
n_test = 1000
x_test = sample(n_test,lb,ub,GoldenSample());
y_true = f.(x_test);
```


```@example welded
my_rad = RadialBasis(x,y,lb,ub)
y_rad = my_rad.(x_test)
mse_rad = norm(y_true - y_rad,2)/n_test
print("MSE Radial: $mse_rad")

my_krig = Kriging(x,y,lb,ub)
y_krig = my_krig.(x_test)
mse_krig = norm(y_true - y_krig,2)/n_test
print("MSE Kriging: $mse_krig")

my_loba = LobacheskySurrogate(x,y,lb,ub)
y_loba = my_loba.(x_test)
mse_rad = norm(y_true - y_loba,2)/n_test
print("MSE Lobachesky: $mse_rad")
```
