# Cantilever beam function
The Cantilever Beam function is defined as:
``f(w,t) = \frac{4L^3}{Ewt}*\sqrt{ (\frac{Y}{t^2})^2 + (\frac{X}{w^2})^2 }``
With parameters L,E,X and Y given.

Let's import Surrogates and Plots:
```@example beam
using Surrogates
using Plots
default()
```

Define the objective function:
```@example beam
function f(x)
    t = x[1]
    w = x[2]
    L = 100.0
    E = 2.770674127819261e7
    X = 530.8038576066307
    Y = 997.8714938733949
    return (4*L^3)/(E*w*t)*sqrt( (Y/t^2)^2 + (X/w^2)^2)
end
```

Let's plot it:
```@example beam
n = 100
lb = [1.0,1.0]
ub = [8.0,8.0]
xys = sample(n,lb,ub,SobolSample());
zs = f.(xys);
x, y = 0:8, 0:8
p1 = surface(x, y, (x1,x2) -> f((x1,x2)))
xs = [xy[1] for xy in xys]
ys = [xy[2] for xy in xys]
scatter!(xs, ys, zs) # hide
p2 = contour(x, y, (x1,x2) -> f((x1,x2)))
scatter!(xs, ys)
plot(p1, p2, title="True function")
```


Fitting different Surrogates:
```@example beam
mypoly = PolynomialChaosSurrogate(xys, zs,  lb, ub)
loba = PolynomialChaosSurrogate(xys, zs,  lb, ub)
rad = RadialBasis(xys,zs,lb,ub)
```

Plotting:
```@example beam
p1 = surface(x, y, (x, y) -> mypoly([x y]))
scatter!(xs, ys, zs, marker_z=zs)
p2 = contour(x, y, (x, y) -> mypoly([x y]))
scatter!(xs, ys, marker_z=zs)
plot(p1, p2, title="Polynomial expansion")
```

```@example beam
p1 = surface(x, y, (x, y) -> loba([x y]))
scatter!(xs, ys, zs, marker_z=zs)
p2 = contour(x, y, (x, y) -> loba([x y]))
scatter!(xs, ys, marker_z=zs)
plot(p1, p2, title="Lobachevsky")
```

```@example beam
p1 = surface(x, y, (x, y) -> rad([x y]))
scatter!(xs, ys, zs, marker_z=zs)
p2 = contour(x, y, (x, y) -> rad([x y]))
scatter!(xs, ys, marker_z=zs)
plot(p1, p2, title="Inverse distance surrogate")
```
