# Rosenbrock function
The Rosenbrock function is defined as:
``f(x) = \sum_{i=1}^{d-1}[ (x_{i+1}-x_i)^2 + (x_i - 1)^2]``

I will treat the 2D version, which is commonly defined as:
``f(x,y) = (1-x)^2 + 100(y-x^2)^2``
Let's import Surrogates and Plots:
```@example rosen
using Surrogates
using Plots
default()
```

Define the objective function:
```@example rosen
function f(x)
    x1 = x[1]
    x2 = x[2]
    return (1-x1)^2 + 100*(x2-x1^2)^2
end
```

Let's plot it:
```@example rosen
n = 100
lb = [0.0,0.0]
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
```@example rosen
mypoly = PolynomialChaosSurrogate(xys, zs,  lb, ub)
loba = PolynomialChaosSurrogate(xys, zs,  lb, ub)
inver = InverseDistanceSurrogate(xys, zs,  lb, ub)
```

Plotting:
```@example rosen
p1 = surface(x, y, (x, y) -> mypoly([x y]))
scatter!(xs, ys, zs, marker_z=zs)
p2 = contour(x, y, (x, y) -> mypoly([x y]))
scatter!(xs, ys, marker_z=zs)
plot(p1, p2, title="Polynomial expansion")
```

```@example rosen
p1 = surface(x, y, (x, y) -> loba([x y]))
scatter!(xs, ys, zs, marker_z=zs)
p2 = contour(x, y, (x, y) -> loba([x y]))
scatter!(xs, ys, marker_z=zs)
plot(p1, p2, title="Lobachesky")
```

```@example rosen
p1 = surface(x, y, (x, y) -> inver([x y]))
scatter!(xs, ys, zs, marker_z=zs)
p2 = contour(x, y, (x, y) -> inver([x y]))
scatter!(xs, ys, marker_z=zs)
plot(p1, p2, title="Inverse distance surrogate")
```
