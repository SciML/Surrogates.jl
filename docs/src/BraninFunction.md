## Branin Function

The Branin Function is commonly used as a test function for metamodelling in computer experiments, especially in the context of optimization.

The expression of the Branin Function is given as:
``f(x) = (x_2 - \frac{5.1}{4\pi^2}x_1^{2} + \frac{5}{\pi}x_1 - 6)^2 + 10(1-\frac{1}{8\pi})\cos(x_1) + 10``

where ``x = (x_1, x_2)`` with ``-5\leq x_1 \leq 10, 0 \leq x_2 \leq 15``

First of all we will import these two packages `Surrogates` and `Plots`.

```@example BraninFunction
using Surrogates
using Plots
default()
```

Now, let's define our objective function:

```@example BraninFunction
function branin(x)
      x1 = x[1]
      x2 = x[2]
      b = 5.1 / (4*pi^2);
      c = 5/pi;
      r = 6;
      a = 1;
      s = 10;
      t = 1 / (8*pi);
      term1 = a * (x2 - b*x1^2 + c*x1 - r)^2;
      term2 = s*(1-t)*cos(x1);
      y = term1 + term2 + s;
end
```
Now, let's plot it:

```@example BraninFunction
n_samples = 80
lower_bound = [-5, 0]
upper_bound = [10,15]
xys = sample(n_samples, lower_bound, upper_bound, SobolSample())
zs = branin.(xys);
x, y = -5:10, 0:15 # hide
p1 = surface(x, y, (x1,x2) -> branin((x1,x2))) # hide
xs = [xy[1] for xy in xys] # hide
ys = [xy[2] for xy in xys] # hide
scatter!(xs, ys, zs) # hide
p2 = contour(x, y, (x1,x2) -> branin((x1,x2))) # hide
scatter!(xs, ys) # hide
plot(p1, p2, title="True function") # hide
```

Now it's time to fitting different surrogates and then we will plot them.
We will have a look on `Kriging Surrogate`:

```@example BraninFunction
kriging_surrogate = Kriging(xys, zs, lower_bound, upper_bound, p=[1.9, 1.9])
```

```@example BraninFunction
p1 = surface(x, y, (x, y) -> kriging_surrogate([x y])) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> kriging_surrogate([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2, title="Kriging Surrogate") # hide
```

Now, we will have a look on `Inverse Distance Surrogate`:
```@example BraninFunction
InverseDistance = InverseDistanceSurrogate(xys, zs,  lower_bound, upper_bound)
```

```@example BraninFunction
p1 = surface(x, y, (x, y) -> InverseDistance([x y])) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> InverseDistance([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2, title="Inverse Distance Surrogate") # hide
```

Now, let's talk about `Lobachesky Surrogate`:
```@example BraninFunction
Lobachesky = LobacheskySurrogate(xys, zs,  lower_bound, upper_bound, alpha = [2.8,2.8], n=8
```

```@example BraninFunction
p1 = surface(x, y, (x, y) -> Lobachesky([x y])) # hide
scatter!(xs, ys, zs, marker_z=zs) # hide
p2 = contour(x, y, (x, y) -> Lobachesky([x y])) # hide
scatter!(xs, ys, marker_z=zs) # hide
plot(p1, p2, title="Lobachesky Surrogate") # hide
```
