# Parallel Optimization

There are some situations where it can be beneficial to run multiple optimizations in parallel. For example, if your objective function is very expensive to evaluate, you may want to run multiple evaluations in parallel. 

## Ask-Tell Interface

To enable parallel optimization, we make use of an Ask-Tell interface. The user will construct the initial surrogate model the same way as for non-parallel surrogate models, but instead of using `surrogate_optimize`, the user will use `potential_optimal_points`. This will return the coordinates of points that the optimizer has determined are most useful to evaluate next. How the user evaluates these points is up to them. The Ask-Tell interface requires more manual control than `surrogate_optimize`, but it allows for more flexibility. After the point has been evaluated, the user will *tell* the surrogate model the new points with the `add_point!` function.

## Virtual Points

To ensure that points of interest returned by `potential_optimal_points` are sufficiently far from each other, the function makes use of *virtual points*. They are used as follows:
1. `potential_optimal_points` is told to return `n` points.
2. The point with the highest merit function value is selected.
3. This point is now treated as a virtual point and is assigned a temporary value that changes the landscape of the merit function. How the temporary value is chosen depends on the strategy used. (see below)
4. The point with the new highest merit is selected.
5. The process is repeated until `n` points have been selected.

The following strategies are available for virtual point selection for all optimization algorithms:

- "Minimum Constant Liar (MinimumConstantLiar)":
  - The virtual point is assigned using the lowest known value of the merit function across all evaluated points.
- "Mean Constant Liar (MeanConstantLiar)":
  - The virtual point is assigned using the mean of the merit function across all evaluated points.
- "Maximum Constant Liar (MaximumConstantLiar)":
  - The virtual point is assigned using the greatest known value of the merit function across all evaluated points.

For Kriging surrogates, specifically, the above and following strategies are available:  

- "Kriging Believer (KrigingBeliever):
  - The virtual point is assigned using the mean of the Kriging surrogate at the virtual point.
- "Kriging Believer Upper Bound (KrigingBelieverUpperBound)":
  - The virtual point is assigned using 3$\sigma$ above the mean of the Kriging surrogate at the virtual point.
- "Kriging Believer Lower Bound (KrigingBelieverLowerBound)":
  - The virtual point is assigned using 3$\sigma$ below the mean of the Kriging surrogate at the virtual point.


In general, MinimumConstantLiar and KrigingBelieverLowerBound tend to favor exploitation, while MaximumConstantLiar and KrigingBelieverUpperBound tend to favor exploration. MeanConstantLiar and KrigingBeliever tend to be compromises between the two.

## Examples

```@example
using Surrogates

lb = 0.0
ub = 10.0
f = x -> log(x) * exp(x)
x = sample(5, lb, ub, SobolSample())
y = f.(x)

my_k = Kriging(x, y, lb, ub)

for _ in 1:10
    new_x, eis = potential_optimal_points(EI(), MeanConstantLiar(), lb, ub, my_k, SobolSample(), 3)
    add_point!(my_k, new_x, f.(new_x))
end
```
