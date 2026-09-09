# Parallel optimization

There are some situations where it can be beneficial to run multiple optimizations in parallel. For example, if your objective function is very expensive to evaluate, you may want to run multiple evaluations in parallel.

```@docs
potential_optimal_points
MinimumConstantLiar
MeanConstantLiar
MaximumConstantLiar
KrigingBeliever
KrigingBelieverUpperBound
KrigingBelieverLowerBound
```

## Ask-Tell Interface

To enable parallel optimization, we make use of an Ask-Tell interface. The user will construct the initial surrogate model the same way as for non-parallel surrogate models, but instead of using `surrogate_optimize!`, the user will use `potential_optimal_points`. This will return the coordinates of points that the optimizer has determined are most useful to evaluate next. How the user evaluates these points is up to them. The Ask-Tell interface requires more manual control than `surrogate_optimize!`, but it allows for more flexibility. After the point has been evaluated, the user will *tell* the surrogate model the new points with the `update!` function.

## Virtual points

To ensure that points of interest returned by `potential_optimal_points` are sufficiently far from each other, the function makes use of *virtual points*. They are used as follows:

 1. `potential_optimal_points` is told to return `n` points.
 2. The best-scoring candidate is selected. `SRBF` minimizes its merit function, `EI` maximizes expected improvement.
 3. This point is now treated as a virtual point: it is added to a temporary copy of the surrogate with an assigned value, which changes the acquisition landscape. How that value is chosen depends on the strategy used (see below). The surrogate you passed in is never modified.
 4. The best-scoring candidate under the updated temporary surrogate is selected. Candidates within the minimum-separation tolerance of an already-chosen point are rejected, so a batch never repeats a point.
 5. The process is repeated until `n` points have been selected.

The following strategies are available for virtual point selection for all optimization algorithms:

  - "Minimum Constant Liar (MinimumConstantLiar)":
    
      + The virtual point is assigned the lowest observed objective value.

  - "Mean Constant Liar (MeanConstantLiar)":
    
      + The virtual point is assigned the mean of the observed objective values.
  - "Maximum Constant Liar (MaximumConstantLiar)":
    
      + The virtual point is assigned the greatest observed objective value.

For Kriging surrogates, specifically, the above and following strategies are available:

  - "Kriging Believer (KrigingBeliever):
    
      + The virtual point is assigned the Kriging mean at that point, predicted by the temporary surrogate, so each belief accounts for the ones already placed in this batch.

  - "Kriging Believer Upper Bound (KrigingBelieverUpperBound)":
    
      + The virtual point is assigned 3$\sigma$ above the temporary surrogate's mean at that point.
  - "Kriging Believer Lower Bound (KrigingBelieverLowerBound)":
    
      + The virtual point is assigned 3$\sigma$ below the temporary surrogate's mean at that point.

In general, MinimumConstantLiar and KrigingBelieverLowerBound tend to favor exploitation, while MaximumConstantLiar and KrigingBelieverUpperBound tend to favor exploration. MeanConstantLiar and KrigingBeliever tend to be compromises between the two.

## Examples

```@example parallel
using Surrogates

lb = 0.0
ub = 10.0
f = x -> log(x) * exp(x)
x = sample(5, lb, ub, SobolSample())
y = f.(x)

my_k = Kriging(x, y, lb, ub)

for _ in 1:10
    new_x,
    eis = potential_optimal_points(
        EI(), MeanConstantLiar(), lb, ub, my_k, SobolSample(), 3)
    update!(my_k, new_x, f.(new_x))
end
```
