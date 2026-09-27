---
title: 'Surrogates.jl: Surrogate Modeling and Surrogate-Based Optimization in Julia'
tags:
  - Julia
  - Surrogate Modeling
authors:
  - name: Sathvik Bhagavan
    orcid: 0000-0003-0785-3586
    corresponding: true
    affiliation: 1
  - name: Ludovico Bessi
    affiliation: 4
  - name: Vikramaditya Narayan
    affiliation: 5
  - name: Christopher Rackauckas
    orcid: 0000-0001-5850-0663
    affiliation: "2, 3"
affiliations:
  - name: EPFL, Switzerland
    index: 1
  - name: Massachusetts Institute of Technology, USA
    index: 2
  - name: JuliaHub, USA
    index: 3
  - name: Google, Switzerland
    index: 4
  - name: Precogly, India
    index: 5
date: 21 September 2026
bibliography: paper.bib
---

# Summary

Many scientific and engineering workflows depend on a function $f$ that is expensive to evaluate: a PDE solve, a finite-element simulation, or a physical experiment. *Surrogate modeling* replaces $f$ by a cheap approximation $g$ fitted to a small design of experiments, and then optimizes, explores, or differentiates $g$ instead.

`Surrogates.jl` gives a single Julia interface to a broad collection of surrogate families: radial basis functions [@buhmann2003rbf], compactly supported Wendland kernels [@wendland1995], Kriging [@sacks1989design; @jones1998efficient] together with its gradient-enhanced (`GEK` [@han2013gek], `GEKPLS` [@bouhlel2019gekpls]) and partial-least-squares variants (`KPLS`, `KPLSK` [@bouhlel2016kpls]), multivariate adaptive regression splines [@friedman1991multivariate], Lobachevsky splines [@allasia2013lobachevsky], inverse-distance (Shepard) interpolation [@shepard1968], linear and second-order polynomial models, and multi-fidelity correction surrogates [@forrester2007multifidelity]. Package extensions add mixtures of experts [@jordan1994hierarchical], neural and gradient-enhanced neural networks [@bouhlel2020gann] via `Flux.jl` [@Fluxjl], support vector machines via `LIBSVM` [@chang2011libsvm], gradient-boosted trees via `XGBoost` [@chen2016xgboost], polynomial chaos expansions via `PolyChaos.jl` [@muehlpfordt2020polychaos], and general Gaussian processes via `AbstractGPs.jl` [@AbstractGPsjl].

Every surrogate is an ordinary Julia callable: `s(x)` evaluates the approximation, `update!(s, x, y)` incorporates new observations without a full refit where the algorithm allows, and models with a predictive distribution (the Kriging family and Gaussian processes) implement `std_error_at_point`. The initial design is drawn with any sampler from `QuasiMonteCarlo.jl` [@QuasiMonteCarlojl] — Sobol [@sobol1967], Latin hypercube [@mckay1979lhs], Halton [@halton1960], and others — plus a `SectionSample` strategy that holds chosen coordinates fixed.

Beyond fitting, `Surrogates.jl` implements surrogate-based optimization through `surrogate_optimize!`: stochastic RBF search [@regis2007stochastic], DYCORS [@regis2013combining], lower confidence bound (`LCBS`) [@cox1992statistical; @srinivas2010gaussian] and expected improvement (`EI`) [@jones1998efficient] acquisition, the SOP algorithm for parallel search centers [@krityakierne2016sop], and two multi-objective loops — a surrogate-screened candidate selection (SMB) and the rolling tide evolutionary algorithm [@fieldsend2015rtea] for noisy objectives. Where $f$ can be evaluated at several points at once, an ask/tell entry point proposes a batch of candidates using constant-liar and Kriging-believer virtual points [@ginsbourger2010kriging], for the SRBF, EI and LCBS strategies. Because fitted surrogates are plain Julia functions, they compose directly with `ForwardDiff.jl` [@revels2016forward] and `Zygote.jl` [@innes2018zygote], so a surrogate can act as a cheap, differentiable component inside a larger program with no glue code.

# Statement of need

Surrogate modeling and surrogate-based optimization are used across aerospace design, hyperparameter tuning, and simulation-based engineering, and mature toolkits exist outside Julia [@bouhlel2019python; @saves2024smt; @eriksson2019pysot]. Julia lacked a single package spanning this range under one interface, so users comparing, say, Kriging against radial basis functions against a neural network on the same problem had to learn several APIs or hand-assemble packages such as `GaussianProcesses.jl` [@GaussianProcessesjl] and `Flux.jl` [@Fluxjl] around their own optimization loop.

`Surrogates.jl` fills that gap for the Julia [@julia] and SciML [@rackauckas2017differentialequations] ecosystems, and reuses SciML infrastructure rather than reimplementing it: where a surrogate fits hyperparameters numerically, the fit is posed as an `OptimizationProblem` and solved through `Optimization.jl` [@optimizationjl]; sampling comes from `QuasiMonteCarlo.jl`; and the common interface (`parameters`, `hyperparameters`, `update!`) implements `SurrogatesBase.jl` [@SurrogatesBasejl], the abstract interface shared with the wider SciML surrogate ecosystem. Surrogates therefore slot into existing Julia optimization and machine learning pipelines instead of forming a closed toolbox.

# Functionality

A typical workflow has three stages:

1. **Sampling.** `sample(n, lb, ub, SobolSample())` builds the design of experiments with any `QuasiMonteCarlo.jl` sampler.
2. **Fitting.** Most surrogates follow `Surrogate(x, y, lb, ub; kwargs...)`, as in `Kriging(x, y, lb, ub)` or `RadialBasis(x, y, lb, ub)`, for scalar or vector inputs alike, and accept `update!` for new observations. The PLS and gradient-enhanced variants take extra positional arguments — `KPLS(x, y, n_comp, lb, ub, theta)` and `GEKPLS(x, y, grads, lb, ub)` — and a subset (radial basis, linear, second-order polynomial, inverse distance, Lobachevsky, and the neural and `AbstractGPs` backends) supports vector-valued responses.
3. **Optimization.** `surrogate_optimize!(f, alg, lb, ub, surrogate, sampler)` drives the surrogate/expensive-function loop, evaluating `f` once per iteration. For parallel evaluation, `potential_optimal_points(alg, strategy, lb, ub, surrogate, sampler, n_parallel)` instead *asks* for a batch of `n_parallel` candidates and the caller *tells* the surrogate their responses with `update!`.

Several surrogates exploit extra problem structure to reach a given accuracy in fewer evaluations of $f$: `GEK` and `GEKPLS` fold in derivative information, often available for free from adjoint-based PDE solvers [@giles2000adjoint], to raise accuracy per sample; `KPLS` and `KPLSK` fit correlation length scales in a PLS-reduced subspace to keep Kriging tractable in high input dimension; and `VariableFidelitySurrogate` corrects a cheap low-fidelity model with a small set of expensive high-fidelity samples, the common multi-fidelity setting in engineering design [@forrester2007multifidelity].

# State of the field

\autoref{tab:comparison} places `Surrogates.jl` among comparable open-source toolkits. In Python, SMT covers the Kriging family with derivatives and mixed-variable support and provides EGO with EI, SBO and LCB infill criteria and qEI batching [@bouhlel2019python; @saves2024smt]; pySOT pairs RBF, GP, MARS and polynomial surrogates with the SRBF, DYCORS, SOP, EI and LCB strategies and asynchronous parallel evaluation [@eriksson2019pysot]; and BoTorch, with Ax, builds Monte-Carlo acquisition functions on PyTorch, including multi-objective batching [@balandat2020botorch; @daulton2020ehvi]. In R, DiceKriging and DiceOptim cover Kriging metamodeling and EGO with noisy, constrained and parallel variants [@roustant2012dicekriging; @DiceOptimCRAN], while mlrMBO makes the surrogate and infill criterion swappable over any `mlr` learner [@bischl2017mlrmbo]. Dakota embeds surrogate-based optimization in a large simulation framework alongside uncertainty quantification and sensitivity analysis [@adams2020dakota]. Within Julia, `BayesianOptimization.jl` fits GP surrogates with expected improvement, probability of improvement, upper confidence bound and Thompson sampling [@BayesianOptimizationjl], and `GaussianProcesses.jl` [@GaussianProcessesjl] and `AbstractGPs.jl` [@AbstractGPsjl] supply the GP models themselves without an optimization loop.

Table: Comparison of open-source surrogate modeling toolkits. \label{tab:comparison}

| Package | Language | Gradient-enhanced Kriging | Surrogate-based optimization | Multi-objective | AD through the surrogate |
|---|---|---|---|---|---|
| `Surrogates.jl` (this work) | Julia | GEK, GEKPLS | SRBF, DYCORS, SOP, EI, LCBS | SMB, RTEA | ForwardDiff, Zygote |
| SMT [@bouhlel2019python; @saves2024smt] | Python | yes | EGO (EI, SBO, LCB), qEI batch | no | analytic derivatives |
| pySOT [@eriksson2019pysot] | Python | no | SRBF, DYCORS, SOP, EI, LCB | no | no |
| BoTorch / Ax [@balandat2020botorch] | Python | no | MC acquisition, batch | yes | PyTorch [@paszke2019pytorch] |
| DiceOptim [@roustant2012dicekriging; @DiceOptimCRAN] | R | no | EGO, noisy and parallel variants | no | no |
| `BayesianOptimization.jl` [@BayesianOptimizationjl] | Julia | no | EI, PI, UCB, Thompson | no | no |

The contribution of `Surrogates.jl` is coverage rather than a new algorithm. Of the packages in \autoref{tab:comparison}, it is the only one that offers classical interpolation, the Kriging family with gradient enhancement and PLS dimension reduction, splines, and machine-learning surrogates behind one API, and the only one combining RBF-based global optimizers, Kriging-family acquisition functions and multi-objective surrogate loops in a single package.

We built a new package rather than contributing to an existing one because neither alternative reaches the same users. SMT and pySOT are Python libraries: calling them from Julia would cross a language boundary on every evaluation of $f$ and would forfeit differentiation through the fitted surrogate, which is the property that lets a surrogate stand in for a simulation inside a larger Julia program. Within Julia, `GaussianProcesses.jl` and `AbstractGPs.jl` are Gaussian-process libraries by design, so the non-GP families and the optimization loops have no natural home there. Where shared infrastructure did make sense we contributed it rather than duplicating: sampling lives in `QuasiMonteCarlo.jl`, solvers in `Optimization.jl`, and the surrogate interface was extracted into `SurrogatesBase.jl` [@SurrogatesBasejl] so that other packages can implement it without depending on this one.

# Software design

Three decisions shape the package. First, the common interface is defined in a separate package, `SurrogatesBase.jl`, rather than here. A surrogate is either an `AbstractDeterministicSurrogate` or an `AbstractStochasticSurrogate`, and implements `update!`, `parameters` and `hyperparameters`; only the stochastic models add `std_error_at_point`. Splitting the abstract interface from the implementations costs an extra package to maintain, but it lets the optimization loops dispatch on capability instead of on concrete type, and lets surrogates defined elsewhere participate.

Second, a fitted surrogate is a callable Julia object, not a handle onto an external fitting engine. This is what makes `ForwardDiff.jl` and `Zygote.jl` differentiate through a prediction with no special support, so a surrogate composes with gradient-based code downstream. The cost is that each family must be written in Julia rather than wrapped, so the heavier backends — `Flux.jl`, `LIBSVM`, `XGBoost`, `PolyChaos.jl`, `AbstractGPs.jl` and the Gaussian-mixture expert model are package extensions, loaded only when the user loads the backend. A default install therefore stays light while the surrogate list stays long.

Third, hyperparameter fitting is posed as an `OptimizationProblem` and handed to `Optimization.jl` [@optimizationjl] instead of carrying a bespoke optimizer: the Kriging family fits its correlation scales by maximum likelihood, searched with multi-start Nelder–Mead [@nelder1965simplex] over a box of admissible scales. Parallel evaluation is exposed as the ask/tell pair above rather than as a callback, because the loop that dispatches expensive jobs to a cluster belongs to the user; the virtual points are placed on a private copy of the surrogate, so candidates within a batch repel each other without the caller's surrogate being touched.

# Research impact statement

`Surrogates.jl` is used as a component by registered Julia packages in several fields: `ADRIA.jl`, a reef restoration decision-support platform from the Australian Institute of Marine Science; `AeroFuse.jl` (formerly `AeroMDAO.jl`), for aircraft design analysis; `Agtor.jl`, an agricultural systems model; `MaterialPointGenerator.jl`, for material point
method landslide simulations; and `VariogramAnalysis.jl`, an implementation of the VARS global sensitivity analysis method [@razavi2016vars]. 

The repository has 383 stars and 78 forks, and the package is a long-standing part of the SciML organization, where it has been the subject of several Google Summer of Code projects.

# AI usage disclosure

Generative AI was used in this project and is visible in its history. The Git log contains commits authored under AI-assisted identities (`ChrisRackauckas-Claude`, `Claude`, `Claude Code`, and `Chris Rackauckas - Beep Boop Edition`, 46 commits in total) and 53 commits carrying a `Co-Authored-By` trailer naming Claude; these cover refactoring, tests,
documentation and bug fixes, all reviewed and merged through the normal pull-request process by human maintainers. This paper was drafted and revised with Anthropic's Claude via Claude Code; the authors verified every citation against publisher records, checked each claim against the source code, and are responsible for the content.

# Acknowledgements

`Surrogates.jl` is developed as part of the SciML organization. Parts of the package were developed with support from Google Summer of Code, which funded several student contributors. We thank everyone who has contributed code, documentation, and bug reports through GitHub.

# References
