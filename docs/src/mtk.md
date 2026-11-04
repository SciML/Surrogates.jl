# Surrogates in acausal models (ModelingToolkit)

A surrogate is most useful when it stands in for something expensive *inside* a larger model. This tutorial replaces a nonlinear PDE solve with a surrogate and uses it to drive an acausal circuit built with [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/), then checks the result against the PDE itself.

The important point about the structure: the PDE here is **steady state**, so the surrogate is still an ordinary map from parameters to a quantity of interest, which is what Surrogates.jl provides. The dynamics live entirely in the circuit. That is the usual shape of this problem. You are not surrogating a dynamical system, you are surrogating a constitutive relation that a dynamical system evaluates over and over.

## The expensive model

A chip on a plate. Steady conduction with a radiative sink,

```math
-\nabla^2 T + \varepsilon (T^4 - T_\text{amb}^4) = P \, g(x,y),
```

with a Dirichlet heat sink along the bottom edge and Robin convection ``-\partial T/\partial n = h T`` on the other three. The two inputs are the heater power ``P`` and the convection coefficient ``h``, which depends on coolant flow. The quantity of interest is the hot-spot temperature at the plate centre. Because the Robin condition enters the operator, every evaluation needs a fresh assembly and factorization, and the radiative term makes it nonlinear, so each solve is a small Newton loop.

```@example mtk
using Surrogates, ModelingToolkit, OrdinaryDiffEq, ForwardDiff
using ModelingToolkit: t_nounits as t, D_nounits as D
using LinearAlgebra, SparseArrays
const Symbolics = ModelingToolkit.Symbolics

const N, EPSR, TAMB = 48, 0.05, 0.2      # nodes per side, radiation coefficient, ambient
const dx = 1 / (N + 1)
const grid = range(dx, 1 - dx, length = N)

# 2D Laplacian as a Kronecker sum of 1D second differences.
const LAP = let e = ones(N - 1)
    T1 = spdiagm(-1 => -e, 0 => fill(2.0, N), 1 => -e) ./ dx^2
    kron(spdiagm(0 => ones(N)), T1) + kron(T1, spdiagm(0 => ones(N)))
end

# The Gaussian heater, and how many convective edges each node touches.
# The bottom edge (j == 1) is the Dirichlet heat sink, so it is excluded.
const GSRC = vec([exp(-((x - 0.5)^2 + (y - 0.5)^2) / (2 * 0.12^2)) for x in grid, y in grid])
const NEDGE = vec([(i == 1) + (i == N) + (j == N) for i in 1:N, j in 1:N])

"Hot-spot temperature for heater power `P` and convection coefficient `h`."
function pde_hotspot(P, h)
    A = LAP - spdiagm(0 => NEDGE ./ (dx^2 * (1 + dx * h)))   # Robin edges enter the operator
    T = fill(0.1, N^2)
    for _ in 1:50                                            # Newton
        F = A * T .+ EPSR .* (T .^ 4 .- TAMB^4) .- P .* GSRC
        norm(F, Inf) < 1e-10 && break
        T .-= (A + spdiagm(0 => 4EPSR .* T .^ 3)) \ F
    end
    c = N ÷ 2
    T[(c - 1) * N + c]                                       # centre node
end
nothing # hide
```

One solve takes about 4.6 ms. That is cheap in isolation and ruinous inside a time-stepping loop that wants it hundreds of times.

## Fitting the surrogate

An ordinary design of experiments over the two inputs, then a `Kriging` fit. Always check the fit on points the surrogate has not seen.

```@example mtk
lb, ub = [0.5, 0.5], [2.0, 4.0]
design = sample(80, lb, ub, SobolSample())
ys = [pde_hotspot(p...) for p in design]
const SURR = Kriging(design, ys, lb, ub)

val = sample(25, lb, ub, HaltonSample())
truth = [pde_hotspot(p...) for p in val]
maximum(abs(SURR(p) - y) for (p, y) in zip(val, truth))     # 7.4e-6
```

80 training solves cost about 1.8 s, and the surrogate reproduces held-out hot-spot temperatures to 7.4e-6 on a response range of 0.015 to 0.057. A surrogate call takes 2.8 us, roughly 1600 times faster than the PDE.

## Making the surrogate symbolic

ModelingToolkit works on symbolic expressions, so a numerical function has to be registered before it can appear in an equation. You need three declarations: the function, its derivative, and the rule connecting them.

```@example mtk
Thot(P, h)   = SURR([P, h])
dThot1(P, h) = ForwardDiff.gradient(SURR, [P, h])[1]
dThot2(P, h) = ForwardDiff.gradient(SURR, [P, h])[2]

@register_symbolic Thot(P, h)
@register_symbolic dThot1(P, h)
@register_symbolic dThot2(P, h)
@register_derivative Thot(P, h) 1 dThot1(P, h)
@register_derivative Thot(P, h) 2 dThot2(P, h)
nothing # hide
```

## The circuit

A thermoelectric source converts the hot-spot temperature into a voltage, ``v = \alpha (T_\text{hot} - T_\text{amb})``, and charges an RC network. The components are acausal: each exposes pins, and `connect` applies Kirchhoff's laws.

```@example mtk
@connector function Pin(; name)
    @variables v(t) i(t) [connect = Flow]
    System(Equation[], t, [v, i], []; name)
end

function OnePort(; name)
    @named p = Pin(); @named n = Pin()
    @variables v(t) i(t)
    System([v ~ p.v - n.v, 0 ~ p.i + n.i, i ~ p.i], t, [v, i], []; systems = [p, n], name)
end

function Resistor(; name, R)
    @named op = OnePort(); @unpack v, i = op
    extend(System([v ~ i * R], t, [], @parameters R = R; name), op)
end

function Capacitor(; name, C)
    @named op = OnePort(); @unpack v, i = op
    extend(System([D(v) ~ i / C], t, [], @parameters C = C; name), op)
end

function Ground(; name)
    @named g = Pin()
    System([g.v ~ 0], t, [], []; systems = [g], name)
end

"Seebeck source whose voltage follows the surrogate's hot-spot temperature."
function SeebeckSource(; name, alpha, Tfun)
    @named op = OnePort(); @unpack v = op
    extend(System([v ~ alpha * (Tfun(t) - TAMB)], t, [], @parameters alpha = alpha; name), op)
end
nothing # hide
```

The heater power and the coolant convection both vary in time, which is what forces the surrogate to be evaluated at every solver step rather than once. Both profiles stay inside the box the surrogate was fitted on, which matters: a surrogate extrapolates badly, so the driving signals must respect the design bounds.

```@example mtk
Pdrive(tt) = 1.25 + 0.60 * sin(2pi * tt)     # inside [0.5, 2.0]
hdrive(tt) = 2.25 + 1.50 * cos(pi * tt)      # inside [0.5, 4.0]

function build_circuit(Tfun; name)
    @named src = SeebeckSource(alpha = 0.05, Tfun = Tfun)
    @named res = Resistor(R = 100.0)
    @named cap = Capacitor(C = 1.0e-3)
    @named gnd = Ground()
    eqs = [connect(src.p, res.p), connect(res.n, cap.p),
           connect(cap.n, src.n), connect(cap.n, gnd.g)]
    System(eqs, t; systems = [src, res, cap, gnd], name)
end

@named circuit = build_circuit(tt -> Thot(Pdrive(tt), hdrive(tt)))
circuit = mtkcompile(circuit)
```

`mtkcompile` reduces the connected network to a single state, the capacitor voltage. The registered derivative shows up in the symbolic Jacobian, which is what an implicit solver needs.

```@example mtk
prob = ODEProblem(circuit, [circuit.cap.v => 0.0], (0.0, 2.0); jac = true, tgrad = true)
sol = solve(prob, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
sol[circuit.cap.v][end]
```

!!! tip "Pass `tgrad = true` for Rosenbrock methods"
    
    `Rodas5P` and friends need a time gradient as well as a Jacobian. Without `tgrad = true`
    they build it by differentiating the right-hand side with respect to `t` automatically.
    That works for a surrogate, which accepts `Dual` numbers, but it fails for anything that
    does not, such as a raw PDE solver you might substitute in for comparison.

## Checking it against the PDE

Replace `Thot` with the PDE itself and solve the identical model.

```@example mtk
Tpde(P, h)   = pde_hotspot(P, h)
dTpde1(P, h) = (pde_hotspot(P + 1e-5, h) - pde_hotspot(P - 1e-5, h)) / 2e-5
dTpde2(P, h) = (pde_hotspot(P, h + 1e-5) - pde_hotspot(P, h - 1e-5)) / 2e-5
@register_symbolic Tpde(P, h)
@register_symbolic dTpde1(P, h)
@register_symbolic dTpde2(P, h)
@register_derivative Tpde(P, h) 1 dTpde1(P, h)
@register_derivative Tpde(P, h) 2 dTpde2(P, h)

@named ref = build_circuit(tt -> Tpde(Pdrive(tt), hdrive(tt)))
ref = mtkcompile(ref)
solref = solve(ODEProblem(ref, [ref.cap.v => 0.0], (0.0, 2.0); jac = true, tgrad = true),
               Rodas5P(); abstol = 1e-8, reltol = 1e-8)
solref[ref.cap.v][end]
```

| quantity | surrogate | PDE |
|---|---|---|
| circuit solve | 2.6 ms | 3.3 s |
| solver steps | 47 | 49 |
| `v_cap` at t = 2 | -0.008766 | -0.008766 |

The two trajectories agree to 3.3e-8 across the whole time span, which is 0.00% of the peak capacitor voltage, for about a 1250-fold reduction in simulation time. An instrumented run shows the surrogate being called 386 times with 47 derivative calls per solve, against 394 PDE solves plus roughly 188 more inside the reference's own derivative rules. The 1.8 s spent training is recovered on the first simulation.
