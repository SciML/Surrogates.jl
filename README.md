![SurrogatesLogo](docs/src/images/Surrogates.png)
## Surrogates.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/Surrogates.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/Surrogates.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/fl7hr18apc7lt4of?svg=true)](https://ci.appveyor.com/project/ludoro/surrogates-jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaDiffEq/Surrogates.jl/badge.svg)](https://coveralls.io/github/JuliaDiffEq/Surrogates.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://surrogates.juliadiffeq.org/stable/)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](http://surrogates.juliadiffeq.org/latest/)

A surrogate model is an approximation method that mimics the behavior of a computationally
expensive simulation. In more mathematical terms: suppose we are attempting to optimize a function
`f(p)`, but each calculation of `f` is very expensive. It may be the case we need to solve a PDE for each point or use advanced numerical linear algebra machinery which is usually costly. The idea is then to develop a surrogate model `g` which approximates `f` by training on previous data collected from evaluations of `f`.
The construction of a surrogate model can be seen as a three steps process:
- Sample selection
- Construction of the surrogate model
- Surrogate optimization

## Resources

- **Surrogate Modeling Toolbox:** <https://smt.readthedocs.io/en/latest/>
- **Surrogate Models And Their Types** <https://en.wikipedia.org/wiki/Surrogate_model>
- **Adaptive Sampling Techniques:** <https://www.emerald.com/insight/content/doi/10.1108/02644401311329352/full/html?fullSc=1>
- **Documentation:** <https://docs.julialang.org/>
- **Packages:** <https://pkg.julialang.org/>
- **Discourse forum:** <https://discourse.julialang.org/>
- **Slack:** <https://julialang.slack.com> (get an invite from <https://slackinvite.julialang.org>)

## Building Julia

First, make sure you have all the [required
dependencies](https://github.com/JuliaLang/julia/blob/master/doc/build/build.md#required-build-tools-and-external-libraries) installed.
Then, acquire the source code by cloning the git repository:

    git clone git://github.com/JuliaLang/julia.git

By default you will be building the latest unstable version of
Julia. However, most users should use the most recent stable version
of Julia. You can get this version by changing to the Julia directory
and running:

    git checkout v1.1.1

Now run `make` to build the `julia` executable.

Building Julia requires 2GiB of disk space and approximately 4GiB of virtual memory.

**Note:** The build process will fail badly if any of the build directory's parent directories have spaces or other shell meta-characters such as `$` or `:` in their names (this is due to a limitation in GNU make).

Once it is built, you can run the `julia` executable after you enter your julia directory and run

    ./julia

Your first test of Julia determines whether your build is working
properly. From the UNIX/Windows command prompt inside the `julia`
source directory, type `make testall`. You should see output that
lists a series of running tests; if they complete without error, you
should be in good shape to start using Julia.

You can read about [getting
started](https://docs.julialang.org/en/v1/manual/getting-started/)
in the manual.

In case this default build path did not work, detailed build instructions
are included in the [build documentation](https://github.com/JuliaLang/julia/blob/master/doc/build).

### Uninstalling Julia

Julia does not install anything outside the directory it was cloned
into. Julia can be completely uninstalled by deleting this
directory. Julia packages are installed in `~/.julia` by default, and
can be uninstalled by deleting `~/.julia`.
