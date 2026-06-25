using SciMLTesting, Surrogates, Test
using JET

run_qa(
    Surrogates;
    explicit_imports = true,
    ei_kwargs = (;
        all_explicit_imports_are_public = (;
            ignore = (
                :Buffer,  # Zygote (not public)
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@deprecate_binding"),  # Base (not public)
                :AbstractVecOrTuple,           # Base (not public)
                :ProductIterator,              # Base.Iterators (not public)
                :RefValue,                     # Base (not public)
                :_check_sequence,              # QuasiMonteCarlo (not public)
                :sample,                       # QuasiMonteCarlo (not public)
            ),
        ),
    ),
    # no_implicit_imports tracked in SciML/Surrogates.jl#564 (heavy `using X`
    # whole-module imports; resolving needs a focused per-file refactor).
    ei_broken = (:no_implicit_imports,),
)
