using SciMLTesting, Surrogates, Test
using JET

# ExplicitImports can only see an extension module once its trigger package is
# loaded, so load every weakdep here to bring the extensions under QA.
using AbstractGPs, Flux, GaussianMixtures, LIBSVM, Optimisers, PolyChaos, XGBoost

const SURROGATE_EXTENSIONS = (
    :SurrogatesAbstractGPsExt,
    :SurrogatesFluxExt,
    :SurrogatesMOEExt,
    :SurrogatesPolyChaosExt,
    :SurrogatesSVMExt,
    :SurrogatesXGBoostExt,
)

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    for ext in SURROGATE_EXTENSIONS
        @test Base.get_extension(Surrogates, ext) !== nothing
    end
end

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
    api_docs_kwargs = (;
        rendered = true,
        # Sampling names are re-exported from QuasiMonteCarlo; `update!` is
        # re-exported from SurrogatesBase. Their source docs live upstream.
        rendered_ignore = (
            :GoldenSample, :GridSample, :HaltonSample, :KroneckerSample,
            :LatinHypercubeSample, :RandomSample, :SamplingAlgorithm,
            :SectionSample, :SobolSample, :sample, :update!,
        ),
    ),
    # no_implicit_imports tracked in SciML/Surrogates.jl#564 (heavy `using X`
    # whole-module imports; resolving needs a focused per-file refactor).
    ei_broken = (:no_implicit_imports,),
)
