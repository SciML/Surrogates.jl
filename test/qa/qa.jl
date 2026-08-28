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
    aqua_kwargs = (;
        # QA loads Flux/XGBoost/JET/etc.; Aqua's subprocess precompile flakes on CI
        # ("done.log was not created, but precompilation exited"), not a real task leak.
        persistent_tasks = false,
    ),
    # These dependency APIs are public, but current releases do not yet attach Julia public metadata.
    ei_kwargs = (;
        all_explicit_imports_are_public = (; ignore = (:Buffer,)),
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@deprecate_binding"),
                :AbstractVecOrTuple,
                :ProductIterator,
                :RefValue,
                :_check_sequence,
            ),
        ),
    ),
    reexports_allow = (
        :GoldenSample, :GridSample, :HaltonSample, :KroneckerSample,
        :LatinHypercubeSample, :RandomSample, :SamplingAlgorithm,
        :SobolSample, :update!,
    ),
)
