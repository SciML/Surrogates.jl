using Surrogates, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(Surrogates)
    Aqua.test_ambiguities(Surrogates, recursive = false)
    Aqua.test_deps_compat(Surrogates)
    Aqua.test_piracies(Surrogates)
    Aqua.test_project_extras(Surrogates)
    Aqua.test_stale_deps(Surrogates)
    Aqua.test_unbound_args(Surrogates)
    Aqua.test_undefined_exports(Surrogates)
end
