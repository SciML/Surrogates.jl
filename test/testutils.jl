# Shared test helpers.
#
# Each surrogate has its own `update!`, so the aliasing invariant below has to be
# checked once per surrogate — what is shared is the assertion, not the code
# path. This helper carries the assertions; the surrogate name is passed in so a
# failure still says which one broke.

using Test

"""
    check_no_caller_aliasing(build_and_update, label, x, y; n_added = 1)

Assert that building a surrogate on `(x, y)` and updating it leaves the caller's
own containers untouched.

`push!`/`append!` onto a surrogate's fields grow the very vectors the caller
passed in, so a surrogate could mutate its own inputs — and a second surrogate
built from the same arrays would then see a design it never asked for. The
containers are snapshotted before `build_and_update` runs, because the aliasing
can happen at construction as easily as in `update!`.

`build_and_update(x, y)` must build the surrogate, apply the update, and return
the surrogate. It comes first so the helper reads as a `do` block. It is returned again here so a caller can add assertions specific
to that model.

`n_added` is the number of samples the update adds, for the surrogates whose
tests add more than one.
"""
function check_no_caller_aliasing(build_and_update, label, x, y; n_added = 1)
    x_before = deepcopy(x)
    y_before = deepcopy(y)
    n_x = length(x)
    n_y = length(y)

    surr = build_and_update(x, y)

    @testset "$(label): update! leaves the caller's containers alone" begin
        # Length first: a `push!` through an alias shows up here even when the
        # contents comparison would be confusing to read.
        @test length(x) == n_x
        @test length(y) == n_y
        @test x == x_before
        @test y == y_before
        # The surrogate's own design must have grown, or the test would pass for
        # an `update!` that silently did nothing.
        @test length(surr.x) == n_x + n_added
    end
    return surr
end

"""
    check_update_representations(build, label; batch = true)

Assert that `update!` accepts a `d`-dimensional point written either way.

A point may be a tuple or a coordinate vector, and every call overload takes
both. `update!` has to as well, whichever representation the design happens to
be stored in — otherwise appending to a tuple-stored design raises a bare
`MethodError` from `vcat`'s conversion, which is what `Surrogates._match_stored`
exists to prevent.

`build(x, y)` builds the surrogate on the design it is given, so the helper can
build the same model over a tuple-stored and a vector-stored design. Pass
`batch = false` for a surrogate that takes one point at a time.
"""
function check_update_representations(build, label; batch = true)
    obj(p) = p[1]^2 + p[2]^2
    lb, ub = [0.0, 0.0], [5.0, 5.0]
    x_tuples = sample(40, lb, ub, SobolSample())
    x_vectors = [collect(p) for p in x_tuples]
    y = obj.(x_tuples)

    grew(store, new_x, new_y) = begin
        surr = build(store, y)
        n = length(surr.x)
        update!(surr, new_x, new_y)
        length(surr.x) - n
    end

    @testset "$(label): update! takes a point either way" begin
        @test grew(x_tuples, (1.0, 2.0), obj((1.0, 2.0))) == 1
        @test grew(x_tuples, [1.0, 2.0], obj((1.0, 2.0))) == 1
        # ... and against a design stored as coordinate vectors, not tuples.
        @test grew(x_vectors, (1.0, 2.0), obj((1.0, 2.0))) == 1
        @test grew(x_vectors, [1.0, 2.0], obj((1.0, 2.0))) == 1
        if batch
            pts_t = [(1.0, 2.0), (2.0, 3.0)]
            pts_v = [[1.0, 2.0], [2.0, 3.0]]
            @test grew(x_tuples, pts_t, obj.(pts_t)) == 2
            @test grew(x_tuples, pts_v, obj.(pts_t)) == 2
        end
    end
    return nothing
end
