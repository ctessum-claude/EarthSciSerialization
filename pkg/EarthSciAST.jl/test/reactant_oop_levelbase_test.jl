# ONE READ VERSION PER MATERIALIZED-OBSERVED LEVEL (ess-oop-levelbase), traced.
#
# The out-of-place emitter fills the materialized array observeds into a flat
# extended vector `ue`, level by level. Under a trace each fill is a
# `stablehlo.dynamic_update_slice` of the WHOLE tensor, and until this feature
# every fill's reads were taken from the version the PREVIOUS fill produced —
# so every intermediate version was live, and
#   * XLA could not update in place (the operand is still read), so the forward
#     pays a whole-buffer copy per fill, and
#   * REVERSE mode could not collapse the chain: the adjoint of each DUS is a
#     full-buffer zeroing DUS plus a copy per live version. On ReSEACT's
#     transport RHS at CONUS that was 91 whole-buffer zeroing writes — 22.4 M
#     elements, ~27 % of the reverse's element traffic — to zero a few thousand
#     slots.
# A level's entries read the state and STRICTLY LOWER levels and never each
# other (that is what `_materialized_obs_levels` builds the levels for), so
# giving the whole level ONE read version is value-exact and leaves each
# intermediate version with a single use.
#
# OPT-IN like every Reactant test here: included by runtests.jl only under
# `ESM_TEST_REACTANT=1`, or run standalone in any env with Reactant:
#
#     ESM_TEST_REACTANT=1 julia --project=@reactant \
#         pkg/EarthSciAST.jl/test/reactant_oop_levelbase_test.jl
#
# WHAT IS ASSERTED
#   1. VALUES, and this is the point of the test: flag ON ≡ flag OFF, BIT for
#      bit, for the primal AND for the reverse-mode gradient. The change moves
#      where an operand comes from, never an arithmetic op or its order, so
#      this is an enforceable `==` rather than a tolerance.
#   2. STRUCTURE, on the RAW (`optimize=false`) module, i.e. what this emitter
#      EMITS: how many DISTINCT versions of the extended tensor the fills read
#      from. One per level is the property; threading gives one per fill, and
#      every extra one is an intermediate version left LIVE — which is what
#      stops XLA updating in place and stops the reverse collapsing the chain.
#      Asserted as a count of distinct read versions rather than a hard-coded
#      op tally, so it does not go stale as the emitter changes elsewhere.
using Test
using EarthSciAST
using Reactant

const ESMb = EarthSciAST
const RXb = Reactant

_b_Dt(v) = Dict{String,Any}("op" => "D", "args" => Any[v], "wrt" => "t")
_b_ix(v, i...) = Dict{String,Any}("op" => "index", "args" => Any[v, i...])
_b_o(o, a...) = Dict{String,Any}("op" => o, "args" => Any[a...])
_b_ao(e) = Dict{String,Any}("op" => "arrayop", "output_idx" => Any["i"],
    "ranges" => Dict{String,Any}("i" => Dict{String,Any}("from" => "n")),
    "args" => Any[], "expr" => e)
_b_state(; kw...) = Dict{String,Any}("type" => "unknown",
                                     (String(k) => v for (k, v) in kw)...)
_b_param(v) = Dict{String,Any}("type" => "parameter", "default" => v)
function _b_doc(name, vars, eqs, N)
    Dict{String,Any}(
        "esm" => "0.5.0", "metadata" => Dict{String,Any}("name" => name),
        "index_sets" => Dict{String,Any}(
            "n" => Dict{String,Any}("kind" => "interval", "size" => N)),
        "models" => Dict{String,Any}("M" => Dict{String,Any}(
            "variables" => vars, "equations" => eqs)))
end

# Several observeds on ONE level (g1…gK, each a function of the state alone) plus
# one on a second level (h, which reads them) — so the fill chain is long enough
# for intermediate versions to exist, and the level boundary is exercised too.
# `h`'s stencil read splits it into interior + boundary kernels, which is how a
# real model gets several producers per observed.
function _b_multi(N, K)
    vars = Dict{String,Any}("u" => _b_state(shape = Any["n"]),
                            "h" => _b_state(shape = Any["n"]),
                            "k" => _b_param(0.25))
    eqs = Any[]
    acc = nothing
    for m in 1:K
        nm = "g$m"
        vars[nm] = _b_state(shape = Any["n"])
        push!(eqs, Dict{String,Any}("lhs" => nm,
            "rhs" => _b_ao(_b_o("+", _b_o("*", Float64(m), _b_ix("u", "i")), 1.0))))
        t = _b_ix(nm, "i")
        acc = acc === nothing ? t : _b_o("+", acc, t)
    end
    push!(eqs, Dict{String,Any}("lhs" => "h",
        "rhs" => _b_ao(_b_o("+", acc, _b_ix("g1", _b_o("+", "i", 1.0))))))
    push!(eqs, Dict{String,Any}("lhs" => _b_ao(_b_Dt(_b_ix("u", "i"))),
        "rhs" => _b_ao(_b_o("-", _b_o("*", "k", _b_ix("h", "i")), _b_ix("u", "i")))))
    return _b_doc("LEVELBASERX", vars, eqs, N)
end

_b_bits(v::AbstractVector{Float64}) = reinterpret(UInt64, v)
_b_dev(p::NamedTuple) = NamedTuple{keys(p)}(map(RXb.ConcreteRNumber, values(p)))

# The DISTINCT values of the extended tensor that a raw MLIR module reads from:
# every `stablehlo.slice` / `stablehlo.gather` whose source operand has the
# extended length, keyed by operand name. One entry per dependency level is the
# property this feature establishes; one per FILL is what threading produced.
function _b_read_versions(mod, n_total::Int)
    src = Set{String}()
    ndus = 0
    for l in split(string(mod), '\n')
        occursin("stablehlo.dynamic_update_slice", l) && (ndus += 1)
        occursin("tensor<$(n_total)xf64>", l) || continue
        m = match(r"stablehlo\.slice\s+(%[\w#]+)\s*\[", l)
        m === nothing && (m = match(r"\"stablehlo\.gather\"\((%[\w#]+),", l))
        m === nothing || push!(src, m.captures[1])
    end
    return src, ndus
end

@testset "one read version per fill level (ESS_OOP_LEVELBASE, traced)" begin
    doc = _b_multi(12, 4)
    # The flag is read at TRACE time, so ONE build serves both arms — which is
    # also the property that lets a caller price it without rebuilding.
    f, u0, p, _, _ = ESMb.build_evaluator(doc; form = :oop)
    u = Float64[0.4 * sin(0.9k) + 1.1 for k in 1:length(u0)]
    ur = RXb.ConcreteRArray(copy(u))
    pr = _b_dev(p)
    tr = RXb.ConcreteRNumber(0.0)
    # DISTINCT callees per arm, deliberately: `@jit` caches on (callee, argument
    # types) and the flag is not part of that key, so tracing the SAME closure
    # twice would hand the second arm the first arm's program and the value
    # comparison below would compare a program with itself.
    g_on(a, b, c) = f(a, b, c)
    g_off(a, b, c) = f(a, b, c)
    dot_on(a, b, c) = sum(f(a, b, c))
    dot_off(a, b, c) = sum(f(a, b, c))
    _grad(h, a, b, c) = Reactant.Enzyme.gradient(Reactant.Enzyme.Reverse, h, a,
                                                 Reactant.Enzyme.Const(b),
                                                 Reactant.Enzyme.Const(c))[1]
    grad_on(a, b, c) = _grad(dot_on, a, b, c)
    grad_off(a, b, c) = _grad(dot_off, a, b, c)

    ref = ESMb._OOP_LEVELBASE[]
    try
        ESMb._OOP_LEVELBASE[] = true
        raw_on = RXb.@code_hlo optimize = false g_on(ur, pr, tr)
        von = Array(RXb.@jit g_on(ur, pr, tr))
        gon = Array(RXb.@jit grad_on(ur, pr, tr))

        ESMb._OOP_LEVELBASE[] = false
        raw_off = RXb.@code_hlo optimize = false g_off(ur, pr, tr)
        voff = Array(RXb.@jit g_off(ur, pr, tr))
        goff = Array(RXb.@jit grad_off(ur, pr, tr))

        # (1) values: identical, to the bit, forward and reverse.
        @test _b_bits(von) == _b_bits(voff)
        @test _b_bits(gon) == _b_bits(goff)
        # ...and identical to the host walk, which this flag cannot reach at all
        # (host writes mutate, so `_oop_read_version` is the identity there).
        @test isapprox(von, f(u, p, 0.0); rtol = 1e-14, atol = 1e-15)

        # (2) structure: the same fills, reading from FEWER distinct versions of
        # the extended tensor — one per level instead of one per fill.
        nt = f.rhs.n_total
        son, dus_on = _b_read_versions(raw_on, nt)
        soff, dus_off = _b_read_versions(raw_off, nt)
        @test dus_on == dus_off > 0        # the same writes, threaded differently
        @test !isempty(son)
        @test length(son) < length(soff)
    finally
        ESMb._OOP_LEVELBASE[] = ref
    end
end
