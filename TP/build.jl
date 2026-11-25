#=
    Build the student and the solution versions of a tutorial.

    The source `.jmd` contains everything. Two kinds of blocks are delimited
    by HTML comments (each marker on its own line):

        <!-- solution -->            <!-- student -->
        **Solution.**                ```julia
        ...                          function solve_dp(...)  # skeleton
        <!-- end solution -->            # TODO
                                     end
                                     ```
                                     <!-- end student -->

    - Solution version: solution blocks are kept, student blocks are removed.
    - Student version: student blocks are kept (their Julia chunks are displayed
      but not evaluated). In solution blocks, the text is removed and the Julia
      chunks are evaluated silently (code, output and figures hidden), so that
      the code given afterwards keeps working.

    Usage (from this directory):

        julia --project=. build.jl battery.jmd wind_bidding.jmd

    produces `battery_student.html`, `battery_solution.html`, etc.
    Options: `--student` or `--solution` to build only one version,
             `--no-eval` to drop the solution chunks entirely in the student version
             (faster, but later chunks fail if they depend on a solution).
=#

using Weave

const MARKERS = Dict(
    "<!-- solution -->" => (:solution, :start),
    "<!-- end solution -->" => (:solution, :stop),
    "<!-- student -->" => (:student, :start),
    "<!-- end student -->" => (:student, :stop),
)
const HIDDEN = ["echo=false", "results=\"hidden\"", "fig=false"]
const NOEVAL = ["eval=false"]

"""
    set_chunk_options(header, options)

Add `options` to the header of a Julia chunk (e.g. "```julia; fig_width=8"),
overriding any existing option with the same name.
"""
function set_chunk_options(header::AbstractString, options::Vector{String})
    keyof(o) = strip(split(o, '=')[1])
    current = strip.(split(chopprefix(rstrip(header), "```julia"), ';'; keepempty=false))
    current = filter(o -> !(keyof(o) in keyof.(options)), current)
    return join(["```julia"; current; options], "; ")
end

"""
    filter_source(lines; solutions, eval_solutions=true)

Return the lines of the `.jmd` source for the solution version (`solutions=true`)
or the student version (`solutions=false`).
"""
function filter_source(lines::Vector{String}; solutions::Bool, eval_solutions::Bool=true)
    out = String[]
    block = :none      # :none, :solution or :student
    fence = :none      # :none, :julia or :other (inside a ``` block)
    for (k, line) in enumerate(lines)
        s = strip(line)
        if haskey(MARKERS, s)
            kind, action = MARKERS[s]
            fence == :none || error("line $k: marker `$s` inside a code block")
            if action == :start
                block == :none || error("line $k: `$s` inside a $block block")
                block = kind
            else
                block == kind || error("line $k: `$s` closes no $kind block")
                block = :none
            end
            continue
        end
        # Track code fences
        is_fence = startswith(s, "```")
        opening = is_fence && fence == :none
        if opening
            fence = startswith(s, "```julia") ? :julia : :other
        end
        chunk = fence   # kind of fence the current line belongs to
        if is_fence && !opening
            fence = :none
        end

        if block == :none
            push!(out, line)
        elseif block == :solution && solutions
            push!(out, line)
        elseif block == :student && !solutions
            # Show the skeleton, but do not evaluate it
            push!(out, opening && chunk == :julia ? set_chunk_options(s, NOEVAL) : line)
        elseif block == :solution && chunk == :julia && eval_solutions
            # Student version: run the solution code silently
            push!(out, opening ? set_chunk_options(s, HIDDEN) : line)
        end
        # Otherwise the line is dropped
    end
    block == :none || error("unclosed $block block")
    fence == :none || error("unclosed code block")
    return out
end

"""
    build(src; solutions, eval_solutions=true, kwargs...)

Weave `src` into `<name>_solution.html` or `<name>_student.html`.
Extra keyword arguments are passed to `Weave.weave` (e.g. `doctype`).
"""
function build(src::String; solutions::Bool, eval_solutions::Bool=true, kwargs...)
    name = splitext(basename(src))[1]
    suffix = solutions ? "solution" : "student"
    # The temporary file lives next to the source, so that relative paths
    # (`include("markov.jl")`, `data/...`) still work.
    tmp = joinpath(dirname(abspath(src)), "$(name)_$(suffix).jmd")
    lines = readlines(src)
    write(tmp, join(filter_source(lines; solutions, eval_solutions), "\n") * "\n")
    try
        weave(tmp; kwargs...)
    finally
        rm(tmp; force=true)
    end
end

function main(args)
    files = filter(a -> !startswith(a, "--"), args)
    if isempty(files)
        error("usage: julia --project=. build.jl [--student|--solution] [--no-eval] file.jmd ...")
    end
    versions = if "--student" in args
        [false]
    elseif "--solution" in args
        [true]
    else
        [false, true]
    end
    eval_solutions = !("--no-eval" in args)
    ENV["GKSwstype"] = "nul"   # do not open plot windows
    for src in files, solutions in versions
        build(src; solutions, eval_solutions)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
