
struct StationaryMarkovChain
    x::Vector{Float64}
    proba::Matrix{Float64}
end

function import_markov_chain(Nd)
    return StationaryMarkovChain(
        readdlm("data/markov_support_$(Nd).txt")[:],
        readdlm("data/markov_weights_$(Nd).txt"),
    )
end

# Sample a discrete distribution according to
# the probability passed in `probs`.
function sample(probs::Vector)
    n = length(probs)
    u = rand()
    cumsum = 0.0
    j = -1
    for k in 1:n
        cumsum += probs[k]
        if cumsum >= u
            j = k
            break
        end
    end
    return j
end

function _project(support, x0)
    nx = length(support)
    ind0, val_min = -1, Inf
    for i in 1:nx
        current_val = abs(support[i] - x0)
        if current_val < val_min
            ind0 = i
            val_min = current_val
        end
    end
    return ind0
end


function generate_price_scenarios(
    markov::StationaryMarkovChain,
    avg_price::Vector,
    nscenarios::Int,
)
    horizon = length(avg_price)
    T = horizon - 1
    Nd = length(markov.x)
    w = zeros(T, nscenarios)

    for k in 1:nscenarios
        ind = rand(1:Nd)
        for t in 1:T
            prob = markov.proba[ind, :]
            ind = sample(prob)
            ξ = markov.x[ind]
            w[t, k] = avg_price[t] * exp(ξ)
        end
    end

    return w
end


