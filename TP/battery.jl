#' ---
#' title : Optimal management of an energy storage
#' author : François Pacaud
#' date : November, 18th 2024
#' ---


#' # Motivation
#' The following project is implemented in Julia, using the
#' optimization modeler JuMP.
#' - If you are new to the Julia language, we recommend the [following introduction](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_julia/).
#' - If you are new to optimization modeler, we recommend the [following tutorial](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/)

#' The project is inspired by recent works on the optimal management
#' of energy storage using Stochastic Dynamic Programming. In particular, we acknowledge
#' the [following article](https://ieeexplore.ieee.org/abstract/document/9721005).

#' # Introduction
#' Energy storage are becoming one of the major technology
#' to [operate a modern grid](https://blog.gridstatus.io/caiso-batteries-apr-2024/).
#' In this tutorial, we are interested in finding the optimal policy to
#' manage a battery when the future energy prices are unknown.
#'
#' The battery has given capacity and maximal rate of charge/discharge.
#' At every time-step, we can either sell (by discharging energy from the battery)
#' or buy electricity (by charging energy to the battery).
#'
#' For our analysis, we use real data from the European spot
#' market [EPEX](https://www.epexspot.com/en/market-data), in the year 2016.
#' The spot price is sampled at an hourly time-step during 3 days (giving a total
#' of 72 time-steps). We import the data and plot the evolution of the price
#' during the three days using the following code:

using DelimitedFiles
using Plots
price = readdlm("data/epex_price.txt")[:]

plot(price, lw=2.0, color=:black, label="EPEX price")
xlabel!("Hours")
ylabel!("Energy price (€)")

#' ---
#' **Question 1:** Comment the evolution of the price. What is the average
#' energy price? Compare with (i) the price to produce 1MWh using nuclear energy
#' (ii) the price paid by the consumer.
#'
#' ---

#' # Part I: Deterministic model
#' Before building a stochastic model for the energy price, we look at the
#' deterministic solution.
#' We start by writing a mathematical model for our energy storage. We introduce
#' the following parameters:
#' - ``C``: battery capacity (in MWh)
#' - ``P``: maximum battery charge/discharge in one hour (in MWh)
#' - ``η_p``: charge efficiency
#' - ``η_d``: discharge efficiency
#' - ``c``: tax paid when injecting energy onto the network (in €/MWh)
#' Our decision variables are:
#' - ``x_t``: Energy stored in the battery at time (SoC) ``t`` (in MWh)
#' - ``p_t``: Energy discharged from the battery between ``t`` and ``t+1`` (in MWh).
#' - ``b_t``: Energy charged to the battery between ``t`` and ``t+1`` (in MWh).
#' The price at time ``t`` is denoted by ``λ_t``.
#' The operational constraints are
#' ```math
#' 0 ≤ x_t ≤ C,  
#' 0 ≤ p_t ≤ P,  
#' 0 ≤ b_t ≤ P.  
#' ```
#' The (discrete) dynamics of the battery writes
#' ```math
#' x_{t+1} = x_t - \frac{1}{η_p} p_t + η_d b_t
#' ```
#' We aim at maximizing our profit, which writes equivalently as minimizing the
#' following cost:
#' ```math
#' ∑_{t=1}^T \Big( λ_t (b_t - p_t) + c p_t \Big)
#' ```
#' The optimization problem writes as a Linear Program (LP).
#' ```math
#' \begin{aligned}
#' \min & \; ∑_{t=1}^T \Big( λ_t (b_t - p_t) + c p_t \Big) \\
#' \text{s.t.}\quad & x_{t+1} = x_t - \frac{1}{η_p} p_t + η_d b_t  \\
#'             & 0 ≤ x_t ≤ C \\
#'             & 0 ≤ b_t ≤ P \\
#'             & 0 ≤ p_t ≤ P \\
#'             & x_0 = 0
#' \end{aligned}
#' ```

#' ---
#' **Question 2.** Name one optimization algorithm that can solve the previous LP problem.
#'
#' ---

#' The data of the problem are stored in a structure `BatteryData`

struct BatteryData
    C::Float64
    P::Float64
    c::Float64
    eta_p::Float64
    eta_d::Float64
end

#' We instantiate a new battery using:

data = BatteryData(5.0, 1.0, 1.0, 0.9, 0.9);

#' We implement the LP problem using the modeler JuMP.

using JuMP

function build_deterministic_model(data::BatteryData, price::Vector)
    T = length(price)
    model = Model()
    @variable(model, 0.0 <= x[1:T+1] <= data.C)
    @variable(model, 0.0 <= p[1:T] <= data.P)
    @variable(model, 0.0 <= b[1:T] <= data.P)
    # Boundary condition
    @constraint(model, x[1] == 0.0)
    # Dynamics
    @constraint(model, dynamics[t=1:T],  x[t+1] == x[t] - (1.0 / data.eta_p) * p[t] + data.eta_d * b[t])
    # Objective
    @objective(
        model,
        Min,
        sum(price[t] * (b[t] - p[t]) + data.c * p[t] for t in 1:T)
    )
    return model
end

#' For a solver, we use the open-source [HiGHS](https://ergo-code.github.io/HiGHS/stable/).
#' Solving the LP using JuMP just amount to

using HiGHS

det_model = build_deterministic_model(data, price)
JuMP.set_optimizer(det_model, HiGHS.Optimizer)
JuMP.optimize!(det_model)

#' ---
#' **Question 3.** In how many iterations does HiGHS converge?
#'
#' ---

#' The final objective is:
JuMP.objective_value(det_model)

#' We can plot the solution using the following lines:
x_sol = JuMP.value.(det_model[:x])
plot(x_sol, lw=2.0, color=:black, label="Battery level")

#' ---
#' **Question 4.** Does this optimal strategy make sense for you? Explain why.
#'
#' ---

#' # Part II: Stochastic model
#' Now, we replace the previous deterministic model by a stochastic model taking
#' explicitly into account the uncertainties in the future energy prices.

#' ### Markovian model
#' We use a classical stochastic model for the energy price. The previous
#' price vector `price` now encodes the average energy price ``\overline{λ}_t`` (a.k.a. expected value).
#' The log-deviation from the expected value is modeled with a stochastic process
#' ``\{ ξ_t(ω) \}_{t}``, here modeled as a stationary Markov chain. The model writes,
#' for all ``t= 1, ⋯, T``,
#' ```math
#' \log(λ_t(ω)) = \log(\overline{λ}_t) + ξ_t(ω)
#' ```
#' or, equivalently,
#' ```math
#' λ_t(ω) = \overline{λ}_t ×  \exp{ξ_t(ω) }
#' ```
#' At each time step ``ξ_t(ω)`` can take ``N`` distinct values ``ξ_1, ⋯, ξ_N``.
#' The transition between ``ξ_{t}`` and ``ξ_{t+1}`` are encoded by the following
#' conditional probabilities:
#' ```math
#' \mathbb{P}[ ξ_{t+1} = ξ_j \; | \; ξ_t = ξ_i ] = p_{ij}
#' ```
#' The values and the transition probabilities are stored as text files in the
#' directory `data`. We provide a set of util functions to manipulate stationary
#' Markov chain in the script `markov.jl`:

include("markov.jl");

#' To import a Markov chain with a discretization size ``N = 4``:

markov = import_markov_chain(4);

#' The transition probabilities are:

markov.proba

#' and the values ``ξ_1, ⋯, ξ_4`` are:

markov.x

#' You can increase the discretization size up to 32 (``N`` can take
#' any values in ``\{4, 8, 16, 32 \}``).

#' The Markov chain `markov` defines our probabilistic model. The future energy prices
#' are now uncertains, with probability distribution given by the Markov chain. You can
#' sample a given number of scenarios from the Markov chain using the
#' function ``generate_price_scenarios``:

n_scenarios = 10
scenarios = generate_price_scenarios(markov, price, n_scenarios)

plot(scenarios, lw=0.5, color=:black, legend=false)
plot!(price, lw=5.0, color=:darkblue)
xlabel!("Hours")
ylabel!("Energy price (€)")


#' ### Stochastic Dynamic programming

#' The problem becomes stochastic. We aim at minimizing the expected value of the cost:
#' ```math
#' \begin{aligned}
#' \min & \; \mathbb{E} \Big[ ∑_{t=1}^T \Big( λ_t(ω) (b_t(ω) - p_t(ω)) + c p_t(ω) \Big)\Big] \\
#' \text{s.t.}\quad & x_{t+1}(ω) = x_t(ω) - \frac{1}{η_p} p_t(ω) + η_d b_t(ω)  \\
#'             & 0 ≤ x_t(ω) ≤ C \\
#'             & 0 ≤ b_t(ω) ≤ P \\
#'             & 0 ≤ p_t(ω) ≤ P \\
#'             & x_0 = 0
#' \end{aligned}
#' ```
#' We want to solve the stochastic problem using the Stochastic Dynamic Programming algorithm.
#' For our Markovian model, the Dynamic Programming equations adapt as follows.
#' The ``Q`` value function satisfies the recursive equations: Starting from ``V_T(x) = 0``,
#' we solve
#' ```math
#' Q_{t, j}(x_t) = \left\{
#' \begin{aligned}
#' \min_{x^+, p,b } \; & \lambda_{t, j} \times (b - p) + c \times p + V_{t+1, j}(x^+) \\
#' \text{s.t.}   & x^+ = x - \frac{1}{η_p} p + η_d b \\
#'               & 0 ≤ p ≤ P \\
#'               & 0 ≤ b ≤ P \\
#'               & 0 ≤ x^+ ≤ C
#' \end{aligned}
#' \right.
#' ```
#' and update the value function ``V_{t, i}`` as
#' ```math
#' V_{t, i} = ∑_{j=1}^N p_{ij} Q_{t, j}
#' ```

#' ---
#' **Question 5.** Write a pseudo-code that adapt the Stochastic Dynamic Programming algorithm
#' in a Markovian setting.
#'
#' ---

#' We discretize each value function on a grid ``\{x^1, ⋯, x^d \}``, and define
#' ```math
#' V_{t, i}^k := V_{t, i}(x^k)   ∀ k=1,⋯,d
#' ```
#' For any ``x ∈ [0, C]``, we can evaluate ``V_{t,i}(x)`` using a linear interpolation.
#' The interpolation is solution of the following LP:
#' ```math
#' V_{t,i}(x) = \min_{α ∈ \mathbb{R}^d} \;  ∑_{k=1}^d α_k V_{t,i}^k   \text{s.t.}   α_k ≥ 0 \; ,
#'   ∑_{k=1}^d α_k = 1 \; , \quad
#'   ∑_{k=1}^d α_k x^k = x
#' ```

#' ---
#' **Question 6.** Show that the interpolated value function ``Q_{t,j}^\ell := Q_{t,j}(x^\ell)``
#' is solution of
#' ```math
#' Q_{t, j}^\ell = \left\{
#' \begin{aligned}
#' \min_{x^+, p,b,\alpha } \; & \lambda_{t, j} \times (b - p) + c \times p + ∑_{k=1}^N α_k V_{t+1, j}^k \\
#' \text{s.t.}   & x^+ = x - \frac{1}{η_p} p + η_d b \\
#'               & 0 ≤ p ≤ P \\
#'               & 0 ≤ b ≤ P \\
#'               & 0 ≤ x^+ ≤ C \\
#'               & α_k ≥ 0 \;,\quad   ∑_{k=1}^d α_k = 1 \; ,   ∑_{k=1}^d α_k x^k = x^+
#' \end{aligned}
#' \right.
#' ```
#' How many variables does the LP have?
#'
#' ---

#' We define a function that takes as input the discretize value function ``V_{t,i}`` and write
#' the previous LP using JuMP.

function build_subproblem_dp(data, price, xp, Vp, optimizer, n_grid)
    @assert length(xp) == length(Vp) == n_grid

    model = Model(optimizer)
    @variable(model, x) # initial state, here considered as a parameter
    @variable(model, 0.0 <= xf <= data.C)
    @variable(model, 0.0 <= p <= data.P)
    @variable(model, 0.0 <= b <= data.P)
    @variable(model, 0.0 <= alpha[1:n_grid])
    @variable(model, θ)
    # Dynamics
    @constraint(model, xf == x - (1.0 / data.eta_p) * p + data.eta_d * b)
    # Simplicial approximation
    @constraint(model, sum(alpha) == 1.0)
    @constraint(model, xf == sum(alpha[i] * xp[i] for i in 1:n_grid))
    @constraint(model, θ == sum(alpha[i] * Vp[i] for i in 1:n_grid))
    # Objective
    @objective(model, Min, price * (b - p) + data.c * p + θ)

    JuMP.set_silent(model)
    return model
end

#' ---
#' **Question 7.** Use the function `build_subproblem_dp` to implement the
#' Stochastic Dynamic Programming algorithm.
#'
#' ---

function solve_dp(
    data::BatteryData,
    markov::StationaryMarkovChain,
    avg_price::Vector;
    optimizer=HiGHS.Optimizer,
    n_grid=11,
)
    T = length(avg_price)
    horizon = T + 1

    Nd = length(markov.x)

    xmin = 0.0
    xmax = data.C
    xgrids = collect(range(xmin, xmax, n_grid))

    V = zeros(horizon, Nd, n_grid)
    Q = zeros(T, Nd, n_grid)

    for t in reverse(1:T)
        # TODO
    end

    return V
end

#' ---
#' **Question 8.** Set ``N=4`` and ``d = 101``. Solve the Dynamic Programming equations using the function `solve_dp`.
#' Compute the numerical optimal solution returned by the algorithm. Plot the value functions
#' at time ``t ∈ \{1, 25, 49, 72 \}`` at the first lattice.
#'
#' ---


#' ---
#' **Question 9.** Set ``N = 4``. How does the objective returned by SDP
#' evolve as we increase the discretization size ``d``? Hint: take ``d ∈ \{11, 51, 101, 501, 1001 \}``.
#' What is the impact of the discretization on the solution time?
#'
#' ---


#' ---
#' **Question 10.** Set ``d = 101``. How does the objective returned by SDP
#' evolve as we increase the number of lattices in the Markov chain? Hint: take ``N ∈ \{4, 8, 16, 32\}``.
#'
#' ---


#' ---
#' **Question 11.** Let ``\overline{λ} = (\overline{λ}_1, ⋯, \overline{λ}_T)`` be the average price.
#' We note by ``v(λ)`` the solution of the deterministic problem for a given price vector ``λ``.
#' Show that ``v(\overline{λ})`` is a upper bound for the optimal value.
#'
#' ---


#' ---
#' **Question 12.** Let ``λ^i ∈ \mathbb{R}^T`` be a random realization of the price process ``\{λ_t(ω)\}_t``.
#' Prove that for a given ``k``, the value ``\frac{1}{k} ∑_{i=1}^k v(λ^i)`` is a statistical lower-bound
#' for the optimal value. Give a confidence interval. Compute numerically the upper-bound for ``N=8``.
#'
#' ---

#' ### Simulating optimal policies
#' We are now interested into implementing a control policy for the battery. At each time
#' ``t``, we should be able to determine how much energy we should charge/discharge to/from the battery.

#' We start by implementing a naive policy. It charges the battery if the price is below average, and discharge it if the price is above average. The policy writes:

struct NaivePolicy
    data::BatteryData
    price::Vector
end

function (pol::NaivePolicy)(t::Int, x::Float64, p::Float64)
    C, P = pol.data.C, pol.data.P
    eta_p, eta_d = pol.data.eta_p, pol.data.eta_d
    if p >= sum(pol.price) / length(pol.price)
        return (0.0, min(P, x * eta_p))
    else
        return (min(P, (C - x) / eta_d), 0.0)
    end
end

#' We provide the following function to simulate a given policy along a given set of scenarios:

function simulate_policy(
    pol,
    scenarios::Matrix,
    data::BatteryData,
)
    horizon, nscenarios = size(scenarios)
    T = horizon - 1

    x = zeros(horizon, nscenarios)
    cost = zeros(nscenarios)

    for k in 1:nscenarios
        for t in 1:T
            price = scenarios[t, k]
            (b, p) = pol(t, x[t, k], price)
            x[t+1, k] = x[t, k] - (1.0 / data.eta_p) * p + data.eta_d * b
            cost[k] += price * (b - p) + data.c * p
        end
    end

    return (cost, x)
end

#' ---
#' **Question 13.** Do you think `NaivePolicy` is a good policy?
#' Generate 1,000 scenarios using `generate_price_scenarios` and simulate the behavior
#' of `NaivePolicy` using `simulate_policy`. Plot the histogram of the cost and the evolution of
#' 10 trajectory for the state-of-charge of the battery. Is the naive policy profitable?
#'
#' ---


#' We now write a policy that uses the value functions computed by the SDP algorithm
#' to determine whether to charge/discharge the battery. The code is given below.

struct DPPolicy
    data::BatteryData
    price::Vector{Float64}
    markov::StationaryMarkovChain
    models::Matrix{JuMP.Model} # store JuMP model in a cache
end

function DPPolicy(data, price, markov, V; optimizer=HiGHS.Optimizer)
    T = length(price) - 1
    Nd = size(V, 2)
    n_grid = size(V, 3)
    xmin, xmax = 0.0, data.C
    xgrids = collect(range(xmin, xmax, n_grid))
    models = Matrix{JuMP.Model}(undef, T, Nd)
    for t in 1:T, i in 1:Nd
        p = price[t] * exp(markov.x[i])
        models[t, i] = build_subproblem_dp(data, p, xgrids, V[t+1, i, :], optimizer, n_grid)
    end
    return DPPolicy(data, price, markov, models)
end

function (pol::DPPolicy)(t::Int, x::Float64, p::Float64)
    # Find nearest position in Markov chain
    Δp = log(p / pol.price[t])
    ind = _project(pol.markov.x, Δp)
    dp_model = pol.models[t, ind]
    JuMP.fix.(dp_model[:x], x)
    JuMP.optimize!(dp_model)
    return (JuMP.value(dp_model[:b]), JuMP.value(dp_model[:p]))
end


#' ---
#' **Question 14.** Explain what `DPPolicy` is doing.
#' Simulate the behavior of `DPPolicy on 1,000 scenarios.
#' Plot the histogram of the cost and the evolution of 10 trajectory for the state-of-charge of the battery.
#' Compare the value obtained in simulation with
#' (1) the numerical optimal value obtained by SDP.
#' (2) the statistical lower-bound.
#' Comment the results.
#'
#' ---


#' # Part III: Expliciting the solution
#' We have seen during the lecture that for linear problems, the optimal policy
#' is polyhedral. Here, it turns out that the model is simple enough we can obtain
#' a complete characterization of the optimal solution. This brings two major simplifications:
#' 1. We can remove the LP solver in the SDP algorithm to obtain a significant speed-up.
#' 2. We can write up explicitly the optimal policy.

#' We start by having a closer look at the optimal solution. We note by ``v_{t+1, i}`` an
#' element of the subdifferential of the convex function ``V_{t+1, i}``:
#' ```math
#' v_{t+1, i}(x) ∈ ∂ V_{t+1, i}(x)
#' ```

#' ---
#' **Question 15.** Write the KKT conditions of the Bellman operator and show they depend on
#' ``v_{t+1, i}(x^+)``.
#'
#' ---

#' ---
#' **Question 16.** Suppose that for all ``t``, the price is non-negative ``λ_t ≥ 0``. Is it
#' a reasonable assumption? Show that in that case we cannot charge and discharge the battery simultaneously:
#' ``p_t × b_t = 0``.
#'
#' ---

#' ---
#' **Question 17.** Deduce from Question 16 that only the three following situations can occur:
#' - 1/ ``b > 0, p =0``.
#' - 2/ ``b = 0, p =0``.
#' - 3/ ``b = 0, p > 0``.
#' Use the KKT conditions to characterize each of the previous 3 situations using the problem's data.
#'
#' ---

#' ---
#' **Question 18.** Show that the sensitivity ``\{ v_{t, i} \}_t`` satisfies a set of
#' recursive equations, analogous of the Dynamic Programming equations.
#' Exploit this property to propose an alternative algorithm to SDP that does not rely on
#' a linear solver.
#'
#' ---

#' ---
#' **Question 19.** Implement the new algorithm and show it return the correct solution.
#' Compare its performance with the SDP algorithm implemented previously in the function `solve_dp`.
#'
#' ---

#' ---
#' **Question 20.** Write the optimal policy. How can we improve the previous `NaivePolicy`
#' to improve its performance?
#'
#' ---
