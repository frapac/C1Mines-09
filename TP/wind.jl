#=
    Synthetic wind model for the day-ahead bidding tutorial.

    The wind *speed* forecast follows a smooth daily profile. The realized speed
    is the forecast plus an AR(1) forecast error, which is zero at the gate closure
    and grows with the lead time. The wind *power* is obtained by passing the speed
    through the power curve of the farm.
=#

using Random

struct WindFarm
    Wmax::Float64       # installed capacity [MW]
    v_in::Float64       # cut-in speed [m/s]
    v_rated::Float64    # rated speed [m/s]
    v_out::Float64      # cut-out speed [m/s]
end

WindFarm(Wmax) = WindFarm(Wmax, 3.0, 12.0, 25.0)

"""
    power_curve(farm, v)

Power [MW] produced by the farm `farm` for a wind speed `v` [m/s].
"""
function power_curve(farm::WindFarm, v::Real)
    if v < farm.v_in || v >= farm.v_out
        return 0.0
    elseif v >= farm.v_rated
        return farm.Wmax
    else
        return farm.Wmax * (v^3 - farm.v_in^3) / (farm.v_rated^3 - farm.v_in^3)
    end
end

"""
    wind_speed_forecast(T)

Synthetic forecast of the wind speed [m/s] for the next `T` hours: a weather
front arriving in the afternoon, superposed with a small daily cycle.
"""
function wind_speed_forecast(T::Int)
    return [
        7.0 + 3.0 * exp(-((t - 16) / 5)^2) + 1.0 * sin(2π * (t - 9) / 24)
        for t in 1:T
    ]
end

"""
    generate_wind_scenarios(farm, v_forecast, nscenarios; ρ=0.9, σ=0.7, seed=nothing)

Sample `nscenarios` scenarios of wind production [MW].
The speed forecast error follows an AR(1) process
    ε_t = ρ ε_{t-1} + σ η_t,   η_t ~ N(0, 1),   ε_0 = 0.

Return a matrix of dimension `(T, nscenarios)`.
"""
function generate_wind_scenarios(
    farm::WindFarm,
    v_forecast::Vector,
    nscenarios::Int;
    ρ=0.9,
    σ=0.7,
    seed=nothing,
)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)
    T = length(v_forecast)
    w = zeros(T, nscenarios)
    for k in 1:nscenarios
        ε = 0.0
        for t in 1:T
            ε = ρ * ε + σ * randn(rng)
            v = max(v_forecast[t] + ε, 0.0)
            w[t, k] = power_curve(farm, v)
        end
    end
    return w
end

"""
    imbalance_prices(λ; α_plus=0.8, α_minus=1.3)

Two-price imbalance settlement: a surplus is paid `α_plus * λ`,
a deficit is charged `α_minus * λ`.
"""
imbalance_prices(λ::Vector; α_plus=0.8, α_minus=1.3) = (α_plus .* λ, α_minus .* λ)
