module Initialization

function initialize_network_parameters(layer_dimensions::Vector{Int})
    parameters = Dict{String, Array{Float64}}()
    for i in 2:length(layer_dimensions)
        fan_in = layer_dimensions[i-1]
        fan_out = layer_dimensions[i]
        bound = sqrt(6 / (fan_in + fan_out)) # Xavier initialization
        W = randn(fan_out, fan_in) .* bound
        b = zeros(fan_out, 1)
        parameters["W$i"] = W
        parameters["b$i"] = b
    end
    return parameters
end

export initialize_network_parameters

end