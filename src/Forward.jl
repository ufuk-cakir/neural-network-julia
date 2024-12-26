module Forward

include("Activations.jl")
using .Activations

function layer_forward(W, b, A_prev, activation_fn=relu)
    Z = W * A_prev .+ b
    A = activation_fn(Z)
    return A, Z
end

function forward(X, parameters)
    A = X
    num_layers = length(parameters) ÷ 2
    for i in 2:num_layers
        W = parameters["W$i"]
        b = parameters["b$i"]
        A, _ = layer_forward(W, b, A)
    end
    W = parameters["W$(num_layers+1)"]
    b = parameters["b$(num_layers+1)"]
    A, _ = layer_forward(W, b, A, softmax)
    return A
end

export forward, layer_forward

end