module Activations

relu(x) = max.(0, x)
softmax(x) = exp.(x) ./ sum(exp.(x), dims=1)

export relu, softmax

end