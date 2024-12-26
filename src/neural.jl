module neural


include("Activations.jl")
include("Loss.jl")
include("Initialization.jl")
include("Train.jl")
include("Forward.jl")
include("Utils.jl")
include("Visualization.jl")
include("MNISTUtils.jl")


using .MNISTUtils
using .Initialization
using .Forward
using .Loss
using .Train
using .Utils
using .Visualization


end