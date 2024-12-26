module Train

using Zygote
using ProgressMeter
using Plots
using Random

include("Forward.jl")
include("Loss.jl")
include("Initialization.jl")
include("Utils.jl")

using .Forward
using .Loss
using .Initialization
using .Utils

function accuracy(params, X, Y_true)
    Y_hat = forward(X, params)
    predicted = argmax(Y_hat, dims=1)
    actual = argmax(Y_true, dims=1)
    return sum(predicted .== actual) / size(Y_true, 2)
end


function mlp_loss(params, X, Y_true)
    Y_hat = forward(X, params)
    return cross_entropy_loss(Y_hat, Y_true)
end
mlp_loss_wrapper(flat_p, structure, X, Y) = mlp_loss(unflatten_params(flat_p, structure), X, Y)


function train_epoch!(X_train_reshaped, y_train, X_test_reshaped, y_test, flat_params, param_structure, lr, batch_size, epoch, train_loss_history, train_acc_history, test_acc_history, best_params, best_test_acc)
    n_train = size(X_train_reshaped, 2)
    shuffled_idx = shuffle(1:n_train)
    num_batches = Int(ceil(n_train / batch_size))
    p = Progress(num_batches, 1, "Training epoch $epoch...", 50)

    for batch_num in 1:num_batches
        start = (batch_num - 1) * batch_size + 1
        stop_ = min(start + batch_size - 1, n_train)
        idx = shuffled_idx[start:stop_]
        Xb = X_train_reshaped[:, idx]
        Yb = y_train[:, idx]

        loss, back = Zygote.pullback(p -> mlp_loss_wrapper(p, param_structure, Xb, Yb), flat_params)
        grad_p = back(1.0)[1]

        @inbounds @simd for i in eachindex(flat_params)
            flat_params[i] -= lr * grad_p[i]
        end
        push!(train_loss_history, loss)
        next!(p)
    end
    params = unflatten_params(flat_params, param_structure)
    train_acc = accuracy(params, X_train_reshaped, y_train)
    test_acc = accuracy(params, X_test_reshaped, y_test)
    push!(train_acc_history, train_acc)
    push!(test_acc_history, test_acc)
    if test_acc > best_test_acc
        best_test_acc = test_acc
        best_params = deepcopy(params)
    end
    return train_loss_history, train_acc_history, test_acc_history, best_params, best_test_acc
end

function train_mlp(
    X_train_reshaped::Matrix{Float64},
    y_train::Matrix{Bool},
    X_test_reshaped::Matrix{Float64},
    y_test::Matrix{Bool},
    layer_dimensions::Vector{Int};
    epochs::Int = 20,
    batch_size::Int = 256,
    lr::Float64 = 0.1,
    plot_animation::Bool = true,
    animation_filename::String = "training_animation.gif"
)

    params = initialize_network_parameters(layer_dimensions)
    println("Number of parameters: ", sum(length(v) for v in values(params)))
    flat_params, param_structure = flatten_params(params)

    n_train = size(X_train_reshaped, 2)
    println("Number of training samples: ", n_train)
    train_loss_history = Float64[]
    train_acc_history = Float64[]
    test_acc_history = Float64[]
    best_params = deepcopy(params)
    best_test_acc = 0.0

    if plot_animation
        anim = @animate for epoch in 1:epochs
            train_loss_history, train_acc_history, test_acc_history, best_params, best_test_acc = train_epoch!(
                X_train_reshaped, y_train, X_test_reshaped, y_test, flat_params, param_structure, lr, batch_size, epoch, train_loss_history, train_acc_history, test_acc_history, best_params, best_test_acc
            )
            p1 = plot(train_loss_history, title="Training Loss", xlabel="Iteration", ylabel="Loss")
            p2 = plot(1:epoch,train_acc_history, title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy")
            p3 = plot(1:epoch,test_acc_history, title="Test Accuracy", xlabel="Epoch", ylabel="Accuracy")
            plot(p1, p2, p3, layout = (1, 3))
        end
        gif(anim, animation_filename, fps = 5)
    else
        for epoch in 1:epochs
            train_loss_history, train_acc_history, test_acc_history, best_params, best_test_acc = train_epoch!(
                X_train_reshaped, y_train, X_test_reshaped, y_test, flat_params, param_structure, lr, batch_size, epoch, train_loss_history, train_acc_history, test_acc_history, best_params, best_test_acc
            )
            println("Epoch $epoch | Last Batch Loss = $(train_loss_history[end]) | Train Accuracy = $train_acc_history[end] | Test Accuracy = $test_acc_history[end]")
        end
    end

    params = unflatten_params(flat_params, param_structure)

    println("Final Train Accuracy = ", accuracy(params, X_train_reshaped, y_train))
    println("Final Test  Accuracy = ", accuracy(params, X_test_reshaped,  y_test))
    println("Best Test Accuracy = ", accuracy(best_params, X_test_reshaped, y_test))

    if !plot_animation
        p1 = plot(train_loss_history, title="Training Loss", xlabel="Iteration", ylabel="Loss")
        p2 = plot(train_acc_history, title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy")
        p3 = plot(test_acc_history, title="Test Accuracy", xlabel="Epoch", ylabel="Accuracy")
        plot(p1, p2, p3, layout = (1, 3))
    end

    return params, best_params, train_loss_history, train_acc_history, test_acc_history
end

export train_epoch!, train_mlp, accuracy

end