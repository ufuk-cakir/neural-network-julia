module Visualizations

using Plots

function plot_training_progress(train_loss, train_acc, test_acc, epoch)
    p1 = plot(train_loss, title="Training Loss", xlabel="Iteration", ylabel="Loss")
    p2 = plot(1:epoch, train_acc, title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy")
    p3 = plot(1:epoch, test_acc, title="Test Accuracy", xlabel="Epoch", ylabel="Accuracy")
    plot(p1, p2, p3, layout=(1, 3))
end

function plot_mnist(X, Y, index)
    image = X[:, :, index]  # Correct the orientation by flipping the matrix
    label = Y[index]

    #rotate image 90 degrees clockwise 
    image = transpose(image)
    image = reverse(image, dims=1)
    heatmap(
        image,  # Correct the orientation by transposing the matrix
        color=:grays,
        axis=false,
        title="Label: $label"
    )
end


export plot_training_progress, plot_mnist

end