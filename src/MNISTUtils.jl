module MNISTUtils

using MLDatasets

function load_mnist()
    X_train, y_train = MNIST(split=:train)[:]
    X_test, y_test = MNIST(split=:test)[:]
    return X_train, y_train, X_test, y_test
end

function plot_mnist_image(X_train, y_train, index)
    image = transpose(reverse(X_train[:, :, index], dims=1))
    label = y_train[index]
    heatmap(image, color=:grays, axis=false, title="Label: $label")
end

export load_mnist, plot_mnist_image

end