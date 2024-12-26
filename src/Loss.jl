module Loss

function cross_entropy_loss(Y_hat, Y)
    eps = 1e-15
    Y_hat_clamped = clamp.(Y_hat, eps, 1 - eps)
    return -sum(Y .* log.(Y_hat_clamped)) / size(Y, 2)
end

export cross_entropy_loss

end