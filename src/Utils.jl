module Utils

function one_hot_encode(y, num_classes)
    [i == y[j] + 1 for i in 1:num_classes, j in 1:length(y)]
end

function preprocess_data(X_train, y_train, X_test, y_test)

    y_train = one_hot_encode(y_train, 10)
    y_test  = one_hot_encode(y_test,  10)

    X_train_reshaped = reshape(Float64.(X_train), 28*28, size(X_train, 3))
    X_test_reshaped  = reshape(Float64.(X_test),  28*28, size(X_test, 3))

    return X_train_reshaped, y_train, X_test_reshaped, y_test
end

function flatten_params(params)
    flat = Float64[]
    structure = Dict{String, Tuple{Int,Int,Int}}()
    currpos = 1
    for k in sort(collect(keys(params)))
        w = params[k]
        n = length(w)
        push!(flat, vec(w)...)
        structure[k] = (currpos, n, size(w, 1))
        currpos += n
    end
    return flat, structure
end

function unflatten_params(vec_params::Vector{T}, structure::Dict{String, Tuple{Int,Int,Int}}) where T
    params = Dict{String,Array{T}}()
    for (k, (start, n, nrow)) in structure
        slice_ = vec_params[start:start+n-1]
        ncol = n ÷ nrow
        params[k] = reshape(slice_, (nrow, ncol))
    end
    return params
end

export one_hot_encode, flatten_params, unflatten_params, preprocess_data

end