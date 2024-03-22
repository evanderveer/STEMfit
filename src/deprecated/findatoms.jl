"""
    find_optimum_threshold(image::Matrix{Gray{<:AbstractFloat}}) -> Float32

    Finds the optimum global threshold value for detecting atoms in `image`. 

    The optimal value is the value for which the number of connected components
    in the image is maximized.

    ## Example 
    ```
        julia> image = rand(Float32, 100, 100)
        100x100 Matrix{Float32}
        julia> threshold = find_optimum_threshold(image)
        0.27
    ```
"""
function find_optimum_threshold(
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
    )

    lab_max  = Matrix(undef, 91,2)
    Threads.@threads for (idx,i) in collect(enumerate(0.05:0.01:0.95)) 
        bw = image .> i;
        labels = label_components(bw)
        lab_max[idx, :] = [i maximum(labels)]
    end

    max_index = findall(x -> x == maximum(lab_max[:,2]), lab_max)
    lab_max[max_index[1][1], 1]
end