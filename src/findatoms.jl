#
#File: findatoms.jl
#Author: Ewout van der Veer
#
#Description:
#Functions for finding atoms in an image.
#

"""
    find_atoms(
        image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
        [,threshold::Real = 0.0,
        use_adaptive::Bool = true,
        window_size::Integer = 8,
        bias::Real = 0.8,
        min_atom_size::Integer = 10,
        plot::Bool = true]
        ) 
        -> Tuple(Matrix{Float32}, Vector{Float32}, Vector{Float32}, Matrix{Gray{Float32}})

Detect atoms in `image` using thresholding. Adaptive thresholding is used by default.

If `threshold` is undefined, determine the optimum threshold value automatically.
If `use_adaptive` is set true, a Niblack adaptive thresholding algorithm is used instead.
The adaptive thresholding can be controlled using the `window_size` and `bias` parameters. 
`min_atom_size` defines the minimum number of pixels an atom must have.

Returns a 2 x n matrix of atom centroids, a size n vector of atom widths,
a size n vector of atom intensities and the binarized image. 
"""
function find_atoms(
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}};
    threshold::Real = 0.0,
    use_adaptive::Bool = true,
    window_size::Integer = 8,
    bias::Real = 0.8,
    min_atom_size::Integer = 10,
    binarization_algorithm::Type{<:AbstractImageBinarizationAlgorithm} = Niblack,
    plot::Bool = true
)   

    if(threshold < 0.0 || threshold > 1.0) 
        throw(DomainError(threshold, "threshold must be between 0 and 1"))
    end   

    if !use_adaptive
        optimum_thresh = (iszero(threshold) ? find_optimum_threshold(image) : threshold)
        bw_opt = image .> optimum_thresh
    else
        bw_opt = binarize(image, binarization_algorithm(bias=bias, window_size=window_size))
    end

    labels_opt = label_components(bw_opt)

    #The first element corresponds to the background
    centroids = component_centroids(labels_opt)[2:end]
    #Turn the centroids vector into a 2 x n matrix
    centroid_matrix = Matrix([j[i] for j in centroids, i in 1:2]')

    #The width of a cluster is appox. the square root of its area
    sizes = (sqrt.(component_lengths(labels_opt)))[2:end]

    #Estimate the intesity from the image value at the cluster centroid
    intensities = Float64.([
                    image[round.(Int32, centroid)...] 
                    for centroid in centroids
                    ])

    #Remove atoms which are very small (<10 px)
    size_filter = sizes .> sqrt(min_atom_size)

    if plot
        plot_atomic_positions(centroid_matrix[:, size_filter], image)
    end

    atom_parameters = [centroid_matrix[:, size_filter]; 
                       intensities[size_filter]';
                       sizes[size_filter]';
                       zeros(Float64, length(sizes[size_filter]))';
                       sizes[size_filter]'
                       ]

    (atom_parameters, bw_opt)
end

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
