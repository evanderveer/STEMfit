"""
    find_atoms(
        image::Matrix{<:Gray{<:AbstractFloat}}
        [,threshold::Real = 0.0,
        use_adaptive::Bool = true,
        window_size::Integer = 8,
        bias::Real = 0.8,
        min_atom_size::Integer = 10]
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
    image::Matrix{<:Gray{<:Real}};
    threshold::Real = 0.0,
    use_adaptive::Bool = true,
    window_size::Integer = 8,
    bias::Real = 0.8,
    min_atom_size::Integer = 10,
    binarization_algorithm::Type{<:AbstractImageBinarizationAlgorithm} = Niblack
)   

    if(threshold < 0.0 || threshold > 1.0) 
        throw(DomainError(threshold, "threshold must be between 0 and 1"))
    end   

    if !use_adaptive
        optimum_thresh = (iszero(threshold) ? find_opt_thresh(image) : threshold)
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

    (centroid_matrix[:, size_filter], sizes[size_filter], intensities[size_filter], bw_opt)
end

"""
    find_opt_thresh(image::Matrix{Gray{<:AbstractFloat}}) -> Float32

Finds the optimum global threshold value for detecting atoms in `image`. 

The optimal value is the value for which the number of connected components
in the image is maximized.

## Example 
```
    julia> image = rand(Float32, 100, 100)
    100x100 Matrix{Float32}
    julia> threshold = find_opt_thresh(image)
    0.27
```
"""
function find_opt_thresh(
    image::Matrix{<:Gray{<:Real}}
)

    lab_max  = Matrix(undef, 91,2)
    for (idx,i) in enumerate(0.05:0.01:0.95) #TODO: Parallelize
        bw = image .> i;
        labels = label_components(bw)
        lab_max[idx, :] = [i maximum(labels)]
    end

    max_index = findall(x -> x == maximum(lab_max[:,2]), lab_max)
    lab_max[max_index[1][1], 1]
end


"""
    filter_image(
        image::Matrix{<:Gray{<:Real}}
        [,number_of_singular_vectors::Union{Integer, Symbol} = :auto,
        kernel_size::Integer = 1,
        gaussian_convolution_only::Bool = false]
    ) 
    -> Matrix{<:Gray{<:AbstractFloat}}

Filter the image using Singular Value Decomposition and Gaussian convolution.

`number_of_singular_vectors` is the number of singular values to use. If set to :auto, 
it will be determined automatically. `kernel_size` is the size of the gaussian kernel 
that is used for the convolution. For low-resolution images, set `kernel_size = 1`. 
If the `gaussian_convolution_only` parameter is set to true, no SVD filtering is done
and only a gaussian convolution is applied to the image. Possibly useful for images with
no or little translational symmetry.
"""
function filter_image(
    image::Matrix{<:Gray{<:Real}};
    number_of_singular_vectors::Union{Integer, Symbol} = :auto,
    kernel_size::Integer = 1,
    gaussian_convolution_only::Bool = false
)
    if !gaussian_convolution_only
        svd_res = svd(image)
        U, Σ, Vᵀ = svd_res.U, Diagonal(svd_res.S), svd_res.Vt
        if number_of_singular_vectors == :auto
            number_of_singular_vectors = sum(svd_res.S .> 2)
        end
        filt_im = Gray.(
                        U[:,1:number_of_singular_vectors]*
                        Σ[1:number_of_singular_vectors, 1:number_of_singular_vectors]*
                        Vᵀ[1:number_of_singular_vectors,:]
                        )
        return (imfilter(filt_im, Kernel.gaussian(kernel_size)), svd_res.S)
    else
        return (imfilter(image, Kernel.gaussian(kernel_size)), nothing)
    end
end

"""
    get_component_convex_hulls(
        binarized_image::Matrix{<:Gray{<:Real}}
    ) 
    -> Vector{Vector{CartesianIndex{2}}}

Gets the convex hulls of all components in *binarized_image*.

If the convex hull of a component cannot be calculated, a vector containing only
CartesianIndex{0,0} is returned. 
"""
function get_component_convex_hulls(
    binarized_image::Matrix{<:Gray{<:Real}}
)
    labeled_image = label_components(binarized_image)
    comp_boxes = component_boxes(labeled_image)
    hulls = Vector{Matrix{Int64}}([])
    for box in comp_boxes
        push!(hulls, get_conv_hull(box, bin_im))
    end
    hulls
end

"""
    get_conv_hull(
        box::Vector{Tuple{Int64, Int64}}, 
        binarized_image::Matrix{<:Gray{<:Real}}
    ) 
    -> Vector{CartesianIndex{2}}

Gets the convex hull of the points inside *box* in *binarized_image*.

If the convex hull of the points cannot be calculated, a vector containing only
CartesianIndex{0,0} is returned. 
"""
function get_conv_hull(
    box::Vector{Tuple{Int64, Int64}}, 
    binarized_image::Matrix{<:Gray{<:Real}}
)
    ymin = clamp(box[1][1]-1, 1, size(binarized_image)[1])
    ymax = clamp(box[2][1]+1, 1, size(binarized_image)[1])
    xmin = clamp(box[1][2]-1, 1, size(binarized_image)[2])
    xmax = clamp(box[2][2]+1, 1, size(binarized_image)[2])
    try
        convex_hull = convexhull(Bool.(binarized_image[ymin:ymax, xmin:xmax]))
        return cartesian_indices_to_matrix(convex_hull, (ymin, xmin))
    catch
        return hcat([ymin, ymax])
    end
end

"""
    cartesian_indices_to_matrix(
        convex_hull::Vector{CartesianIndex{2}}, 
        shift::Tuple{<:Integer, <:Integer}
    ) 
    -> Matrix{<:Integer}

Convert a list of CartesianIndex{} vertices into a matrix of vertices. 
Optionally shift each vertex by *shift*.
"""
function cartesian_indices_to_matrix(
    convex_hull::Vector{CartesianIndex{2}},
    shift::Tuple{<:Integer, <:Integer} = (0,0)
)
    matrix = Matrix{Int64}(undef, 2, length(convex_hull))
    for (i,vertex) in enumerate(convex_hull)
        matrix[:, i] .= Tuple(vertex)
    end
    return(matrix .+ shift)
end