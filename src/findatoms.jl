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


"""
    filter_image(
        image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
        [,number_of_singular_vectors::Union{Integer, Symbol} = :auto,
        kernel_size::Integer = 1,
        gaussian_convolution_only::Bool = false,
        plot::Bool = false]
    ) 
    -> Matrix{<:Gray{<:AbstractFloat}}

Filter the image using Singular Value Decomposition and Gaussian convolution.

`number_of_singular_vectors` is the number of singular values to use. If set to :auto, 
it will be determined automatically. `kernel_size` is the size of the gaussian kernel 
that is used for the convolution. For low-resolution images, set `kernel_size = 1`. 
If the `gaussian_convolution_only` parameter is set to true, no SVD filtering is done
and only a gaussian convolution is applied to the image. Possibly useful for images with
no or little translational symmetry. If `plot` is set to true, a plot will be generated
of the power of each singular vector of the image for diagnostic purposes.
"""
function filter_image(
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}};
    number_of_singular_vectors::Union{Integer, Symbol} = :auto,
    kernel_size::Integer = 1,
    gaussian_convolution_only::Bool = false,
    plot::Bool = false,
    kwargs...
)
    if !gaussian_convolution_only
        svd_res = svd(image)
        U, Σ, Vᵀ = svd_res.U, Diagonal(svd_res.S), svd_res.Vt
        if number_of_singular_vectors == :auto
            log_Σ = log.(svd_res.S[1:100])
            A = [collect(1:100) ones(100)]
            sv_dist = log_Σ .- A*inv(A'*A)*A'*log_Σ
            number_of_singular_vectors = 
                round.(Int64, collect(1:100)[sv_dist .== minimum(sv_dist)]*1.2)[1]
        end
        filt_im = Gray.(
                        U[:,1:number_of_singular_vectors]*
                        Σ[1:number_of_singular_vectors, 1:number_of_singular_vectors]*
                        Vᵀ[1:number_of_singular_vectors,:]
                        )
        filtered_image = imfilter(filt_im, Kernel.gaussian(kernel_size))
    else
        filtered_image = imfilter(image, Kernel.gaussian(kernel_size))
    end

    if plot && !gaussian_convolution_only
        plot_singular_vectors(svd_res.S, number_of_singular_vectors; kwargs...)
    end

    filtered_image
end

"""
    get_component_convex_hulls(
        binarized_image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
    ) 
    -> Vector{Vector{CartesianIndex{2}}}

Gets the convex hulls of all components in *binarized_image*.

If the convex hull of a component cannot be calculated, a vector containing only
CartesianIndex{0,0} is returned. 
"""
function get_component_convex_hulls(
    binarized_image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
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
        binarized_image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
    ) 
    -> Vector{CartesianIndex{2}}

Gets the convex hull of the points inside *box* in *binarized_image*.

If the convex hull of the points cannot be calculated, a vector containing only
CartesianIndex{0,0} is returned. 
"""
function get_conv_hull(
    box::Vector{Tuple{Int64, Int64}}, 
    binarized_image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
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


"""
    function find_optimal_threshold_parameters(
        image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}; 
        min_atom_size::Integer = 5,
        max_iterations::Integer = 10)
        -> Tuple{Float64, Int64}

    Uses a heuristic iterative approach to find reasonable values for the bias and 
    window size for Niblack image thresholding given an `image` and minimum number 
    of pixels per atom `min_atom_size`. Returns the best guess for the bias and 
    window size as a tuple.

    The maximum number of iterations can be set using the keyword argument 
    `max_iterations`.

    This function does not run very fast, but it allows for automatic thresholding
    without prior knowledge about the image.
"""
function find_optimal_threshold_parameters(
                    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}; 
                    min_atom_size::Integer = 5,
                    max_iterations::Integer = 10
                                            )
    change_b = change_w = iterations = 0
    b = 1.0
    w = 1

    while (change_b < 0.2 || change_w != 0) && iterations < max_iterations
        new_w = try
            guess_w(image, b, min_atom_size)::Int64
        catch
            w
        end            
        
        change_w = w - new_w
        w = new_w

        new_b = try
            guess_b(image, new_w, min_atom_size)::Float64
        catch
            b
        end
        change_b = b - new_b
        b = new_b
        iterations += 1
    end
    (b,w)
end


"""
    guess_b(
        image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}},
        w::Integer, 
        min_atom_size::Integer
    )

    Use a heuristic approach to find the best guess for the bias given an 
    image, window size `w` and minimum number of pixels per atom `min_atom_size`.
"""
function guess_b(
                image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}},
                w::Integer, 
                min_atom_size::Integer
                )
    res = Matrix{Float64}(undef, 2, 40)
    Threads.@threads for (i,b) in collect(enumerate(LinRange(-5,5,40)))
        num_atoms = length(STEMfit.find_atoms(image, 
                                              bias=b, 
                                              window_size=w, 
                                              min_atom_size=min_atom_size)[2])::Int64
        res[:,i] .= (b, num_atoms)
    end
    num_atoms_thresh = 0.01*maximum(res[2,:])
    sum_weights = sum(res[2, res[2,:] .> num_atoms_thresh])
    sum(prod(col) for col in eachcol(res) if col[2] > num_atoms_thresh)/sum_weights
end

"""
    guess_b(
        image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}},
        w::Integer, 
        min_atom_size::Integer
    )

    Use a heuristic approach to find the best guess for the window size given an 
    image, bias `b` and minimum number of pixels per atom `min_atom_size`.
"""
function guess_w(
                image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}, 
                b::AbstractFloat, 
                min_atom_size::Integer
                )
    res = Matrix{Float64}(undef, 2, 40)
    Threads.@threads for (i,w) in collect(enumerate(LinRange(0, size(image)[1]/10, 40)))
        num_atoms = length(STEMfit.find_atoms(image, 
                                              bias=b, 
                                              window_size=round(Int64,w), 
                                              min_atom_size=min_atom_size)[2])::Int64
        res[:,i] .= (w, num_atoms)
    end
    num_atoms_thresh = 0.9*maximum(res[2,:])
    sum_weights = sum(res[2, res[2,:] .> num_atoms_thresh])
    round(Int64, sum(prod(col) for col in eachcol(res) if col[2] > num_atoms_thresh)/sum_weights)
end