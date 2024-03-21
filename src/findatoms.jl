#
#File: findatoms.jl
#Author: Ewout van der Veer
#
#Description:
#Functions for finding atoms in an image.
#

const ELLIPTICITY_CUTOFF = 1
struct ThresholdingParameters
    use_adaptive::Bool
    threshold::Real
    bias::Real
    window_size::Real
    minimum_atom_size::Int
    binarization_algorithm::Type{<:AbstractImageBinarizationAlgorithm}
    plot::Bool
end

function ThresholdingParameters(
    ;
    use_adaptive::Bool = true,
    threshold::Real = 0.5,
    bias::Real = 0.0, 
    window_size::Real = 1,
    minimum_atom_size::Int = 10, 
    binarization_algorithm::Type{<:AbstractImageBinarizationAlgorithm} = Niblack,
    plot::Bool = true
)  
    thresholding_parameters = ThresholdingParameters(
        use_adaptive,
        threshold,
        bias,
        window_size,
        minimum_atom_size,
        binarization_algorithm,
        plot)
    check_thresholding_parameters(thresholding_parameters)
    return thresholding_parameters
end

"""
    check_thresholding_parameters(
        thresholding_parameters::ThresholdingParameters
        )

    Check the validity of thresholding parameters.

    This function validates the provided thresholding parameters. The function checks whether 
    the threshold is within the range (0, 1), the window size is greater than or equal to 1, 
    and the minimum atom size is greater than or equal to 1. If any of these conditions is not 
    met, a `DomainError` is thrown.

    # Arguments
    - `thresholding_parameters::ThresholdingParameters`: 
        A struct containing thresholding parameters.

    # Throws
    - `DomainError`: 
        If any of the thresholding parameters do not meet the specified criteria.

    # Examples
    ```julia
    params = ThresholdingParameters(0.5, 5, 2)
    check_thresholding_parameters(params)  # No error will be thrown

    params = ThresholdingParameters(1.2, 5, 2)
    check_thresholding_parameters(params)  # Throws DomainError
"""
function check_thresholding_parameters(
    thresholding_parameters::ThresholdingParameters
    )

    if !(0.0 < thresholding_parameters.threshold < 1.0) 
        throw(DomainError(thresholding_parameters.threshold, 
                          "threshold must be between 0 and 1"))
    end 
    if !(thresholding_parameters.window_size >= 1) 
        throw(DomainError(thresholding_parameters.window_size, 
                          "window size must be between larger than 1"))
    end 
    if !(thresholding_parameters.minimum_atom_size >= 0) 
        throw(DomainError(thresholding_parameters.minimum_atom_size, 
                          "minimum atom size must be between larger than 0"))
    end 
end

struct AtomParameters{T<:Real}
    centroids::Matrix{T}
    sizes::Matrix{T}
    intensities::Vector{T}
    angles::Vector{T}
end

function AtomParameters(
    ;
    centroids::Matrix{T},
    sizes::Matrix{T},
    intensities::Vector{T},
    angles::Vector{T}
) where {T<:Real}
    if !(size(centroids, 2) == size(sizes, 2) == length(intensities) == length(angles))
        throw(ArgumentError("atom parameters lists do not have the same length"))
    end
    return AtomParameters(centroids, sizes, intensities, angles)
end

"""
    find_atoms(
        image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}},
        thresholding_parameters::ThresholdingParameters
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
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}},
    thresholding_parameters::ThresholdingParameters
)   

    binarized_image = binarize_image(image, thresholding_parameters)

    atom_parameters = get_atom_parameters(binarized_image, image, thresholding_parameters)

    if thresholding_parameters.plot
        plot_atomic_positions(atom_parameters.centroids, image)
    end

    (atom_parameters, binarized_image)
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
    binarize_image(
        image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}, 
        thresholding_parameters::ThresholdingParameters
    )

    Binarize the input image using the specified thresholding parameters.

    This function takes an input image and thresholding parameters and produces a binarized 
    version of the image. If `thresholding_parameters.use_adaptive` is false, a global threshold 
    is applied to the image. If `thresholding_parameters.use_adaptive` is true, an adaptive 
    thresholding algorithm is used.

    # Arguments
    - `image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}`: 
        The input image to be binarized. It can be a grayscale image or a matrix of real values.
    - `thresholding_parameters::ThresholdingParameters`: 
        A struct containing thresholding parameters.

    # Returns
    - `binarized_image::BitMatrix`: The resulting binarized image represented as a BitMatrix.

    # Examples
    ```julia
    image = rand(Gray, 256, 256)
    params = ThresholdingParameters(0.5, 5, 2, false)
    binarized_result = binarize_image(image, params)
"""
function binarize_image(
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}, 
    thresholding_parameters::ThresholdingParameters
)
    if !thresholding_parameters.use_adaptive
        optimum_threshold = (iszero(thresholding_parameters.threshold) ? 
                             find_optimum_threshold(image) : 
                             thresholding_parameters.threshold)
        binarized_image = image .> optimum_threshold
    else
        binarized_image = binarize(image, 
        thresholding_parameters.binarization_algorithm(bias=thresholding_parameters.bias, 
                                                       window_size=thresholding_parameters.window_size))
    end
end

"""
    get_atom_parameters(
        binarized_image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}},
        image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}},
        thresholding_parameters::ThresholdingParameters
    )

    Extract atom parameters from a binarized image using the specified thresholding parameters.

    This function analyzes a binarized image to identify connected components (atoms) and 
    extracts relevant parameters such as centroids, sizes, intensities, and  ellipiticity 
    angles. The extraction process is based on the provided thresholding parameters.

    # Arguments
    - `binarized_image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}`: 
        The binarized image representing identified components.
    - `image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}`: 
        The original image from which atom parameters are extracted.
    - `thresholding_parameters::ThresholdingParameters`: 
        A struct containing thresholding parameters.

    # Returns
    - `AtomParameters`:
        A struct containing extracted atom parameters, including `centroids`, `sizes`, 
            `intensities`, and `angles`.

    # Examples
    ```julia
    binarized_image = binarize_image(image, thresholding_params)
    atom_params = get_atom_parameters(binarized_image, image, thresholding_params)
"""
function get_atom_parameters(
    binarized_image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}},
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}},
    thresholding_parameters::ThresholdingParameters
)

    component_labels = label_components(binarized_image)

    #The first element corresponds to the background
    centroid_vector = component_centroids(component_labels)[2:end]
    #Turn the centroids vector into a 2 x n matrix
    centroid_matrix = centroid_vector_to_matrix(centroid_vector)

    #The width of a cluster is appox. the square root of its area
    #sizes = (sqrt.(component_lengths(component_labels)))[2:end]
    (size_matrix, ellipticity_angles) = ellipticities(component_labels)

    #Estimate the intesity from the image value at the cluster centroid
    intensities = Float64.([image[round.(Int, centroid)...] for centroid in centroid_vector])

    atom_size_filter = (size_matrix[1,:] .* size_matrix[2,:]) .> 
                        thresholding_parameters.minimum_atom_size

    AtomParameters(
        centroids = centroid_matrix[:, atom_size_filter], 
        sizes = size_matrix[:, atom_size_filter], 
        intensities = intensities[atom_size_filter], 
        angles = ellipticity_angles[atom_size_filter]
        )
end

"""
    centroid_vector_to_matrix(
        centroid_vector::Vector{<:Tuple{<:Real, <:Real}}
        )

    Convert a vector of centroid tuples to a 2xn matrix.

    This function takes a vector of centroid tuples, where each tuple represents a 2D 
    coordinate (x, y), and converts it into a 2xn matrix, where n is the number of centroids. 

    # Arguments
    - `centroid_vector::Vector{<:Tuple{<:Real, <:Real}}`: 
        A vector of 2D centroid tuples.

    # Returns
    - `centroid_matrix::Matrix{<:Real}`: 
        A 2xn matrix containing the x and y coordinates of centroids.

    # Examples
    ```julia
    centroid_vector = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    matrix_result = centroid_vector_to_matrix(centroid_vector)
"""
function centroid_vector_to_matrix(
    centroid_vector::Vector{<:Tuple{<:Real, <:Real}}
    )
    hcat(collect.(centroid_vector)...)
end

"""
    ellipticities(component_labels::AbstractMatrix{Int})

    Calculate sizes and ellipticities of connected components in a labeled component image.

    This function takes a matrix of component labels and calculates the sizes and ellipticities 
    of connected components. The components are represented by integer labels in the input matrix.

    # Arguments
    - `component_labels::AbstractMatrix{Int}`: 
        A matrix where each element represents the label of the connected component to which the 
        corresponding pixel belongs.

    # Returns
    - `(sizes::Matrix{Float64}, angles::Vector{Float64})`: 
        A tuple containing the sizes and ellipticities of connected components. 
    - `sizes`: 
        A 2xn matrix, where n is the number of connected components. Each column contains the width 
        and height of the corresponding component.
    - `angles`: 
        A vector containing the ellipticity angles (in radians) of the connected components.

    # Examples
    ```julia
    labels_matrix = [1 1 0; 0 2 2; 3 0 3]
    (sizes_result, angles_result) = ellipticities(labels_matrix)
"""
function ellipticities(
    component_labels::AbstractMatrix{Int}
)
    number_of_atoms = maximum(component_labels)

    sizes = Matrix{Float64}(undef, 2, number_of_atoms)
    angles = Vector{Float64}(undef, number_of_atoms)

    Threads.@threads for label in 1:number_of_atoms
        pixels = Tuple.(findall(x -> x == label, component_labels))
        coordinates = [first.(pixels) last.(pixels)]

        try
            (atom_size, ellipticity_angle) = atom_sizes(coordinates)
            sizes[:, label] .= atom_size
            angles[label] = ellipticity_angle
        catch
            sizes[:, label] .= [0, 0]
            angles[label] = 0
        end
        
    end
    (sizes, angles)
end

"""
    atom_sizes(coordinates::AbstractMatrix{<:Real})

    Calculate the size and ellipticity angle of an atom based on the coordinates of pixels in
    the component representing it.

    This function takes a matrix of 2D coordinates of a component and calculates the size and 
    ellipticity angle. The size is determined by the eigenvalues of the covariance matrix, and 
    the ellipticity angle is calculated from the eigenvectors.

    # Arguments
    - `coordinates::AbstractMatrix{<:Real}`: 
        A 2xn matrix representing the 2D coordinates of the pixels.

    # Returns
    - `(atom_size::Vector{Float64}, ellipticity_angle::Float64)`: 
        A tuple containing the size and ellipticity angle of the atom. 
    - `atom_size`: 
        A vector containing the width and height of the atom.
    - `ellipticity_angle`: 
        The ellipticity angle in degrees.

    # Examples
    ```julia
    atom_coordinates = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    (size_result, angle_result) = atom_sizes(atom_coordinates)
"""
function atom_sizes(coordinates::AbstractMatrix{<:Real})

    weights = ones(size(coordinates, 1)) #TODO:get weights from image
    normalized_weights = weights/sum(weights)

    center = sum(coordinates .* normalized_weights, dims=1)/sum(normalized_weights)

    #Enforce symmetry on covariance matrix so eigenvalues are always real
    covariance_matrix = Symmetric(transpose(coordinates .- center) * 
                                  Diagonal(diagm(normalized_weights)) * 
                                  (coordinates .- center) /
                                  (1 - sum(normalized_weights .^ 2))) 

    eigenvalues = eigvals(covariance_matrix)
    eigenvectors = eigvecs(covariance_matrix)

    ellipticity_angle = atand(eigenvectors[1,1] / eigenvectors[2,1])

    if ellipticity_angle <= -90
        ellipticity_angle += 180
    elseif ellipticity_angle > 90
        ellipticity_angle -= 180
    end
    
    atom_size = reverse(sqrt.(eigenvalues)) .* 4

    if atom_size[1]/atom_size[2] < ELLIPTICITY_CUTOFF
        ellipticity_angle = 0
    end
    (atom_size, ellipticity_angle)
end
