mutable struct ImageModel{T<:Real, U<:NNTree, V<:Real, Y<:Real}
    gaussian_parameters::Vector{SVector{6, T}}
    unit_cell::UnitCell{V}
    atom_tree::U
    background::Matrix{Y} 
    nearest_neighbor_indices_tensor::Array{Int32}
end

function ImageModel(
    gaussian_parameters::AbstractMatrix{T},
    unit_cell,
    background;
    num_nearest_neighbors = 10
) where {T<:Real}
    parameter_vectors = [SVector{6, T}(col) for col in eachcol(gaussian_parameters)]
    tree = KDTree(gaussian_parameters[1:2,:])

    nearest_neighbor_indices_tensor = Array{Int32}(undef, 
                                                   size(background)..., 
                                                   num_nearest_neighbors)
    fill_index_tensor!(nearest_neighbor_indices_tensor, tree)
    
    ImageModel(parameter_vectors, unit_cell, tree, T.(background), nearest_neighbor_indices_tensor)
end


function fill_index_tensor!(
    tensor::AbstractArray{<:Integer}, 
    tree::NNTree
    )
    n = size(tensor)[3]
    Threads.@threads for i in CartesianIndices(tensor[:,:,1])
        (y,x) = Tuple(i)
        @inbounds tensor[y, x, :] = Int32.(knn(tree, [y,x], n)[1])
    end
end


function intensity(
    model::AbstractVector,
    y::Real,
    x::Real
    ) 
    (y0, x0, A, a, b, c) = model
    A*exp(-(a * (x - x0)^2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)^2))
end

function produce_image(
    image_model::ImageModel{T,U,V,Y}
) where {T,U,V,Y}
    image_size = size(image_model.nearest_neighbor_indices_tensor[:,:,1])
    output_image = zeros(T, image_size...)
    slice = zeros(T, image_size...)
    fill_image!(
        output_image, 
        slice, 
        image_model.nearest_neighbor_indices_tensor, 
        image_model.gaussian_parameters
        )
    image_model.background .+ output_image
end

function produce_image(
    image_model::ImageModel{T,U,V,Y},
    yrange::UnitRange,
    xrange::UnitRange
) where {T,U,V,Y}

    image_size = (yrange.stop - yrange.start + 1, xrange.stop - xrange.start + 1)
    output_image = zeros(T, image_size...)
    slice = zeros(T, image_size...)
    fill_image!(
        output_image, 
        slice, 
        image_model.nearest_neighbor_indices_tensor, 
        image_model.gaussian_parameters,
        yrange,
        xrange
        )
    image_model.background[yrange, xrange] .+ output_image
end

function produce_image(
    u::AbstractVector{T},
    p::GaussianParameters
) where {T}

    image_size = (yrange.stop - yrange.start + 1, xrange.stop - xrange.start + 1)
    output_image = zeros(T, image_size...)
    slice = zeros(T, image_size...)
    fill_image!(
        output_image, 
        slice, 
        nearest_neighbor_indices_tensor, 
        gaussian_parameters,
        yrange,
        xrange
        )
    background[yrange, xrange] .+ output_image
end

function fill_image!(
    output_image::AbstractMatrix{<:Real}, 
    slice::AbstractMatrix{<:Real}, 
    nearest_neighbor_indices_tensor::AbstractArray{<:Integer}, 
    gaussian_parameters::AbstractVector{<:AbstractVector{<:Real}}
)
    for j in 1:size(nearest_neighbor_indices_tensor)[3]
        fill_slice_intensity!(
            slice,
            nearest_neighbor_indices_tensor[:, :, j], 
            gaussian_parameters
            )
        output_image .+= slice
    end
end

function fill_image!(
    output_image::AbstractMatrix{<:Real}, 
    slice::AbstractMatrix{<:Real}, 
    nearest_neighbor_indices_tensor::AbstractArray{<:Integer}, 
    gaussian_parameters::AbstractVector{<:AbstractVector{<:Real}},
    yrange::UnitRange, 
    xrange::UnitRange
)
    for j in 1:size(nearest_neighbor_indices_tensor)[3]
        fill_slice_intensity!(
            slice,
            nearest_neighbor_indices_tensor[yrange, xrange, j], 
            gaussian_parameters,
            yrange,
            xrange
            )
        output_image .+= slice
    end
end

function fill_slice_intensity!(
    slice::AbstractMatrix{<:Real},
    nearest_neighbor_index_matrix::AbstractMatrix{<:Integer}, 
    gaussian_parameters::AbstractVector{<:AbstractVector{<:Real}}
    )
     for i in CartesianIndices(slice)
        @inbounds slice[i] = 
            intensity(gaussian_parameters[nearest_neighbor_index_matrix[i]], Tuple(i)...)
    end
end

function fill_slice_intensity!(
    slice::AbstractMatrix{<:Real},
    nearest_neighbor_index_matrix::AbstractMatrix{<:Integer}, 
    gaussian_parameters::AbstractVector{<:AbstractVector{<:Real}},
    yrange::UnitRange,
    xrange::UnitRange
    )
     for i in CartesianIndices(slice)
        (y,x) = (yrange.start, xrange.start) .+ Tuple(i)
        @inbounds slice[i] = 
            intensity(gaussian_parameters[nearest_neighbor_index_matrix[i]], y, x)
    end
end



function transform_gaussian_parameters(
    σ_X::Real,
    σ_Y::Real,
    θ::Real
)
    a = cos(θ)^2/(2*σ_X^2) + sin(θ)^2/(2*σ_Y^2)
    c = sin(θ)^2/(2*σ_X^2) + cos(θ)^2/(2*σ_Y^2)
    b = -sin(2*θ)/(4*σ_X^2) + sin(2*θ)/(4*σ_Y^2)
    (a, b, c)
end

function get_initial_gaussian_parameters(
    σ_X::AbstractVector{T},
    σ_Y::AbstractVector{T},
    θ::AbstractVector{T}
) where {T<:Real}
    length(σ_X) == length(σ_Y) == length(θ) || 
            throw(ArgumentError("arguments must have the same length"))


    a = Vector{T}(undef, length(θ))
    b = Vector{T}(undef, length(θ))
    c = Vector{T}(undef, length(θ))
    for (n,(i,j,k)) in enumerate(zip(σ_X,σ_Y,θ))
        (a[n], b[n], c[n]) = transform_gaussian_parameters(i,j,k)
    end
    (a, b, c)
end

function get_initial_gaussian_parameters(
    σ_X::AbstractVector{T},
    σ_Y::AbstractVector{T}
) where {T<:Real}
    get_initial_gaussian_parameters(σ_X, σ_Y, zeros(T, length(σ_X)))
end

function get_initial_gaussian_parameters(
    σ::AbstractVector{T}
) where {T<:Real}
    get_initial_gaussian_parameters(σ, σ, zeros(T, length(σ)))
end
"""
function check_model(
    image_model;
    fix=true
)
    check_function(g) = check_gaussian(g, 
                        1:image_model.model_size[2], 
                        1:image_model.model_size[1]
                        )
    faulty_gaussians = sum(check_function.(image_model.gaussians))
    println(string(faulty_gaussians) * " Gaussian functions were found to have problems.")
    if fix
        filter!(g -> !check_function(g), image_model.gaussians)
        new_positions = hcat([Float64.([g.y0, g.x0]) for g in image_model.gaussians]...)

        image_model.atom_tree = KDTree(new_positions)
    end
end

function check_gaussian(
    gaussian::Gaussian,
    valid_range_x = -Inf:Inf,
    valid_range_y = -Inf:Inf
)
    0 < gaussian.A < 1 &&
    isposdef([gaussian.a gaussian.b;gaussian.b gaussian.c]) &&
    gaussian.x0 in valid_range_x &&
    gaussian.y0 in valid_range_y
end"""

function convert_image_model_to_dual!(
    image_model::ImageModel{T,U,V,Y}
    ) where {T,U,V,Y}
    Z = ForwardDiff.Dual{Nothing, T, 6}
    f(x) = SVector{6, Z}([Z(element) for element in x])
    new_gaussian_parameters = f.(image_model.gaussian_parameters)
    image_model = ImageModel(
                    new_gaussian_parameters, 
                    image_model.unit_cell,
                    image_model.atom_tree,
                    image_model.background,
                    image_model.nearest_neighbor_indices_tensor                    
                    )
    return image_model
end