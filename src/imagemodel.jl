#
#File: imagemodel.jl
#Author: Ewout van der Veer
#
#Description:
# Functions dealing with a model of an atomic resolution TEM image.
#

mutable struct ImageModel{T<:Real, U<:NNTree, V<:Real, Y<:Real}
    gaussian_parameters::Vector{MVector{6, T}}
    unit_cell::UnitCell{V}
    atom_tree::U
    background::Matrix{Y} 
    nearest_neighbor_indices_tensor::Array{Int32}
    size::Tuple{Int64,Int64}
end

function ImageModel(
    gaussian_parameters::AbstractMatrix{T},
    unit_cell,
    background;
    num_nearest_neighbors = 10
) where {T<:Real}
    parameter_vectors = [MVector{6, T}(col) for col in eachcol(gaussian_parameters)]
    tree = KDTree(gaussian_parameters[1:2,:])

    nearest_neighbor_indices_tensor = Array{Int32}(undef, 
                                                   size(background)..., 
                                                   num_nearest_neighbors)
    fill_index_tensor!(nearest_neighbor_indices_tensor, tree)
    
    ImageModel(parameter_vectors, 
               unit_cell, 
               tree, 
               T.(background), 
               nearest_neighbor_indices_tensor,
               size(background))
end

"""
    fill_index_tensor!(
        tensor::AbstractArray{<:Integer}, 
        tree::NNTree
    )

Fills `tensor[y,x,1:n]` with the n nearest neighbors for each point x,y
"""
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

"""
    intensity(
        parameters::AbstractVector,
        y::Real,
        x::Real
    )

Calculates the intensity of a gaussian defined by `parameters` at a point x,y
"""
function intensity(
    parameters::AbstractVector,
    y::Real,
    x::Real
    ) 
    (y0, x0, A, a, b, c) = parameters
    A*exp(-(a * (x - x0)^2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)^2))
end

function produce_image(
    image_model::ImageModel{T,U,V,Y};
    range = nothing,
    output_image::Union{Nothing, AbstractMatrix{T}} = nothing,
    slice::Union{Nothing, AbstractMatrix{T}} = nothing,
    kwargs...
) where {T,U,V,Y}

    #If no ranges are given, use the full model size
    if range === nothing
        range = (1:image_model.size[1], 1:image_model.size[2]) 
    end

    #Preallocate matrices for image and individual slices
    if output_image === nothing
        output_image = parent(zeros(T, range...))
    else
        output_image .= zero(T)
    end

    if slice === nothing
        slice = parent(zeros(T, range...))
    else
        slice .= zero(T)
    end

    fill_image!(
        output_image, 
        slice, 
        image_model.nearest_neighbor_indices_tensor, 
        image_model.gaussian_parameters,
        range;
        kwargs...
        )

    image_model.background[range...] .+ parent(output_image)
end


function fill_image!(
    output_image::AbstractMatrix{<:Real}, 
    slice::AbstractMatrix{<:Real}, 
    nearest_neighbor_indices_tensor::AbstractArray{<:Integer}, 
    gaussian_parameters::AbstractVector{<:AbstractVector{<:Real}},
    range::Tuple{<:UnitRange, <:UnitRange};
    kwargs...
)
    for j in 1:size(nearest_neighbor_indices_tensor)[3]
        fill_slice_intensity!(
            OffsetArray(slice, range...),
            OffsetArray(nearest_neighbor_indices_tensor[range..., j], range...), 
            gaussian_parameters;
            kwargs...
            )
        output_image .+= parent(slice)
    end
end


function fill_slice_intensity!(
    slice::AbstractMatrix{T},
    nearest_neighbor_index_matrix::AbstractMatrix{<:Integer}, 
    gaussian_parameters::AbstractVector{<:AbstractVector{<:Real}};
    exclude_index::Int32 = Int32(0)
    ) where {T<:Real}
     for i in CartesianIndices(slice)
        (y,x) = Tuple(i)
        if nearest_neighbor_index_matrix[i] == exclude_index
            @inbounds slice[i] = zero(T)
        else
            @inbounds slice[i] = 
            intensity(gaussian_parameters[nearest_neighbor_index_matrix[i]], y, x)
        end
    end
end

function transform_gaussian_parameters(
    σ_Y::Real,
    θ::Real,
    σ_X::Real    
)
    a = cosd(θ)^2/(2*σ_X^2) + sind(θ)^2/(2*σ_Y^2)
    c = sind(θ)^2/(2*σ_X^2) + cosd(θ)^2/(2*σ_Y^2)
    b = -sind(2*θ)/(4*σ_X^2) + sind(2*θ)/(4*σ_Y^2)
    (a, b, c)
end

function inverse_transform_gaussian_parameters(
    a::T,
    b::T,
    c::T
) where {T<:Real}

    if a-c == zero(T)
        θ = 0.0 #degrees
    else
        θ = 0.5*atand(2*b/(a-c)) - 90.0
    end
    σ_X = sqrt(1/(2*(a*cosd(θ)^2 + 2*b*cosd(θ)*sind(θ) + c*sind(θ)^2)))
    σ_Y = sqrt(1/(2*(a*sind(θ)^2 - 2*b*cosd(θ)*sind(θ) + c*cosd(θ)^2)))
    (σ_Y, θ, σ_X)
end

function get_initial_gaussian_parameters(
    σ_Y::AbstractVector{T},
    θ::AbstractVector{T},
    σ_X::AbstractVector{T}    
) where {T<:Real}
    length(σ_X) == length(σ_Y) == length(θ) || 
            throw(ArgumentError("arguments must have the same length"))


    a = Vector{T}(undef, length(θ))
    b = Vector{T}(undef, length(θ))
    c = Vector{T}(undef, length(θ))
    for (n,(i,j,k)) in enumerate(zip(σ_Y, θ, σ_X))
        (a[n], b[n], c[n]) = transform_gaussian_parameters(i,j,k)
    end
    (a, b, c)
end

function get_initial_gaussian_parameters(
    σ_Y::AbstractVector{T},
    σ_X::AbstractVector{T}
) where {T<:Real}
    get_initial_gaussian_parameters(σ_Y, zeros(T, length(σ_X)), σ_X)
end

function get_initial_gaussian_parameters(
    σ::AbstractVector{T}
) where {T<:Real}
    get_initial_gaussian_parameters(σ, zeros(T, length(σ)), σ)
end

function fitting_parameters(
    atom_parameters,
    background_image
)
    atom_positions = atom_parameters[1:2, :]
    atom_intensities = atom_parameters[3, :]
    σ_Y = atom_parameters[4, :]
    θ = atom_parameters[5, :]
    σ_X = atom_parameters[6, :]
    (a, b, c) = get_initial_gaussian_parameters(σ_Y, θ, σ_X)
    A = get_intensity_above_background(atom_positions, atom_intensities, background_image);
    [atom_positions; A';a';b';c'];
end

function model_to_matrix(
    model::ImageModel{T,U,V,Y}
) where {T,U,V,Y}

    matrix = Matrix{T}(undef, 6, length(model.gaussian_parameters))
    for (i, parameters) in enumerate(model.gaussian_parameters)
        matrix[:,i] = vec(parameters)
    end
    matrix
end

function save_model(
    model::ImageModel,
    filename::String;
    add_background::Bool = false
    )
    matrix = model_to_matrix(model)
    (max_y, max_x) = size(model.background)
    for col in eachcol(matrix)
        col[4:6] .= inverse_transform_gaussian_parameters(col[4:6]...)

        if add_background
            position = [clamp(col[1], 1, max_y), clamp(col[2], 1, max_x)]
            col[3] = col[3] + model.background[round.(Int, position)...]
        end
    end
    f = open(filename, "w")
    writedlm(f, ["y0" "x0" "A" "σX" "θ" "σY"], ',')
    writedlm(f, matrix', ',')
    close(f)
end