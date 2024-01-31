"""
File: latticeparameter.jl
Author: Ewout van der Veer

Description:
Functions for calculating the local lattice parameter and strain from 
atomic positions. 
    
"""

mutable struct Results{T<:Real}
    atom_parameters::AtomParameters{T}
    unit_cell::UnitCell
    lattice_parameters::Matrix{T}
    valid_atoms::BitVector
    strain::Union{Matrix{T}, Nothing}
    polarization::Union{Matrix{T}, Nothing}
    pixel_sizes::Union{Tuple{Real, Real}, Nothing}
end

function Results(
    atom_parameters::AtomParameters{T},
    unit_cell::UnitCell,
    lattice_parameters::Matrix{T},
    valid_atoms::BitVector;
    strain::Union{Matrix{T}, Nothing} = nothing,
    polarization::Union{Matrix{T}, Nothing} = nothing,
    pixel_sizes::Union{Tuple{Real, Real}, Nothing} = nothing
) where T<:Real
    if size(atom_parameters.centroids, 2) != size(lattice_parameters, 2)
        error("Number of atoms not equal to number of lattice parameter values")
    end
    if length(valid_atoms) != size(lattice_parameters, 2)
        error("Number of atoms validity labels not equal to number of lattice parameter values")
    end
    if !(isnothing(strain)) && size(strain, 2) != size(lattice_parameters, 2)
        error("Number of strain values not equal to number of lattice parameter values")
    end
    if !(isnothing(polarization)) && size(polarization, 2) != size(lattice_parameters, 2)
        error("Number of polarization values not equal to number of lattice parameter values")
    end

    Results(atom_parameters, unit_cell, lattice_parameters, valid_atoms, strain, polarization, pixel_sizes)
end

"""
    calculate_lattice_parameters(
        positions::AbstractMatrix{T},
        unit_cell::UnitCell[,
        tolerance::Real = 0.2,
        strictness::Integer = 4]
    ) where {T<:Real} -> Matrix{T}

Calculates the local lattice parameter for each atom in `positions` along the 
basis vectors of `unit_cell`.

`tolerance` determines the maximum deviation of a neighboring atom from its expected
position to be considered a valid neighbor. `strictness` is the number of valid 
neighbors required to calculate the lattice parameter (4 <= `strictness` <= 8).

Returns a matrix of lattice parameters. The first row of the matrix are the lattice
parameters along the first basis vector, the second row are the lattice parameters
along the second basis vector. 
"""
function calculate_lattice_parameters(#TODO: Clean up
    atom_parameters::AtomParameters{T},
    unit_cell::UnitCell;
    tolerance::Real = 0.2,
    strictness::Integer = 4
) where {T<:Real}

    if strictness > 8 || strictness < 4
        throw(ArgumentError("Strictness must be between 4 and 8"))
    end

    positions = atom_parameters.centroids

    vec_1 = unit_cell.vector_1
    vec_2 = unit_cell.vector_2
    inv_matrix = inv([vec_1 vec_2])
    transformed_positions = inv_matrix * positions

    tree = KDTree(transformed_positions)

    #Matrix is nearest neighbor positions in transformed space
    nn_matrix = [-1 -1 -1 0 0 1 1 1; -1 0 1 -1 1 -1 0 1]
    
    latt_param_matrix = similar(positions)
    
    for i in 1:size(transformed_positions)[2]

        #Get nearest neighbor indices and distances
        nearest_neighbor_idxs = nn(tree, transformed_positions[:, i] .+ nn_matrix)

        #Check that all neighbors are within tolerance of their expected positions
        #otherwise, reject this point
        valid_atoms = nearest_neighbor_idxs[2] .< tolerance
        if sum(valid_atoms) < strictness
            latt_param_matrix[:, i] = zeros(T, 2)
        else
            neighbor_positions = transformed_positions[:, nearest_neighbor_idxs[1]]
            diff_vectors = abs.(neighbor_positions .- transformed_positions[:, i])

            vector_1_positions = BitVector([1,1,1,0,0,1,1,1]) 
            vector_2_positions = BitVector([1,0,1,1,1,1,0,1]) 
            vec_1_latt_param = mean(diff_vectors[1, vector_1_positions .&& valid_atoms])
            vec_2_latt_param = mean(diff_vectors[2, vector_2_positions .&& valid_atoms])

            latt_param_matrix[:, i] = [vec_1_latt_param, vec_2_latt_param]
        end
    end
    #Transform back
    lattice_parameters = abs.(norm.([vec_1, vec_2]) .* latt_param_matrix)
    valid_atom_labels = valid_lattice_parameter_filter(lattice_parameters) 

    Results(atom_parameters, unit_cell, lattice_parameters, valid_atom_labels)
end

valid_lattice_parameter_filter(lattice_parameters) = (sum(lattice_parameters, dims=1) .!= 0)[1,:]

"""
    plot_image_with_grid(
        image::Union{AbstractMatrix{<:Gray{<:AbstractFloat}}, AbstractMatrix{<:AbstractFloat}},
        grid_spacing::Integer = 50;
        plot_size=nothing
    ) 

Plot `image` with a grid overlaid on top. `grid_spacing` defines the distance between grid lines. 
`plot_size` determines the size of the displayed plot. If `plot_size=nothing`, the image is displayed
at full size.
"""
function plot_image_with_grid!(
    image::Union{AbstractMatrix{<:Gray{<:AbstractFloat}}, AbstractMatrix{<:AbstractFloat}};
    grid_spacing::Integer = 50,
    plot_size = nothing
)
    if plot_size===nothing; plot_size=size(image); end
    plot(   image, 
            size=plot_size, 
            alpha=0.5, 
            xticks=grid_spacing*(0:floor(size(image)[2]/grid_spacing)), 
            yticks=grid_spacing*(0:floor(size(image)[1]/grid_spacing))
            )
    hline!(grid_spacing * (0:floor(size(image)[1]/grid_spacing)), c=:red, style=:dash, label=false, linewidth=2)
    vline!(grid_spacing * (0:floor(size(image)[2]/grid_spacing)), c=:red, style=:dash, label=false, linewidth=2)
end

"""
    function get_strain_from_lattice_parameters(
        lattice_parameters::AbstractMatrix{T},
        bulk_lattice_parameter_dict::Dict{<:Integer, <:Tuple},
        layer_assignments::AbstractVector{<:Integer}    
    ) where {T<:Real} -> Matrix{T}

Calculates the strain from the local lattice parameters values in `lattice_parameters`,
the bulk lattice_parameters `bulk_lattice_parameter_dict` and layer assignments. 
`bulk_lattice_parameter_dict` is a dictionary whose keys correspond to the layer indices
in `layer_assignments` and whose values are tuples of bulk lattice parameters in the same
directions as the two rows of `lattice_parameters`.
"""
function get_strain_from_lattice_parameters(#TODO: Make it work with Results struct
    lattice_parameters::AbstractMatrix{T},
    bulk_lattice_parameter_dict::Dict{<:Integer, <:Tuple},
    layer_assignments::AbstractVector{<:Integer}    
) where {T<:Real}
    if length(layer_assignments) != size(lattice_parameters)[2]
        throw(ArgumentError("lattice_parameters and layer_assignment must have the same size"))
    end

    #Make sure all layers have their parameters in the dict
    layer_set = (keys(bulk_lattice_parameter_dict)..., 0)
    if !all((x -> x ∈ layer_set).(Set(layer_assignments)))
        throw(ArgumentError("bulk lattice parameters of some layers were not provided"))
    end

    strain_matrix = similar(lattice_parameters)
    for (i, col) in enumerate(eachcol(lattice_parameters))
        if layer_assignments[i] == 0
            strain_matrix[:, i] = zeros(T, 2)
            continue
        end

        strain = col ./ bulk_lattice_parameter_dict[layer_assignments[i]] .- 1
        strain_matrix[:, i] = strain
    end
    strain_matrix
end

"""
    layer_assignments(
        atom_positions::AbstractMatrix{T}, 
        layer_boundaries::AbstractVector
    )
    
Creates a vector of layer indices of each atom in `atom_positions` based on 
the `layer_boundaries`. Assumes that layer boundaries are horizontal. The values
in `layer_boundaries` are the y-values of the boundaries in the image.
"""
function layer_assignments(#TODO: Make it work with Results struct
    atom_parameters::AtomParameters{T}, 
    layer_boundaries::AbstractVector;
    plot::Bool = true
    ) where T

    atom_positions = atom_parameters.centroids

    layer_boundaries = [zero(T), T.(layer_boundaries)..., maximum(atom_positions[1, :])]
    layer_filters = []
    for boundary in eachindex(layer_boundaries[2:end])
        push!(layer_filters, layer_boundaries[boundary] .<
                             atom_positions[1, :] .<= 
                             layer_boundaries[boundary+1]) 
    end
    layer_assignments = sum(filter .* i for (i,filter) in enumerate(layer_filters), dims=2)

    if plot
        map_layer_assignment(atom_positions, layer_assignments)
    end
    layer_assignments
end

"""
    get_pixel_size(
        reference_latt_param::AbstractMatrix{<:Real},
        basis_vector_distances::Tuple{<:Real, <:Real}
    )

Calculates the pixel size based on a known unit cell
"""
function get_pixel_size(
    uc::UnitCell,
    basis_vector_distances::Tuple{<:Real, <:Real}
)
    pixel_distances = norm.((uc.vector_1, uc.vector_2))
    Tuple(basis_vector_distances ./ pixel_distances)
end

"""
    get_pixel_size(
        reference_latt_param::AbstractMatrix{<:Real},
        basis_vector_distances::Tuple{<:Real, <:Real}
    )

Calculates the pixel size based on a reference
"""
function add_pixel_size(
    results::Results,
    range_y::Union{UnitRange, StepRangeLen},
    range_x::Union{UnitRange, StepRangeLen},
    known_lattice_parameters::Tuple{Real, Real}
)
    reference_filter = isinrange.(results.atom_parameters.centroids[1,:], Ref(range_y)) .&& 
                       isinrange.(results.atom_parameters.centroids[2,:], Ref(range_x)) .&& 
                       results.valid_atoms
    reference_lattice_parameters = results.lattice_parameters[:, reference_filter]
    pixel_distances = mean(reference_lattice_parameters, dims=2)
    results.pixel_sizes = Tuple(known_lattice_parameters ./ pixel_distances)
end

isinrange(value, range) = minimum(range) < value < maximum(range)