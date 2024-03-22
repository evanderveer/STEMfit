"""
File: strain.jl
Author: Ewout van der Veer

Description:
Functions for calculating the local strain from atomic positions. 
"""

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