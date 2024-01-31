function save_atomic_parameters_mat(
    filename::String,
    atom_parameters::AbstractMatrix{<:Real},
    lattice_parameters::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    strain::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    valid_atoms::Union{AbstractVector{Bool}, Nothing} = nothing
)
    if all([atom_parameters, lattice_parameters, strain, valid_atoms] .=== nothing)
        throw(ArgumentError("no data was provided"))
    end

    sizes = [size(i, 2) for i in [atom_parameters, lattice_parameters, strain] 
                                if i !== nothing]
    size_set = valid_atoms === nothing ? Set(sizes) : Set([sizes..., length(valid_atoms)])

    if length(size_set) != 1
        throw(ArgumentError("all parameters must have the same length"))
    end

    if size(atom_parameters, 1) == 6
        data_matrix = [["y0", "x0", "A", "length", "width", "angle"] atom_parameters]
    elseif size(atom_parameters, 1) == 2
        data_matrix = [["y0", "x0"] atom_parameters]
    end

    if lattice_parameters !== nothing
        data_matrix = [data_matrix; ["vector 1 lp", "vector 1 lp"] lattice_parameters]
    end

    if strain !== nothing
        data_matrix = [data_matrix; ["vector 1 strain", "vector 1 strain"] strain]
    end

    if valid_atoms !== nothing
        data_matrix = data_matrix[:, [true; valid_atoms]]
    end

    writedlm(filename, permutedims(data_matrix, (2,1)), ',')
end

function save_atomic_parameters(
    filename::String,
    atom_parameters::AtomParameters,
    lattice_parameters::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    strain::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    valid_atoms::Union{AbstractVector{Bool}, Nothing} = nothing
)
    atom_parameters_matrix = [atom_parameters.centroids; 
                       atom_parameters.intensities';
                       atom_parameters.sizes;
                       atom_parameters.angles']
    save_atomic_parameters_mat(filename,
                           atom_parameters_matrix,
                           lattice_parameters,
                           strain,
                           valid_atoms)
end

function save_atomic_parameters(
    filename::String,
    results::Results,

)
    
    save_atomic_parameters(filename,
                           results.atom_parameters,
                           results.lattice_parameters,
                           results.strain,
                           results.valid_atoms)
end

function load_atomic_parameters(
    filename::String
    )
    raw_data = readdlm(filename, ',')
    data = Float64.(raw_data[2:end, :])

    data_matrix = permutedims(data[:,1:6], (2,1))
    AtomParameters(data_matrix[1:2,:], data_matrix[4:5,:], data_matrix[3,:], data_matrix[6,:])
end