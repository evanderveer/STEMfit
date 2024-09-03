"""
File: fileio.jl
Author: Ewout van der Veer

Description:
Functions for loading and saving results. 
    
"""

function save_results_nm(
    results::Results,
)
    headers = ["y0 (nm)", "x0 (nm)", "A", "length (nm)", "width (nm)", "angle (deg)"]
    data_matrix = [headers atom_parameters_to_matrix(results.atom_parameters, results.pixel_sizes)]

    lattice_parameters_nm = similar(results.lattice_parameters)
    for i in axes(lattice_parameters_nm, 2)
        lattice_parameters_nm[:, i] = results.lattice_parameters[:, i] .* results.pixel_sizes
    end
    if results.lattice_parameters !== nothing
        data_matrix = [data_matrix; ["vector 1 lp (nm)", "vector 2 lp (nm)"] lattice_parameters_nm]
    end
end

function save_results_px(
    results::Results,
)
    headers = ["y0 (px)", "x0 (px)", "A", "length (px)", "width (px)", "angle (deg)"]
    data_matrix = [headers atom_parameters_to_matrix(results.atom_parameters)]

    if results.lattice_parameters !== nothing
        data_matrix = [data_matrix; ["vector 1 lp (px)", "vector 2 lp (px)"] results.lattice_parameters]
    end
end

function save_results(
    filename::String,
    results::Results,
    )
    
    check_results_valid(results)

    if isnothing(results.pixel_sizes)
        data_matrix = save_results_px(results)
    else
        data_matrix = save_results_nm(results)
    end

    if results.strain !== nothing
        data_matrix = [data_matrix; ["vector 1 strain", "vector 2 strain"] results.strain]
    end

    if results.valid_atoms !== nothing
        data_matrix = data_matrix[:, [true; results.valid_atoms]]
    end

    writedlm(filename, permutedims(data_matrix, (2,1)), ',')
end

function check_results_valid(results::Results)
    sizes = [size(i, 2) for i in [
                                  results.atom_parameters.centroids, 
                                  results.lattice_parameters, 
                                  results.strain
                                 ] 
        if i !== nothing]
    size_set = results.valid_atoms === nothing ? Set(sizes) : Set([sizes..., length(results.valid_atoms)])
    if length(size_set) != 1
        throw(ArgumentError("all parameters must have the same length"))
    end
end

function atom_parameters_to_matrix(ap::AtomParameters)
    [ap.centroids; 
     ap.intensities';
     ap.sizes;
     ap.angles']
end

function atom_parameters_to_matrix(
    ap::AtomParameters, 
    px_size::Tuple{Real, Real}
    )
    ap_centroids_nm = similar(ap.centroids)
    for i in axes(ap_centroids_nm, 2)
        ap_centroids_nm[:, i] = ap.centroids[:, i] .* px_size
    end

    ap_sizes_nm = similar(ap.sizes)
    for i in axes(ap_sizes_nm, 2)
        px_size_rot = get_rotated_pixel_size(px_size, ap.angles[i])
        ap_sizes_nm[:, i] = ap.sizes[:, i] .* px_size_rot
    end
    [ap_centroids_nm; 
     ap.intensities';
     ap_sizes_nm;
     ap.angles']
end

function get_rotated_pixel_size(
    px_size::Tuple{Real, Real}, 
    angle::AbstractFloat
)
    (
        sqrt((px_size[1]*cosd.(-angle))^2 + (px_size[2]*sind.(-angle))^2),
        sqrt((px_size[1]*cosd.(90+angle))^2 + (px_size[2]*sind.(90+angle))^2)
    )
end

function load_atomic_parameters(
    filename::String
    )
    raw_data = readdlm(filename, ',')
    data = Float64.(raw_data[2:end, :])

    data_matrix = permutedims(data[:,1:6], (2,1))
    AtomParameters(data_matrix[1:2,:], data_matrix[4:5,:], data_matrix[3,:], data_matrix[6,:])
end