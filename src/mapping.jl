function map_lattice_parameter(
    results::Results;
    use_nm_units::Bool = true,
    kwargs...
    )
    lattice_parameters = use_nm_units ? 
                         convert_to_nm(results.lattice_parameters, results.pixel_sizes) : 
                         results.lattice_parameters
    atom_positions = use_nm_units ? 
                     convert_to_nm(results.atom_parameters.centroids, results.pixel_sizes) : 
                     results.atom_parameters.centroids

    lattice_parameters = lattice_parameters[:, results.valid_atoms]
    atom_positions = atom_positions[:, results.valid_atoms]

    clim = percentile(lattice_parameters[1,:], [5,95]) 
    crange = abs(-(clim...))
    clim = clim .+ (-crange, crange) .* 0.5

    p = scatter(atom_positions[2, :],
                -atom_positions[1, :],
                marker_z=lattice_parameters[1, :],
                markerstrokewidth=0,
                clims=Tuple(clim),
                xlabel="x (nm)",
                ylabel="y (nm)",
                title="Vector 1 lattice parameter",
                label=false,
                c=cgrad(:inferno),
                dpi=300;
                kwargs...
                )  
                
    clim = percentile(lattice_parameters[2,:], [5,95]) 
    crange = abs(-(clim...))
    clim = clim .+ (-crange, crange) .* 0.5

    q = scatter(atom_positions[2, :],
                -atom_positions[1, :],
                marker_z=lattice_parameters[2, :],
                markerstrokewidth=0,
                clims=Tuple(clim),
                xlabel="x (nm)",
                ylabel="y (nm)",
                title="Vector 2 lattice parameter",
                label=false,
                c=cgrad(:inferno),
                dpi=300;
                kwargs...
                )   
    r = plot(p,q, dpi=300)
    display(r)
    (p, q)
end

function map_ellipticity(
    results::Results;
    use_nm_units::Bool = true,
    kwargs...
    )
    atom_positions = use_nm_units ? 
                     convert_to_nm(results.atom_parameters.centroids, results.pixel_sizes) : 
                     results.atom_parameters.centroids

    ellipticities = get_ellipticities(results)[results.valid_atoms]
    angles = results.atom_parameters.angles[results.valid_atoms]
    atom_positions = atom_positions[:, results.valid_atoms]

    clim = percentile(ellipticities, [5,95]) 
    crange = abs(-(clim...))
    clim = clim .+ (-crange, crange) .* 0.5

    p = scatter(atom_positions[2, :],
                -atom_positions[1, :],
                marker_z=ellipticities,
                markerstrokewidth=0,
                clims=Tuple(clim),
                xlabel="x (nm)",
                ylabel="y (nm)",
                title="Ellipticity",
                label=false,
                c=cgrad(:inferno),
                dpi=300;
                kwargs...
                )

    circ_color_scheme = hcat(cgrad(:inferno)[20:236]..., reverse(cgrad(:inferno)[20:236])...)
    circ_color_scheme = [circ_color_scheme[2*i] for i in 1:216]
    circ_color_scheme = PlotUtils.ContinuousColorGradient(circ_color_scheme)
    clim = (-90,90)

    q = scatter(atom_positions[2, :],
                -atom_positions[1, :],
                marker_z=angles,
                markerstrokewidth=0,
                clims=Tuple(clim),
                xlabel="x (nm)",
                ylabel="y (nm)",
                title="Ellipticity angle",
                label=false,
                c=circ_color_scheme,
                dpi=300;
                kwargs...
                )  
    r = plot(p,q, dpi=300)
    display(r)
    (p, q)
end

#TODO: Fix this
function map_strain(
    atom_positions,
    strain;
    kwargs...
    )

    (p, q) = map_lattice_parameter(
        atom_positions,
        strain,
        title="strain";
        kwargs...
    )
    (p, q)
end

#TODO: Fix this
function map_layer_assignment(
    atom_positions, 
    layer_assignment;
    markersize=2)
    p = scatter(atom_positions[2, :],
                -atom_positions[1, :],
                marker_z=layer_assignment,
                size=(500,500),
                markerstrokewidth=0,
                markersize=markersize,
                xlabel="x (nm)",
                ylabel="y (nm)",
                title="Layer assignments",
                label=false,
                c=cgrad(:inferno),
                legend=false,
                dpi=300
                )
    display(p)
    p
end