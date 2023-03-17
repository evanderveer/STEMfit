"""
File: plotting.jl
Author: Ewout van der Veer

Description:
Functions for plotting and saving of results.
    
"""

function plot_singular_vectors(
    Σ,
    num_sv;
    kwargs...
)
    max_x = round(Int64, num_sv*2)
    ylim = extrema(Σ[1:max_x]) .* (0.5, 2)
    p = scatter(Σ, yaxis=:log, 
                   xlim=(0, max_x), 
                   ylim=ylim, 
                   xlabel="Singular value", 
                   ylabel="Power",
                   label=false; 
                   kwargs...)
    vline!(p, 
           [num_sv],
           label="Maximum singular value")
    display(p);
end

function plot_atomic_positions(
    atom_positions,
    image;
    markersize=2,
    markerstrokewidth=0,
    c=:red,
    image_size = 3/4,
    kwargs...
)
    p = plot(image)
    scatter!(p, atom_positions[2,:], 
             atom_positions[1,:], 
             markersize=markersize, 
             markerstrokewidth=markerstrokewidth, 
             c=c, 
             legend=false, 
             size=round.(Int64, size(image).*image_size);
             kwargs...)
    display(p);
end

"""
    plot_unit_cells(
        unit_cells,
        neighbors;
        kwargs...
    )

Plot a list of unit cells.
"""
function plot_unit_cells( #Very ugly --> refactor
    unit_cells,
    neighbors;
    kwargs...
)
    unit_cell(vectors) = Shape(
        [0,vectors[1][2], vectors[1][2]+vectors[2][2], vectors[2][2]],
        [0,vectors[1][1], vectors[1][1]+vectors[2][1], vectors[2][1]]
                                )

    xlim = 1.5 .* extrema(neighbors[2,:])
    ylim = 1.5 .* extrema(neighbors[1,:])
    uc_plots = []
    for (i,uc) in enumerate(unit_cells) 
        p = scatter(
                    neighbors[2,:], 
                    neighbors[1,:], 
                    xlim=xlim, 
                    ylim=ylim, 
                    label=false,
                    xaxis=false,
                    yaxis=false;
                    kwargs...
                    )
        plot!(
              p, 
              unit_cell([uc.vector_1, uc.vector_2]), 
              opacity=.5, 
              annotations=(xlim[2]*0.5, ylim[2]*0.8,"Unit cell " * string(i)),
              annotationfontsize=10,
              annotationhaligns=:center,
              label=false,
              c=:red
              )
        push!(uc_plots, p)
    end
    r = plot(uc_plots..., 
         layout=(ceil(Int32,length(uc_plots)/4), 4), 
         size = (140*1.7*4, 160*1.5*ceil(Int64,length(uc_plots)/4))
         
         )
    display(r)
end

function plot_unit_cells( #Very ugly --> refactor
    unit_cells;
    xlim=(-70,70),
    ylim=(-70,70),
    kwargs...
)
    unit_cell(vectors) = Shape(
        [0,vectors[1][2], vectors[1][2]+vectors[2][2], vectors[2][2]],
        [0,vectors[1][1], vectors[1][1]+vectors[2][1], vectors[2][1]]
                                )

    uc_plots = []
    for (i,uc) in enumerate(unit_cells) 
        p = plot( 
              unit_cell([uc.vector_1, uc.vector_2]), 
              opacity=.5, 
              annotations=(xlim[2]*0.5, ylim[2]*0.8,"Unit cell " * string(i)),
              annotationfontsize=10,
              annotationhaligns=:center,
              xlim=xlim, 
              ylim=ylim, 
              xaxis=false,
              yaxis=false,
              label=false,
              c=:red
              )
        push!(uc_plots, p)
    end
    r=plot(uc_plots..., 
         layout=(ceil(Int32,length(uc_plots)/4), 4), 
         size = (140*1.7*4, 160*1.5*ceil(Int64,length(uc_plots)/4))
         
         )
    display(r)
end

function plot_histogram(
    data::AbstractMatrix{<:AbstractFloat};
    xlabel
)
    xlim = percentile([data[1,:]; data[2,:]], [5,95]) 
    xrange = abs(-(xlim...))
    xlim = xlim .+ (-xrange, xrange) .* 0.5
    bins=xlim[1]:(xlim[2]-xlim[1])/250:xlim[2]

    p = histogram(data[1,:], 
                xlim=xlim, 
                bins=bins, 
                alpha=0.8,
                linewidth=0,
                label="Basis vector 1", 
                xlabel=xlabel,
                ylabel="Count",
                dpi=300)
    histogram!(p, data[2,:], 
                bins=bins, 
                alpha=0.8,
                linewidth=0,
                label="Basis vector 2",
                dpi=300)
    display(p)
    p
end

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

function map_lattice_parameter(
    atom_positions,
    lattice_parameters;
    markersize = 2,
    title = "lattice parameter"
)

    clim = percentile(lattice_parameters[1,:], [5,95]) 
    crange = abs(-(clim...))
    clim = clim .+ (-crange, crange) .* 0.5

    p = scatter(atom_positions[2, :],
                -atom_positions[1, :],
                marker_z=lattice_parameters[1, :],
                size=(600,600),
                markerstrokewidth=0,
                markersize=markersize,
                clims=Tuple(clim),
                xlabel="x (nm)",
                ylabel="y (nm)",
                title="Vector 1 "*title,
                label=false,
                c=cgrad(:inferno),
                dpi=300
                )  
                
    clim = percentile(lattice_parameters[2,:], [5,95]) 
    crange = abs(-(clim...))
    clim = clim .+ (-crange, crange) .* 0.5

    q = scatter(atom_positions[2, :],
                -atom_positions[1, :],
                marker_z=lattice_parameters[2, :],
                size=(600,600),
                markerstrokewidth=0,
                markersize=markersize,
                clims=Tuple(clim),
                xlabel="x (nm)",
                ylabel="y (nm)",
                title="Vector 2 "*title,
                label=false,
                c=cgrad(:inferno),
                dpi=300
                )   
    r = plot(p,q, size=(1200,450), dpi=300)
    display(r)
    (p, q)
end

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


function save_atomic_parameters(
    filename::String;
    atom_parameters::AbstractMatrix{<:Real},
    lattice_parameters::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    strain::Union{AbstractMatrix{<:Real}, Nothing} = nothing,
    valid_atoms::Union{AbstractVector{Bool}, Nothing} = nothing
)
    if all([atom_parameters, lattice_parameters, strain, valid_atoms] .=== nothing)
        throw(ArgumentError("no data was provided"))
    end

    sizes = [size(i)[2] for i in [atom_parameters, lattice_parameters, strain] 
                                if i !== nothing]
    size_set = valid_atoms === nothing ? Set(sizes) : Set([sizes..., length(valid_atoms)])

    if length(size_set) != 1
        throw(ArgumentError("all parameters must have the same length"))
    end

    data_matrix = [["y0", "x0", "A", "σX", "θ", "σY"] atom_parameters]

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

function load_atomic_parameters(
    filename::String
    )
    raw_data = readdlm(filename, ',')
    data = Float64.(raw_data[2:end, :])

    permutedims(data[:,1:6], (2,1))
end