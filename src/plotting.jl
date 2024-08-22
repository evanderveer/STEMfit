"""
File: plotting.jl
Author: Ewout van der Veer

Description:
Functions for plotting and saving of results.
"""

function get_ellipticities(
    results::Results
    )
    ratios = []
    for size in eachcol(results.atom_parameters.sizes)
        if size[1]>size[2]
            push!(ratios, size[1]/size[2])
        else
            push!(ratios, size[2]/size[1])
        end
    end
    ratios
end

function lattice_parameter_histogram(
    results::Results;
    use_nm_units::Bool = true,
    x_limits::Union{Tuple{Real, Real}, Nothing} = nothing,
    kwargs...
    )
    data = results.lattice_parameters[:, results.valid_atoms]
    
    if use_nm_units
        data = convert_to_nm(data, results.pixel_sizes)
    end

    if isnothing(x_limits)
        xlim = percentile([data[1,:]; data[2,:]], [5,95]) 
        xrange = abs(-(xlim...))
        xlim = xlim .+ (-xrange, xrange) .* 0.5
        bins=xlim[1]:(xlim[2]-xlim[1])/250:xlim[2]
    end
    
    p = histogram(data[1,:], 
                xlim=xlim, 
                bins=bins, 
                alpha=0.8,
                linewidth=0,
                label="Basis vector 1", 
                xlabel=use_nm_units ? "Lattice parameter (nm)" : "Lattice parameter (px)",
                ylabel="Count",
                dpi=300;
                kwargs...)
    histogram!(p, data[2,:], 
                bins=bins, 
                alpha=0.8,
                linewidth=0,
                label="Basis vector 2",
                dpi=300;
                kwargs...)
    display(p)
    p
end

"""
    function convert_to_nm(
        matrix::AbstractMatrix{<:Real},
        pixel_sizes::Tuple{<:Real, <:Real}
    )

    Converts the values in the first two rows of `matrix` from pixel into length
    units using the given `pixel_sizes`.
"""
function convert_to_nm(
    atom_parameters::AtomParameters,
    pixel_sizes::Tuple{<:Real, <:Real}
    ) 
    atom_parameters.centroids .*= [pixel_sizes...]
    atom_parameters.sizes .*= [pixel_sizes...]
end

function convert_to_nm(
    matrix::AbstractMatrix{<:Real},
    pixel_sizes::Tuple{<:Real, <:Real}
    ) 
    matrix .* [pixel_sizes...]
end

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


function plot_singular_vectors(
    singular_value_powers::AbstractVector{<:Real},
    number_of_singular_values::Int;
    kwargs...
    )
    x_limits = (0, round(Int, number_of_singular_values*2))
    y_limits = extrema(singular_value_powers[1:x_limits[2]]) .* (0.5, 2)

    p = scatter(singular_value_powers, yaxis=:log, 
                   xlim=x_limits, 
                   ylim=y_limits, 
                   xlabel="Singular value", 
                   ylabel="Power",
                   label=false; 
                   kwargs...)

    #Vertical line at the number of singular vectors used
    vline!(p, 
           [number_of_singular_values],
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
    plot_size = (size(image,2), size(image,1)).*image_size
    scatter!(p, atom_positions[2,:], 
             atom_positions[1,:], 
             markersize=markersize, 
             markerstrokewidth=markerstrokewidth, 
             c=c, 
             legend=false, 
             size=round.(Int64, plot_size);
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



