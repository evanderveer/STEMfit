function plot_singular_vectors(
    Σ,
    num_sv;
    kwargs...
)
    max_x = round(Int64, num_sv*2)
    ylim = round.(Int64, extrema(Σ[1:max_x]) .* (0.5, 2))
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