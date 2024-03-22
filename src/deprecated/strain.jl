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
    plot(
        image, 
        size=plot_size, 
        alpha=0.5, 
        xticks=grid_spacing*(0:floor(size(image)[2]/grid_spacing)), 
        yticks=grid_spacing*(0:floor(size(image)[1]/grid_spacing))
        )
    hline!(grid_spacing * (0:floor(size(image)[1]/grid_spacing)), c=:red, style=:dash, label=false, linewidth=2)
    vline!(grid_spacing * (0:floor(size(image)[2]/grid_spacing)), c=:red, style=:dash, label=false, linewidth=2)
end