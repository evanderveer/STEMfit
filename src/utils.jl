"""
    plot_unit_cells(
        unit_cells,
        neighbors;
        xlim=(-70,70),
        ylim=(-70,70),
        kwargs...
    )

Plot a list of unit cells.
"""
function plot_unit_cells( #Very ugly --> refactor
    unit_cells,
    neighbors;
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
    plot(uc_plots..., 
         layout=(ceil(Int32,length(uc_plots)/4), 4), 
         size = (140*1.7*4, 160*1.5*ceil(Int64,length(uc_plots)/4))
         
         )

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
    plot(uc_plots..., 
         layout=(ceil(Int32,length(uc_plots)/4), 4), 
         size = (140*1.7*4, 160*1.5*ceil(Int64,length(uc_plots)/4))
         
         )

end

function show_images(images...; rows=1, enlarge=1, zoom=true, kwargs...) 
    image_sizes = size.(images)
    if !any(s1 != s2 for (s1,s2) in image_sizes) && zoom == true
        image_sizes = [s[1] for s in image_sizes]
        enlargement_factors = enlarge*maximum(image_sizes)./image_sizes
        zoomed_images = [enlarge_image(im, Int64(ef)) for (im,ef) in zip(images, enlargement_factors)]
        return mosaicview(zoomed_images, nrow=rows; kwargs...)
    end
    mosaicview(enlarge_image.(images, Int64(enlarge)), nrow=rows; kwargs...)
end

function plot_atoms_on_image(
                            image, 
                            atom_positions; 
                            xlim=nothing, 
                            ylim=nothing, 
                            markersize=2, 
                            markerstrokewidth=0, 
                            c=:red, 
                            legend=false,  
                            xaxis=false, 
                            yaxis=false, 
                            plot_size=(600,600), 
                            kwargs...)

    if xlim===nothing; xlim=(1, size(image)[2]); end
    if ylim===nothing; ylim=(1, size(image)[1]); end
    p = plot(image, xlim=xlim, ylim=ylim);
    scatter!(
            p, 
            atom_positions[2,:], 
            atom_positions[1,:],
            markersize=markersize, 
            markerstrokewidth=markerstrokewidth, 
            c=c, 
            legend=legend,  
            xaxis=xaxis, 
            yaxis=yaxis, 
            size=plot_size;
            kwargs...)
    p
end

"""
    load_image(
        filename::String;
        convert::Bool = true
    )
    -> Matrix{<:Gray{Float32}}

Load an image from a file. Optionally convert into the right format.    
"""
function load_image(
                    filename::String;
                    convert::Bool = false
                   )

    img = load(filename)
    if typeof(img) != Matrix{Gray{N0f8}} && convert == false 
        throw(ErrorException("Only 8-bit grayscale images are supported. 
Convert the image to 8-bit before importing. 
Alternatively, set convert=true to convert automatically. 
Use at your own peril!"))
    end
    if convert
        img = Matrix{Gray{N0f8}}(img)
    end
    Gray{Float64}.(img)
end

"""
    uc_area(basis_vectors::AbstractVector{<:AbstractVector{<:AbstractFloat}})
        -> Real

Calculate the area of a parallelogram of two basis vectors.
"""
function uc_area(basis_vectors::AbstractVector{<:AbstractVector{<:Real}})
    abs(cross([basis_vectors[1]..., 0],
          [basis_vectors[2]..., 0])[3])
end

"""Calculate the angle (0-360deg) of a vector wrt the vector [1,0]."""
function uc_angle(basis_vector::AbstractVector{<:Real})
    (y,x) = Float32.(basis_vector)
    if x == 0f0 || y == 0f0
        return 0f0
    elseif x<0f0 && y<0f0
        return 180f0 + atand(x/y)
    elseif x<0f0 && y>0f0
        return 360f0 + atand(x/y)
    elseif x>0f0 && y<0f0
        return 90 + atand(-x/y)
    elseif x>0f0 && y>0f0
        return atand(x/y)
    end
    throw(ArgumentError)

end

"""
    uc_angle(basis_vectors::AbstractVector{<:AbstractVector{<:AbstractFloat}})
        -> Real

Calculate the angle between two basis vectors.
"""
function uc_angle(basis_vectors::AbstractVector{<:AbstractVector{<:Real}})
    (a,b) = basis_vectors
    angle = acosd(clamp(
                dot(a,b)/(norm(a)*norm(b)), -1, 1
               )
         )
    if isnan(angle) 
        return 0f0
    end
    return angle
end



"""
    decimal_part(value::Real) -> Float64

Calculate the decimal part of a number.

# Example 
```doctest
julia> decimal_part(1.3)
0.3
```
"""
decimal_part(value::Real) = Float64(value - floor(value))

function stretch_image(image)
    im_min = minimum(image)
    im_max = maximum(image)

    (image .- im_min) ./ (im_max - im_min)
end

function residual_image(im1, im2; no_text = false)
    res_img = abs.(Gray.(im1) .- Gray.(im2))
    if !no_text
        println("Maximum deviation: "*string(Float16(maximum(res_img)*100))*" %")
        println("Total residual: "*string(Float16(residual(im1,im2))))
    end
    stretch_image(res_img)
end

function enlarge_image(
    image::AbstractMatrix{T},
    enlargement_factor::Integer
    ) where T
    image = parent(image)
    new_image = zeros(T, (size(image).*enlargement_factor)...)
    for coord in CartesianIndices(image)
        new_start = (Tuple(coord)  .- (1,1)).*enlargement_factor .+ (1,1)
        new_end = new_start .+ (enlargement_factor-1, enlargement_factor-1)
        new_range = (new_start[1]:new_end[1], new_start[2]:new_end[2])
        new_image[new_range...] .= image[coord]        
    end 
    new_image
end

function avg_perc_dev(im1, im2)
    a = [abs(px1-px2)/px1 for (px1, px2) in zip(im1,im2)]
    mean(a*100)
end

function inverse_matrix(uc::UnitCell)
    inv([uc.vector_1 uc.vector_2])
end

function save_atomic_positions!(
    filename::String,
    matrix::AbstractMatrix{<:Real};
    headers::AbstractVector{<:String}
)
    if length(headers) != size(matrix)[2]
        throw(ArgumentError("Incorrect number of headers."))
    end
    f = open(filename, "w")
    writedlm(f, permutedims(headers), ',')
    writedlm(f, matrix, ',')
    close(f)
end


function print_range(
    therange::UnitRange,
    thelist::Vector
)
    for (i,j) in zip(therange,thelist)
        println(string(i)*'\t'*string(j)*'\r')
    end 
end

function print_range(
    therange::Array,
    thelist::Vector
)
    for (i,j) in zip(therange,thelist)
        println(string(i)*'\t'*string(j)*'\r')
    end 
end

#Make sure matrices of dual numbers can be displayed as images
Gray(dual::Dual) = Gray(dual.value)
Float64(dual::Dual) = Float64(dual.value)
Float16(dual::Dual) = Float16(dual.value)
Dual(pixel::Gray{T}) where T = Dual(T(pixel))
#ForwardDiff.Dual(pixel::Gray{T}) where T = Dual(T(pixel))