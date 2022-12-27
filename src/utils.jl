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
    Gray{Float32}.(img)
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
function uc_angle(basis_vector::Vector{<:Real})
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

function residual_image(im1, im2)
    res_img = abs.(Gray.(im1) .- Gray.(im2))
    println("Maximum deviation: "*string(Float16(maximum(res_img)*100))*" %")
    stretch_image(res_img)
end

function residual(im1, im2)
    Float32(sum((im1 .- im2).^2))
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

"""
Makes a randomized validation image. Returns the image and list of atom locations.
"""
function make_validation_image(;size, num_atoms, noise_level=0.2, jiggle_atoms=0.1, width_range=(0.005, 0.2), noise_function=randn)
    model = ImageModel(size)

    total_atoms = num_atoms[1]*num_atoms[2]
    centroids = Matrix{Float32}(undef, 2, 0)
    atom_pitch_y = size[1] / (num_atoms[1]+1)
    atom_pitch_x = size[2] / (num_atoms[2]+1)


    jiggle_y = rand(total_atoms).*jiggle_atoms.*atom_pitch_y
    jiggle_x = rand(total_atoms).*jiggle_atoms.*atom_pitch_x
    jiggle_matrix = hcat(jiggle_y, jiggle_x)'

    
    for i in 1:total_atoms
        col_number = floor((i-1)/num_atoms[1]+1)
        row_number = ((i-1)-(col_number-1)*num_atoms[1]+1)
        atom_y = row_number * atom_pitch_y
        atom_x = col_number * atom_pitch_x
        centroids = [centroids Float32.([atom_y, atom_x])]
    end
    centroids = centroids .+ jiggle_matrix
    intensities = rand(total_atoms)./2 .+0.5
    widths = rand(total_atoms).*(width_range[2]-width_range[1]).+width_range[1]
    bs = zeros(total_atoms);

    reset_lattices!(model)
    add_lattice!(model, centroids', intensities, widths, bs, widths)
    initialize!(model)

    noise = abs.(noise_function(size...) .* noise_level)
    out_img = Gray.(stretch_image(produce_image(model) .+ noise))
    return (out_img, centroids)
end

"""
Makes a set of *num_images* validation images, which are saved to *folder*. 

"""
function produce_validation_images(folder, num_images)
    for i in 1:num_images
        num_atoms = (rand(5:60),rand(5:60))
        noise_level=rand()*2
        jiggle_atoms = rand()*0.2
        width_rand = rand()*0.01+0.01
        width_range = (width_rand, width_rand*rand(1:5))
        noise_function = rand([Random.randexp, randn])
        v_im,atom_pos = make_validation_image(
                            size=(1000,1000), 
                            num_atoms=num_atoms, 
                            noise_level=noise_level, 
                            jiggle_atoms=jiggle_atoms, 
                            width_range=width_range, 
                            noise_function=noise_function)
        
        save(folder*string(i)*".png", v_im)

        f = open(folder*string(i)*"_matadata.txt", "w")
        write(f, "num_atoms = "*string(num_atoms)*"\n")
        write(f, "noise_level = "*string(noise_level)*"\n")
        write(f, "jiggle_atoms = "*string(jiggle_atoms)*"\n")
        write(f, "width_range = "*string(width_range)*"\n")
        write(f, "noise_function = "*string(noise_function)*"\n")
        
        close(f)
        f = open(folder*string(i)*"_atoms.txt", "w")
        writedlm(f, atom_pos)
        close(f)
    end
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