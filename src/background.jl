#
#File: background.jl
#Author: Ewout van der Veer
#
#Description:
# Functions for finding the background contribution in an image.
#

"""
    construct_background(
        image::Matrix{<:Gray{<:AbstractFloat}},
        kernel::Tuple{<:Integer, <:Integer}
    ) 
    -> Matrix{Gray{Float64}}

Construct a background from *image* using a kernel of size *kernel*. 
The entries of *kernel* must be odd.
"""
function construct_background(
    image::Matrix{<:Gray{<:AbstractFloat}},
    kernel::Tuple{<:Integer, <:Integer}
)
    bck_img = mapwindow(minimum, image, kernel)
    gaussian_kernel = @. Int64(2*floor(kernel/4)+1)
    imfilter(bck_img, Kernel.gaussian(gaussian_kernel))
end

"""
    construct_background(
        image::Matrix{<:Gray{<:AbstractFloat}},
        uc::UnitCell;
        fix_edges::Symbol = :none
    ) 
    -> Matrix{Gray{Float64}}

Construct a background from *image* using a the unit cell *uc* to calculate the kernel size.

The optional *fix_edges* parameter shrinks the kernel in either the horizontal (*fix_edges* = :horizontal),
vertical (*fix_edges* = :vertical) or no (*fix_edges* = :none) direction to prevent artifacts at interfaces between layers.
"""
function construct_background(
    image::Matrix{<:Gray{<:AbstractFloat}},
    uc::UnitCell;
    fix_edges::Symbol = :none
)
    (minmap_kernel, gaussian_kernel) = get_kernels(uc, fix_edges)
    bck_img = mapwindow(minimum, image, minmap_kernel)
    imfilter(bck_img, Kernel.gaussian(gaussian_kernel))
end


"""
    get_kernels(
        uc::UnitCell, 
        fix_edges::Symbol
    )
    -> Tuple{}

Calculate kernels for image filtering from the unit cell *uc*. *fix_edges* determines if the kernels are shrunk. 
Returns a tuple of the mapping kernel and the gaussian convolution kernel.
"""
function get_kernels(
    uc::UnitCell, 
    fix_edges::Symbol
)
    uc_vertical = abs(uc.vector_1[1]) + abs(uc.vector_2[1])
    uc_horizontal = abs(uc.vector_1[2]) + abs(uc.vector_2[2])
    uc_transformed = (uc_vertical, uc_horizontal)

    minmap_kernel = @. Int64(2*floor(uc_transformed*0.75/2)+1) #Round to the nearest odd integer
    gaussian_kernel = @. Int64(2*floor(uc_transformed*0.75/4)+1)

    if fix_edges == :horizontal
        minmap_kernel = (Int64(2*floor(minmap_kernel[1]/4)+1), minmap_kernel[2])
    elseif fix_edges == :vertical
        minmap_kernel = (minmap_kernel[1], Int64(2*floor(minmap_kernel[2]/4)+1))
    end

    (minmap_kernel, gaussian_kernel)
end

"""
    get_intensity_above_background(
        positions::AbstractMatrix{<:Real},
        intensities::AbstractVector{<:Real},
        background::AbstractMatrix{<:Gray{<:Real}}
    )
    -> Real

Calculates the intensity of an atom above the given background
"""
function get_intensity_above_background(
    positions::AbstractMatrix{<:Real},
    intensities::AbstractVector{<:Real},
    background::AbstractMatrix{<:Gray{<:Real}}
)
    A = similar(intensities)
    for (n, (I, pos)) in enumerate(zip(intensities, eachcol(positions)))
        bck_intensity = background[round.(Int64, pos)...]
        A[n] = I - bck_intensity
    end
    A
end