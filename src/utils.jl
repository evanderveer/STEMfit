#
#File: utils.jl
#Author: Ewout van der Veer
#
#Description:
# Miscellaneous utility functions.
#

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
                    convert::Bool = false,
                    downscale_factor::Union{<:Integer, Nothing} = nothing
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
    if downscale_factor === nothing
        return Gray{Float64}.(img)
    end
    downscale_image(Gray{Float64}.(img), downscale_factor)
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

function show_images(images...; rows=1, enlarge=1, zoom=true, kwargs...) 
    image_sizes = size.(images)
    if !any(s1 != s2 for (s1,s2) in image_sizes) && zoom == true
        image_sizes = [s[1] for s in image_sizes]
        enlargement_factors = enlarge*maximum(image_sizes)./image_sizes
        zoomed_images = [enlarge_image(im, ef) for (im,ef) in zip(images, enlargement_factors)]
        return mosaicview(zoomed_images, nrow=rows; kwargs...)
    end
    mosaicview(enlarge_image.(images, Int64(enlarge)), nrow=rows; kwargs...)
end

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
    enlargement_factor::Real
    ) where T
    image = parent(image)
    imresize(image, round.(Int64, size(image).*enlargement_factor))
end

function downscale_image(image::AbstractMatrix{T}, factor::Integer) where T
    image = parent(image)
    new_size = floor.(Int64, size(image) ./ factor)
    new_image = Matrix{T}(undef, new_size...)
    for i in CartesianIndices(new_image)
        new_i = (Tuple(i) .- (1,1)) .* factor .+ (1,1)
        new_image[i] = mean(image[new_i[1]:(new_i[1]+factor-1), new_i[2]:(new_i[2]+factor-1)])
    end
    new_image
end


#Make sure matrices of dual numbers can be displayed as images
Gray(dual::Dual) = Gray(dual.value)
Float64(dual::Dual) = Float64(dual.value)
Float16(dual::Dual) = Float16(dual.value)
Dual(pixel::Gray{T}) where T = Dual(T(pixel))
#ForwardDiff.Dual(pixel::Gray{T}) where T = Dual(T(pixel))