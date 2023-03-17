#
#File: image.jl
#Author: Ewout van der Veer
#
#Description:
# Functions for dealing with images.
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
    filter_image(
        image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
        [,number_of_singular_vectors::Union{Integer, Symbol} = :auto,
        kernel_size::Integer = 1,
        gaussian_convolution_only::Bool = false,
        plot::Bool = false]
    ) 
    -> Matrix{<:Gray{<:AbstractFloat}}

Filter the image using Singular Value Decomposition and Gaussian convolution.

`number_of_singular_vectors` is the number of singular values to use. If set to :auto, 
it will be determined automatically. `kernel_size` is the size of the gaussian kernel 
that is used for the convolution. For low-resolution images, set `kernel_size = 1`. 
If the `gaussian_convolution_only` parameter is set to true, no SVD filtering is done
and only a gaussian convolution is applied to the image. Possibly useful for images with
no or little translational symmetry. If `plot` is set to true, a plot will be generated
of the power of each singular vector of the image for diagnostic purposes.
"""
function filter_image(
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}};
    number_of_singular_vectors::Union{Integer, Symbol} = :auto,
    kernel_size::Integer = 1,
    gaussian_convolution_only::Bool = false,
    plot::Bool = false,
    kwargs...
)
    if !gaussian_convolution_only
        svd_res = svd(image)
        U, Σ, Vᵀ = svd_res.U, Diagonal(svd_res.S), svd_res.Vt
        if number_of_singular_vectors == :auto
            log_Σ = log.(svd_res.S[1:100])
            A = [collect(1:100) ones(100)]
            sv_dist = log_Σ .- A*inv(A'*A)*A'*log_Σ
            number_of_singular_vectors = 
                round.(Int64, collect(1:100)[sv_dist .== minimum(sv_dist)]*1.2)[1]
        end
        filt_im = Gray.(
                        U[:,1:number_of_singular_vectors]*
                        Σ[1:number_of_singular_vectors, 1:number_of_singular_vectors]*
                        Vᵀ[1:number_of_singular_vectors,:]
                        )
        filtered_image = imfilter(filt_im, Kernel.gaussian(kernel_size))
    else
        filtered_image = imfilter(image, Kernel.gaussian(kernel_size))
    end

    if plot && !gaussian_convolution_only
        plot_singular_vectors(svd_res.S, number_of_singular_vectors; kwargs...)
    end

    filtered_image
end

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

"""
    stretch_image(image::AbstractMatrix)
    -> AbstractMatrix

Makes sure all values of image lie in the range (0,1)
"""
function stretch_image(image::AbstractMatrix)
    image_extrema = extrema(image)
    (image .- image_extrema[1]) ./ -(image_extrema)
end

"""
    residual_image(
        image_1::AbstractMatrix, 
        image_2::AbstractMatrix; 
        no_text::Bool = false
        )
    -> AbstractMatrix

Calculates the absolute difference between two images, returns the difference image
with all values in the range (0,1). If `no_text = false`, the total residual value 
and maximum single pixel deviation are printed to stdout.
"""
function residual_image(
    image_1::AbstractMatrix, 
    image_2::AbstractMatrix; 
    no_text::Bool = false
    )
    residual_image = abs.(Gray.(image_1) .- Gray.(image_2))
    if !no_text
        println("Maximum deviation: " * string(Float16(maximum(residual_image) * 100)) * " %")
        println("Total residual: " * string(Float16(residual(image_1, image_2))))
    end
    stretch_image(residual_image)
end

"""
    function enlarge_image(
        image::AbstractMatrix,
        enlargement_factor::Real
        ) 
    -> AbstractMatrix

Enlarges `image` by a factor of `factor`. Returns the enlarged image.
"""
function enlarge_image(
    image::AbstractMatrix{T},
    factor::Real
    ) where T
    image = parent(image) #In case an OffsetArray is passed
    imresize(image, round.(Int64, size(image) .* factor))
end

"""
    function enlarge_image(
        image::AbstractMatrix,
        factor::Integer
        ) 
    -> AbstractMatrix

Downscales `image` by a factor of `factor`. The new pixel values are the mean
of blocks of `factor` x `factor` pixels in the original image. Returns the smaller image.
"""
function downscale_image(
    image::AbstractMatrix{T}, 
    factor::Integer
    ) where T
    image = parent(image) #In case an OffsetArray is passed
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