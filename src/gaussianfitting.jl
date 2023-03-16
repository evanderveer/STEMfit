#
#File: gaussianfitting.jl
#Author: Ewout van der Veer
#
#Description:
# Functions for fitting 2D gaussian functions to an image.
#

struct GaussianParameters{T,U,V,Y}
    index::Int32
    image_model::ImageModel{T,U,V,Y}
    subimage::AbstractMatrix{T}
    subbackground::AbstractMatrix{T}
    neighbor_image::AbstractMatrix{T}
    range::Tuple{UnitRange, UnitRange}
end



function get_valid_range(
    center::Tuple,
    image_size::Tuple,
    window_size::Real
)

    center_rounded = round.(Int32, center)
    window_size_rounded = round(Int32, window_size)
    ymin = clamp(center_rounded[1] - window_size_rounded, 1, image_size[1])
    ymax = clamp(center_rounded[1] + window_size_rounded, 1, image_size[1])
    xmin = clamp(center_rounded[2] - window_size_rounded, 1, image_size[2])
    xmax = clamp(center_rounded[2] + window_size_rounded, 1, image_size[2])
    
    if xmin == 1
        xrange = 1:(1+2*window_size)
    elseif xmax == image_size[2]
        xrange = (image_size[2]-2*window_size):image_size[2]
    else
        xrange = xmin:xmax
    end

    if ymin == 1
        yrange = 1:(1+2*window_size)
    elseif ymax == image_size[1]
        yrange = (image_size[1]-2*window_size):image_size[1]
    else
        yrange = ymin:ymax
    end
    
    (yrange, xrange)
end

function construct_parameter_set(
    image_model::ImageModel{T,U,V,Y},
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}
) where {T,U,V,Y}
    window_size = model_unit_cell_to_window_size(image_model)

    ranges = [get_valid_range(
                                (gaussian[1], gaussian[2]), 
                                size(image_model.background), 
                                window_size
                             )
                             for gaussian in image_model.gaussian_parameters]

    #Preallocate images for the calculation
    slice = Matrix{T}(undef, 2*window_size+1, 2*window_size+1)
    output_image = Matrix{T}(undef, 2*window_size+1, 2*window_size+1)

    gaussian_parameter_vector = [] 

    for (idx,rng) in enumerate(ranges)
        neighbor_image = produce_image(image_model, 
                                       range=rng, 
                                       exclude_index=Int32(idx),
                                       slice=slice[1:size.(rng)[1][1],1:size.(rng)[2][1]],
                                       output_image=output_image[1:size.(rng)[1][1],1:size.(rng)[2][1]])

        new_gp = GaussianParameters(
                Int32(idx),
                image_model,
                OffsetArray(Y.(image[rng...]), rng...),
                image_model.background[rng...],
                OffsetArray(neighbor_image, rng...),
                rng)

        push!(gaussian_parameter_vector, new_gp)
    end

    gaussian_parameter_vector
end

function neighbor_indices(
    image_model,
    range,
    index
)
    indices_full = unique!(sort!(vec(
        image_model.nearest_neighbor_indices_tensor[range..., :]
                    )))
    indices_full[indices_full .!= index]
end

function gaussian_to_optimization(
    gaussian_parameters::AbstractVector{T}
) where {T<:Real}
    
    (y0, x0, A, a, b, c) = gaussian_parameters
    covariance_matrix = [a b;b c]
    (a_t,b_t,c_t) = cholesky(Hermitian(covariance_matrix)).L[BitMatrix([1 0;1 1])]
    MVector{6, T}(y0, x0, A, a_t, b_t, c_t)
end

function optimization_to_gaussian(
    op::AbstractVector{T}
) where T<:Real
    
    if length(op) != 6
        throw(ArgumentError("parameters vector must have length 6"))
    end
    L_matrix = SMatrix{2, 2, T}([op[4] 0;op[5] op[6]])
    MVector{6, T}([op[1:3]; (L_matrix*L_matrix')[BitMatrix([1 1;0 1])]])
end


function loss_function(
    u::AbstractVector{Z},
    p::GaussianParameters{T,U,V,Y}
) where {T,U,V,Y,Z}
    u_g = optimization_to_gaussian(u)
    submodel = add_gaussian_to_image(Z.(p.neighbor_image), u_g)
    residual(submodel, p.subimage)
end

function fit!(
    image_model::ImageModel, 
    image::Union{AbstractMatrix{<:Gray{<:Real}}, AbstractMatrix{<:Real}}; 
    tolerance::Float64 = 0.001, 
    number_to_fit::Integer = typemax(Int),  
    A_limit::AbstractFloat = 0.0, 
    use_bounds::Bool = true,
    preconditioning::Bool = true
    )

    if preconditioning
        println("Starting preconditioning procedure")
        flush(stdout)
        precondition_atom_widths!(image_model, image)
    end

    optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())   
    gaussian_fit_levels = [floor(10*x[3])/10 for x in image_model.gaussian_parameters]

    println("Starting fitting procedure")
    flush(stdout)
    for A in 0.9:-0.1:A_limit
        println("Fitting: " * string(100-A*100) * "% \r")
        flush(stdout)

        gaussian_parameters = construct_parameter_set(image_model, image)
        initial_parameters = gaussian_to_optimization.(image_model.gaussian_parameters) 
        (lower_bounds, upper_bounds) = get_bounds(initial_parameters)
        gaussians_to_fit = gaussian_fit_levels .== A
        
        n = clamp(sum(gaussians_to_fit), 0:number_to_fit)

        fit_gaussians(initial_parameters[gaussians_to_fit], 
                      gaussian_parameters[gaussians_to_fit], 
                      optf, 
                      n,  
                      tolerance
                    )       

        invalid_gaussians = is_invalid_gaussian.(
            gaussian_to_optimization.(image_model.gaussian_parameters[gaussians_to_fit]),
            lower_bounds[gaussians_to_fit],
            upper_bounds[gaussians_to_fit]
            )
        if sum(invalid_gaussians) > 0 && use_bounds
            n = clamp(sum(invalid_gaussians), 0:number_to_fit)

            fit_gaussians(initial_parameters[gaussians_to_fit][invalid_gaussians], 
                gaussian_parameters[gaussians_to_fit][invalid_gaussians], 
                optf, 
                n, 
                lower_bounds[gaussians_to_fit][invalid_gaussians],  
                upper_bounds[gaussians_to_fit][invalid_gaussians],  
                tolerance
            )  

        end
        invalid_gaussians = is_invalid_gaussian.(
            gaussian_to_optimization.(image_model.gaussian_parameters[gaussians_to_fit]),
            lower_bounds[gaussians_to_fit],
            upper_bounds[gaussians_to_fit]
            )

    end
end



function fit_gaussians(us, ps, optf, n, x_tol)

    for i in 1:n
        prob = OptimizationProblem(optf, us[i], ps[i])

        res = solve(prob, BFGS(linesearch=BackTracking()), x_tol=x_tol)

        ps[i].image_model.gaussian_parameters[ps[i].index] = 
            optimization_to_gaussian(res)
    end
end

function fit_gaussians(us, ps, optf, n, lbs, ubs, x_tol)

    for i in 1:n
        prob = OptimizationProblem(optf, us[i], ps[i], lb=lbs[i], ub=ubs[i])

        res = solve(prob, BFGS(linesearch=BackTracking()), x_tol=x_tol)

        ps[i].image_model.gaussian_parameters[ps[i].index] = 
            optimization_to_gaussian(res)
    end
end

residual(im1::AbstractMatrix{<:Real}, im2::AbstractMatrix{<:Real}) = 
        sum((im1 .- im2) .^ 2)
residual(im1::AbstractMatrix{<:Gray{T}}, im2::AbstractMatrix{<:Real}) where 
        {T<:Real} = sum((T.(im1) .- im2) .^ 2)
residual(im1::AbstractMatrix{<:Real}, im2::AbstractMatrix{<:Gray{T}}) where 
        {T<:Real} = sum((im1 .- T.(im2)) .^ 2)
residual(im1::AbstractMatrix{<:Gray{T}}, im2::AbstractMatrix{<:Gray{V}}) where 
        {T<:Real, V<:Real} = sum((T.(im1) .- V.(im2)) .^ 2)


model_unit_cell_to_window_size(model::ImageModel) = 
    round(Int32, 0.75* maximum([norm(model.unit_cell.vector_1), 
                                norm(model.unit_cell.vector_2)]))


function add_gaussian_to_image(
    image::AbstractMatrix{T},
    parameters::AbstractVector{T}
    ) where T
    new_im = similar(image)
    for i in CartesianIndices(image)
        (y,x) = Tuple(i)
        @inbounds new_im[i] = 
            intensity(parameters, y, x)
    end
    new_im .+ image
end

function get_bounds(us)
    lbs = Vector{Vector{Float64}}(undef, length(us))
    ubs = Vector{Vector{Float64}}(undef, length(us))
    for (i,u) in enumerate(us)
        lb = Float64.([u[1]-3,u[2]-3,clamp(u[3]-0.15, 0, 1),u[4]*0.855,-Inf,u[6]*0.85])
        ub = Float64.([u[1]+3,u[2]+3,clamp(u[3]+0.15, 0, 1),u[4]*1.15,Inf,u[6]*1.15])
        
        lbs[i] = lb
        ubs[i] = ub
    end
    (lbs,ubs)
end

is_invalid_gaussian(u, lb, ub) = any(u .< lb) || any(u .> ub)

function precondition_atom_widths!(
    image_model,
    image
)

    #Use finite differences here so we don have to adapt the code to accept dual numbers
    optf = OptimizationFunction(optimize_width, Optimization.AutoFiniteDiff())
    prob = OptimizationProblem(optf, [1.0], (image=image, image_model=image_model))
    res = solve(prob, BFGS())[1]
    change_atom_widths!(res, image_model)
end

function change_atom_widths!(
    factor,
    image_model
)
    for gaussian in image_model.gaussian_parameters
        gaussian[4] *= factor
        gaussian[6] *= factor
    end
end

function optimize_width(
    factor,
    p
)
    change_atom_widths!(factor[1], p.image_model)
    res = residual(p.image, produce_image(p.image_model))
    change_atom_widths!(1/factor[1], p.image_model)
    res
end

function fitting_parameters(
    atom_parameters,
    background_image
)
    atom_positions = atom_parameters[1:2, :]
    atom_intensities = atom_parameters[3, :]
    atom_widths = atom_parameters[4, :]
    (a, b, c) = STEMfit.get_initial_gaussian_parameters(atom_widths)
    A = STEMfit.get_intensity_above_background(atom_positions, atom_intensities, background_image);
    [atom_positions; A';a';b';c'];
end