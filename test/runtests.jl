import STEMfit
using Test
using Images
using Random

const DATA_PATH = joinpath(@__DIR__, "data")
const IMAGE_PATH = joinpath(DATA_PATH, "raw")
const TEST_IMAGES = joinpath.(IMAGE_PATH, readdir(IMAGE_PATH))
const TIF_IMAGE = joinpath(@__DIR__, "data/raw/image.tif")
const IMAGE_DATA_16BIT = Gray{Float64}.(load(joinpath(IMAGE_PATH, "image.tif")))
const IMAGE_DATA_8BIT = Gray{Float64}.(load(joinpath(IMAGE_PATH, "image-8bit.tif")))
const IMAGE_PATH_DOWNSCALED = joinpath(@__DIR__, "data/downscaled")
const IMAGES_DOWNSCALED = joinpath.(IMAGE_PATH_DOWNSCALED, readdir(IMAGE_PATH_DOWNSCALED)[2:end])
const FILTERED_IMAGE = Gray{Float64}.(load(joinpath(@__DIR__, "data/filtered/filtered image.tif")))

@testset "image.jl"            begin; include("./image.jl/tests.jl");            end
@testset "background.jl"       begin; include("./background.jl/tests.jl");       end
@testset "dbscan.jl"           begin; include("./dbscan.jl/tests.jl");           end
@testset "findatoms.jl"        begin; include("./findatoms.jl/tests.jl");        end
@testset "gaussianfitting.jl"  begin; include("./gaussianfitting.jl/tests.jl");  end
@testset "imagemodel.jl"       begin; include("./imagemodel.jl/tests.jl");       end
@testset "latticeparameter.jl" begin; include("./latticeparameter.jl/tests.jl"); end
@testset "plotting.jl"         begin; include("./plotting.jl/tests.jl");         end

