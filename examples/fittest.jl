using Pkg; Pkg.activate("C:\\Users\\ewout\\Documents\\venv\\Julia-STEMfit");
import STEMfit
using BenchmarkTools
using DelimitedFiles

background_image = STEMfit.load_image("background_image.png")
filtered_image = STEMfit.load_image("filtered_image.png")
image = STEMfit.load_image("image.tif", convert=true)
atom_data = readdlm("atomic_positions.csv", ',')[2:end, :]'
atom_positions = Float64.(atom_data[1:2,:])
atom_widths = Float64.(atom_data[3,:])
atom_intensities = Float64.(atom_data[4,:])

unit_cell = STEMfit.UnitCell((198,89,([-14.062311, 0.0013809234],[-0.21224983, -14.094682])))

atom_tree = STEMfit.KDTree(atom_positions)

(a, b, c) = STEMfit.get_initial_gaussian_parameters(atom_widths./2)
A = STEMfit.get_intensity_above_background(atom_positions, atom_intensities, background_image);

image_model = STEMfit.ImageModel(size(filtered_image), unit_cell, atom_tree, atom_positions, A, a, b, c)
STEMfit.set_background!(image_model, background_image);

println("Starting fitting procedure")
flush(stdout)
@time STEMfit.fit_optim!(image, image_model, length(image_model.gaussians))

final_sim_image = STEMfit.produce_image(image_model)
final_res_im = STEMfit.residual_image(image, final_sim_image)
STEMfit.save( "final_res_im.png",final_res_im)
STEMfit.save("final_sim_im.png",final_sim_image)