# STEMfit

`STEMfit` is a package written in Julia for the analysis and fitting of (scanning) transmission electron microscopy (STEM) images.
`STEMfit` was made to handle large images (as many as 10 000's of atoms) of moderately defective materials (i.e. with significant strain, twin/grain boundaries, etc.) with high performance and minimal user input. 

The `examples` folder contains [Jupyter Notebooks](https://jupyter.org/) demonstrating the use of `STEMfit`.

Note that `STEMfit` is in an early stage of development. Features may be added, removed or modified without notice.

## How to install

Download the [Julia language](https://julialang.org/) or install the Julia extension in [Visual Studio Code](https://code.visualstudio.com/). To use notebooks like the example notebook, also download and install [Jupyter](https://jupyter.org/) or the Jupyter extension to VSCode. Then add `STEMfit` to your environment using

```
] add https://github.com/evanderveer/STEMfit/#main
```

or 

```
using Pkg; Pkg.add("https://github.com/evanderveer/STEMfit/")
```
