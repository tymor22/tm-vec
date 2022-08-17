# Installation

First create a conda environment with python=3.9 installed.  If you are using cpu, use

`conda create -n tmvec faiss-cpu python=3.9 -c pytorch`

If you are using gpu use

`conda create -n tmvec faiss-gpu python=3.9 -c pytorch`

Once your conda enviroment is installed and activated (i.e. `conda activate tmvec`), then install tm-vec via
`pip install tm-vec`. If you are using a GPU, you may need to reinstall the gpu version of pytorch.  
See the [pytorch](https://pytorch.org/) webpage for more details.
