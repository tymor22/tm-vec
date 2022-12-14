# Paper
TM-Vec: template modeling vectors for fast homology detection and alignment: https://www.biorxiv.org/content/10.1101/2022.07.25.501437v1

# Installation

First create a conda environment with python=3.9 installed.  If you are using cpu, use

`conda create -n tmvec faiss-cpu python=3.9 -c pytorch`

If you are using gpu use

`conda create -n tmvec faiss-gpu python=3.9 -c pytorch`

Once your conda enviroment is installed and activated (i.e. `conda activate tmvec`), then install tm-vec via
`pip install tm-vec`. If you are using a GPU, you may need to reinstall the gpu version of pytorch.  
See the [pytorch](https://pytorch.org/) webpage for more details.

# Models
Download the TM-vec model trained on SwissModel pairs:

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model.ckpt

Download the TM-vec config file for the model trained on SwissModel pairs: 

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model_params.json

Download the TM-vec model trained on CATH pairs:

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model.ckpt

Download the TM-vec config file for model trained on CATH pairs: 

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model_params.json
