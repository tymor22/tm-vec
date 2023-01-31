# Paper
TM-Vec: template modeling vectors for fast homology detection and alignment: https://www.biorxiv.org/content/10.1101/2022.07.25.501437v1

[Embed sequences with TM-vec](https://colab.research.google.com/github/tymor22/tm-vec/blob/master/Embed_sequences_using_TM_Vec.ipynb)

# Installation

First create a conda environment with python=3.9 installed.  If you are using cpu, use

`conda create -n tmvec faiss-cpu python=3.9 -c pytorch`

If you are using gpu use

`conda create -n tmvec faiss-gpu python=3.9 -c pytorch`

Once your conda enviroment is installed and activated (i.e. `conda activate tmvec`), then install tm-vec via
`pip install tm-vec`. If you are using a GPU, you may need to reinstall the gpu version of pytorch.  
See the [pytorch](https://pytorch.org/) webpage for more details.

# Models
Download the model weights/config of the base TM-vec model trained on SwissModel pairs (trained on protein chains up to 300 residues long, works best on shorter sequences):

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model.ckpt

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model_params.json

Download the model weights/config of the large TM-vec model trained on SwissModel pairs (trained on protein chains up to 1000 residues long):

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model_large.ckpt

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model_large_params.json

Download the model weights/config of the large TM-vec model trained on CATH pairs (trained on CATH S100 domains sampled from ProtTucker training domains):

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model_large.ckpt

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model_large_params.json


Download the model weights/config of the base TM-vec model trained on CATH pairs (trained on CATH S40):

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model.ckpt

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model_params.json



# Run TM-Vec + DeepBLAST from the command line

Arguments for running TM-Vec search + DeepBLAST alignments from the command line. Python script is located here: scripts/tm_vec_run.py.

- There are several parameters that the user must provide, including the data to query (in Fasta format), the number of nearest neighbors to return (I️.e. Top N), and the option to perform alignments using DeepBLAST (True or False).  Additionally, the user should provide output paths for where output files should go. These include output paths for the nearest neighbor outputs, the embeddings for the queried sequences, and for alignment outputs (if alignments were done). Outputs (nearest neighbors, embeddings, alignments) will be written in numpy files (npy). 
- There are several database parameters the user needs to provide at the command line, including the lookup database that will be queried (TM-Vec embeddings database), the lookup database's metadata, and the lookup database's sequences (relevant for alignments). 
- There are several model parameters to provide at the command line. These include the weights for the TM-Vec model that will embed the user’s query sequences (note that this model should be the same model as the model used to make the lookup embedding database- i.e. TM-Vec CATH model or TM-Vec SWISS-MODEL model.), the config file for the TM-Vec model, and the DeepBLAST alignment model that will run alignments. 


Example run:
scripts/tm_vec_run.py --input_data --k_nearest_neighbors --align --path_output_neigbhors --path_output_embeddings --path_output_alignments --database --metadata --database_sequences --tm_vec_model_path --tm_vec_config_path --tm_vec_align_path


User parameters:

—input_data
“Fasta file for query proteins”

—k_nearest_neighbors
“Number of nearest neighbors to return (default is 5 per query)”

—align
“Option to return alignments (boolean)”

Output paths:

—path_output_neigbhors

“Nearest neighbor outputs”

—path_output_embeddings
“Embeddings for queried sequences”

—path_output_alignments
“Alignment output file if alignment is true”

Database parameters:

—database
“Lookup database that will be queried. These are the embeddings.” 

—metadata
“Metadata file for the lookup database”

—database_sequences
"Lookup database sequences."

TM-Vec model:

—tm_vec_model_path
“TM-Vec model path (weights)”

—tm_vec_config_path
“TM-Vec model config path”

DeepBLAST model:

—tm_vec_align_path
“Align model path”





