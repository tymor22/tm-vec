# Paper
TM-Vec: template modeling vectors for fast homology detection and alignment: https://www.biorxiv.org/content/10.1101/2022.07.25.501437v1

[Embed sequences with TM-vec](https://colab.research.google.com/github/tymor22/tm-vec/blob/master/google_colabs/Embed_sequences_using_TM_Vec.ipynb)

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
```
wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model.ckpt

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model_params.json
```

Download the model weights/config of the large TM-vec model trained on SwissModel pairs (trained on protein chains up to 1000 residues long):
```
wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model_large.ckpt

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_swiss_model_large_params.json
```

Download the model weights/config of the large TM-vec model trained on CATH pairs (trained on CATH S100 domains sampled from ProtTucker training domains):

```
wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model_large.ckpt

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model_large_params.json
```

Download the model weights/config of the base TM-vec model trained on CATH pairs (trained on CATH S40):

```
wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model.ckpt

wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model_params.json
```

# Databases

We have embedded several sequence databases that users can search against. We have included embeddings for all CATH domains and SWISS-PROT sequences here. See the search tutorials or the scripts folder for how to run searches against these databases. Metadata for these sequences is position indexed. The embeddings and metadata are stored as numpy array (npy format) which can loaded as follows: np.load(file_path, allow_pickle=True).

Download the embeddings and metadata for CATH domains (the model that you should query with is tm_vec_cath_model_large)

```
wget https://users.flatironinstitute.org/thamamsy/public_www/cath_large.npy

wget https://users.flatironinstitute.org/thamamsy/public_www/cath_large_metadata.npy
```

Download the embeddings and metadata for SWISS-PROT chains (the model that you should query with here is tm_vec_swiss_model_large)

```
wget https://users.flatironinstitute.org/thamamsy/public_www/swiss_large.npy

wget https://users.flatironinstitute.org/thamamsy/public_www/swiss_large_metadata.npy
```

# Run TM-Vec + DeepBLAST from the command line

See the DeepBLAST wiki on how to [build TM-vec databases](https://github.com/flatironinstitute/deepblast/wiki/Building-the-TMvec-search-database) and search against [TM-vec databases](https://github.com/flatironinstitute/deepblast/wiki/Searching-proteins)

