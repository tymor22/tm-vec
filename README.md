# Paper
TM-Vec: template modeling vectors for fast homology detection and alignment: https://www.biorxiv.org/content/10.1101/2022.07.25.501437v1

[Embed sequences with TM-vec](https://colab.research.google.com/github/tymor22/tm-vec/blob/master/google_colabs/Embed_sequences_using_TM_Vec.ipynb)

## Notice
This fork of TM-vec is undergoing limited maintainence in the foreseeable future.
See the following fork for continue maintainence / developments : https://github.com/valentynbez/tmvec

# Installation

First create a conda environment with python=3.9 installed.  If you are using cpu, use

`conda create -n tmvec faiss-cpu python=3.9 -c pytorch`

If the installation fails, you may need to install mkl via `conda install mkl=2021 mkl_fft `

If you are using gpu use

`conda create -n tmvec faiss-gpu python=3.9 -c pytorch`

Once your conda enviroment is installed and activated (i.e. `conda activate tmvec`), then install tm-vec via
`pip install tm-vec`. If you are using a GPU, you may need to reinstall the gpu version of pytorch.
See the [pytorch](https://pytorch.org/) webpage for more details.

# Models
It is recommended to first download the `Prot-T5-XL-UniRef50` model weights.  This can be done as follows.```
```
mkdir Rostlab && cd "$_"
wget https://zenodo.org/record/4644188/files/prot_t5_xl_uniref50.zip
unzip prot_t5_xl_uniref50.zip
cd ..
```
There are 4 different TM-vec models that are available
- `tmvec_swiss_model` : the base TM-vec model trained on SwissModel pairs (trained on protein chains up to 300 residues long, works best on shorter sequences)
- `tmvec_swiss_model_large` : the large TM-vec model trained on SwissModel pairs (trained on protein chains up to 1000 residues long)
- `tm_vec_cath_model_large` : the large TM-vec model trained on CATH pairs (trained on CATH S100 domains sampled from ProtTucker training domains)
- `tm_vec_cath_model` : the base TM-vec model trained on CATH pairs (trained on CATH S40)

All of these TMvec models are available on Figshare : https://figshare.com/s/e414d6a52fd471d86d69

# Databases

We have embedded several sequence databases that users can search against. We have included embeddings for all CATH domains and SWISS-PROT sequences here. See the search tutorials or the scripts folder for how to run searches against these databases. Metadata for these sequences is position indexed. The embeddings and metadata are stored as numpy array (npy format) which can loaded as follows: np.load(file_path, allow_pickle=True).

There are two databases
- `tm_vec_cath_model_large` : CATH domains (the model that you should query with is tm_vec_cath_model_large). 
- `swiss_large` : SWISS-PROT chains (the model that you should query with here is tm_vec_swiss_model_large)

Each of these databases has corresponding metadata to link the sequences to the embeddings.

Both of these databases can be found on Zenodo : https://zenodo.org/records/11199459

# Run TM-Vec + DeepBLAST from the command line

See the DeepBLAST wiki on how to [build TM-vec databases](https://github.com/flatironinstitute/deepblast/wiki/Building-the-TMvec-search-database) and search against [TM-vec databases](https://github.com/flatironinstitute/deepblast/wiki/Searching-proteins)

