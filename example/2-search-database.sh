#!/bin/bash
# -C a100,ib
# -p gpu
# --gpus=1
#SBATCH --time=60:00:00
#SBATCH --tasks-per-node=1

source ~/ceph/venv/deepblast/bin/activate

module -q purge
module load gcc python cuda cudnn

export CUDA_HOME=$CUDA_BASE


wget https://users.flatironinstitute.org/thamamsy/public_www/embeddings_cath_s100_final.npy
wget https://users.flatironinstitute.org/thamamsy/public_www/embeddings_cath_s100_w_metadata.tsv
wget https://users.flatironinstitute.org/thamamsy/public_www/cath_large.npy
wget https://users.flatironinstitute.org/thamamsy/public_www/cath_large_metadata.npy
wget https://users.flatironinstitute.org/thamamsy/public_www/cath-domain-seqs-large.fa
wget https://users.flatironinstitute.org/thamamsy/public_www/cath-domain-seqs-large.fai
wget https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/checkpoints/deepblast-pt-l8.ckpt


# output alignment format
tmvec-search \
    --query bagel.fa \
    --tm-vec-model tm_vec_cath_model.ckpt \
    --tm-vec-config tm_vec_cath_model_params.json \
    --database bagel_database/db.npy \
    --metadata bagel_database/meta.npy \
    --database-fasta bagel.fa \
    --database-faidx bagel.fai \
    --deepblast-model deepblast-pt-l8.ckpt \
    --device 'gpu' \
    --k-nearest-neighbors 1 \
    --output-format alignment \
    --output alignments.txt \
    --output-embeddings test.npy

# output tabular format
tmvec-search \
    --query bagel.fa \
    --tm-vec-model tm_vec_cath_model.ckpt \
    --tm-vec-config tm_vec_cath_model_params.json \
    --database bagel_database/db.npy \
    --metadata bagel_database/meta.npy \
    --database-fasta bagel.fa \
    --database-faidx bagel.fai \
    --device 'gpu' \
    --output-format tabular \
    --output tabular.txt \
    --output-embeddings test.npy
