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

python /mnt/home/jmorton/ceph/research/gert/deep_blast_training/collect_env_details.py


#wget https://users.flatironinstitute.org/thamamsy/public_www/embeddings_cath_s100_final.npy
#wget https://users.flatironinstitute.org/thamamsy/public_www/embeddings_cath_s100_w_metadata.tsv
#wget https://users.flatironinstitute.org/thamamsy/public_www/cath_large.npy
#wget https://users.flatironinstitute.org/thamamsy/public_www/cath_large_metadata.npy
#wget https://users.flatironinstitute.org/thamamsy/public_www/cath-domain-seqs-large.fa
#wget https://users.flatironinstitute.org/thamamsy/public_www/cath-domain-seqs-large.fai
#wget https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/checkpoints/deepblast-l8.ckpt


tm_vec_run.py \
    --query bagel.fa \
    --tm-vec-model tm_vec_cath_model.ckpt \
    --tm-vec-config tm_vec_cath_model_params.json \
    --database cath_large.npy \
    --metadata cath_large_metadata.npy \
    --database-fasta cath-domain-seqs-large.fa \
    --database-faidx cath-domain-seqs-large.fai \
    --protrans-model ~/ceph/prot_t5_xl_uniref50 \
    --deepblast-model deepblast-l8.ckpt \
    --device 'gpu' \
    --path_output_neigbhors neighbors.npy \
    --path_output_embeddings embeddings.npy \
    --path_output_alignments alignments.npy
