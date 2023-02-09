#!/bin/bash
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#SBATCH -C a100,ib
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=60:00:00
#SBATCH --tasks-per-node=1

source ~/ceph/venvs/deepblast/bin/activate

module -q purge
module load gcc python cuda cudnn


#wget https://users.flatironinstitute.org/thamamsy/public_www/embeddings_cath_s100_final.npy
#wget https://users.flatironinstitute.org/thamamsy/public_www/embeddings_cath_s100_w_metadata.tsv
#wget https://users.flatironinstitute.org/thamamsy/public_www/cath_large.npy
#wget https://users.flatironinstitute.org/thamamsy/public_www/cath_large_metadata.npy
#wget https://users.flatironinstitute.org/thamamsy/public_www/cath-domain-seqs-large.fa
#wget https://users.flatironinstitute.org/thamamsy/public_www/cath-domain-seqs-large.fai
#wget https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/checkpoints/deepblast-l8.ckpt


tm_vec_run.py \
    --query test.fa \
    --tm-vec-model tm_vec_cath_model.ckpt \
    --tm-vec-config tm_vec_cath_model_params.json \
    --database cath_large.npy \
    --metadata cath_large_metadata.npy \
    --database-fasta cath-domain-seqs-large.fa \
    --database-faidx cath-domain-seqs-large.fai \
    --deepblast-model deepblast-l8.ckpt \
    --device 'gpu' \
    --path_output_neigbhors neighbors.npy \
    --path_output_embeddings embeddings.npy \
    --path_output_alignments alignments.npy
