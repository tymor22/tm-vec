
#wget https://users.flatironinstitute.org/thamamsy/public_www/embeddings_cath_s100_final.npy
#wget https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/checkpoints/deepblast-l8.ckpt

# usage: tm_vec_run.py [-h] --query QUERY --database DATABASE [--metadata METADATA] --tm-vec-model TM_VEC_MODEL --tm-vec-config TM_VEC_CONFIG [--deepblast-model DEEPBLAST_MODEL]
#                      [--protrans-model PROTRANS_MODEL] [--device DEVICE] [--k_nearest_neighbors K_NEAREST_NEIGHBORS] [--align ALIGN] [--database_sequences DATABASE_SEQUENCES]
#                      --path_output_neigbhors PATH_OUTPUT_NEIGBHORS [--path_output_embeddings PATH_OUTPUT_EMBEDDINGS] [--path_output_alignments PATH_OUTPUT_ALIGNMENTS]


tm_vec_run.py \
    --query bagel.fa \
    --tm-vec-model tm_vec_cath_model.ckpt \
    --tm-vec-config tm_vec_cath_model_params.json \
    --database embeddings_cath_s100_final.npy \
    --database-sequences ? \
    --deepblast-model deepblast-l8.ckpt \
    --device 'gpu' \
    --path_output_neigbhors neighbors.npy \
    --path_output_embeddings embeddings.npy \
    --path_output_alignments alignments.npy
