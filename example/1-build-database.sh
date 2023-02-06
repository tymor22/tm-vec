#wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model.ckpt
#wget https://users.flatironinstitute.org/thamamsy/public_www/tm_vec_cath_model_params.json


tm_encode.py \
    --input-fasta bagel.fa \
    --tm-vec-model tm_vec_cath_model.ckpt \
    --tm-vec-config-path tm_vec_cath_model_params.json \
    --output bagel_database
