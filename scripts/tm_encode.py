#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import featurize_prottrans, embed_tm_vec, encode
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from Bio import SeqIO
import faiss
from pathlib import Path
import argparse
import pickle


parser = argparse.ArgumentParser(description='Process TM-Vec arguments')

parser.add_argument("--input_data",
        type=Path,
        required=True,
        help="Input data"
)


parser.add_argument("--tm_vec_model_path",
        type=Path,
        required=True,
        help="Model path for TM-Vec embedding model"
)


parser.add_argument("--tm_vec_config_path",
        type=Path,
        required=True,
        help="Config path for TM-Vec embedding model"
)


parser.add_argument("--path_output_database",
        type=Path,
        required=True,
        help="Output path for the database"
)


parser.add_argument("--path_output_metadata",
        type=Path,
        help="Output path for the metadata"
)

#Load arguments
args = parser.parse_args()

#Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Load the ProtTrans model and ProtTrans tokenizer
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
gc.collect()
model = model.to(device)
model = model.eval()
print("ProtTrans model downloaded")


#Load the Tm_Vec_Align TM model
tm_vec_model_config = trans_basic_block_Config.from_json(args.tm_vec_config_path)
model_deep = trans_basic_block.load_from_checkpoint(args.tm_vec_model_path, config=tm_vec_model_config)
model_deep = model_deep.to(device)
model_deep = model_deep.eval()
print("TM-Vec model loaded")


#Read in query sequences
with open(args.input_data) as handle:
    headers = []
    seqs = []
    for record in SeqIO.parse(handle, "fasta"):
        headers.append(record.id)
        seqs.append(str(record.seq))

flat_seqs = [seqs[i] for i in range(len(seqs))]
print("Sequences inputed")

#Embed all query sequences
encoded_database = encode(flat_seqs, model_deep, model, tokenizer, device)

#Metadata array
metdata = np.array(headers)


#Outputting results

#Write out the embeddings
np.save(args.path_output_database, encoded_database)
#Write out metdata
np.save(args.path_output_metadata, metdata)

print("Done")
