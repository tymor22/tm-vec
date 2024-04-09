import numpy as np

import torch
from torch import nn
import re
import faiss
from tqdm import tqdm

#Function to extract ProtTrans embedding for a sequence
def featurize_prottrans(sequences, model, tokenizer, device):

    sequences = [(" ".join(sequences[i])) for i in range(len(sequences))]
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = embedding.last_hidden_state.cpu().numpy()

    features = []
    for seq_num in range(len(sequences)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    prottrans_embedding = torch.tensor(features[0])
    prottrans_embedding = torch.unsqueeze(prottrans_embedding, 0).to(device)

    return (prottrans_embedding)



#Embed a protein using tm_vec (takes as input a prottrans embedding)
def embed_tm_vec(prottrans_embedding, model_deep, device):
    padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(device)
    tm_vec_embedding = model_deep(prottrans_embedding, src_mask=None, src_key_padding_mask=padding)

    return(tm_vec_embedding.cpu().detach().numpy())


#Predict the TM-score for a pair of proteins (inputs are TM-Vec embeddings)
def cosine_similarity_tm(output_seq1, output_seq2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    dist_seq = cos(output_seq1, output_seq2)

    return(dist_seq)


def encode(sequences, model_deep, model, tokenizer, device):
    i = 0
    embed_all_sequences=[]
    for i in tqdm(range(len(sequences))):
        protrans_sequence = featurize_prottrans(sequences[i:i+1], model, tokenizer, device)
        embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device)
        embed_all_sequences.append(embedded_sequence)
    return np.concatenate(embed_all_sequences, axis=0)


def encode_gen(sequences, model_deep, model, tokenizer, device):
    i = 0
    embed_all_sequences=[]
    while i < len(sequences):
        protrans_sequence = featurize_prottrans(sequences[i:i+1], model, tokenizer, device)
        embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device)
        i = i + 1
        yield embedded_sequence


def save_embeds(names, embeds, file_name):
    with open(file_name, 'wb') as file:
        for name, seq_embed in enumerate(zip(names, embeds)):
            np.savez(file, **{str(name): seq_embed}, allow_pickle=False)


def load_database(path):
    lookup_database = np.load(path)
    #Build an indexed database
    d = lookup_database.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(lookup_database)
    index.add(lookup_database)

    return(index)


def query(index, queries, k=10):
    faiss.normalize_L2(queries)
    D, I = index.search(queries, k)

    return(D, I)


def _format_id(ix, iy):
    """ Assumes that len(ix) > len(iy) """
    diff = len(ix) - len(iy)
    ix = ix + ' '
    iy = iy + ' ' * (diff + 1)
    return ix, iy


def format_ids(ix, iy):
    if len(ix) > len(iy):
        ix, iy = _format_id(ix, iy)
    else:
        iy, ix = _format_id(iy, ix)
    return ix, iy
