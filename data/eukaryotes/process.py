import os
import pickle
from collections import defaultdict

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm


def find_amino_acid(x):
    return ('B' in x) | ('O' in x) | ('J' in x) | ('U' in x) | ('X' in x) | ('Z' in x)

group_dict = defaultdict(list)
pair_path='cerevisiae.tsv'
seqs_path='cerevisiae.fa'
ID_list = []
seq_list = []
for fa in SeqIO.parse(seqs_path, 'fasta'):
    ID = fa.description
    seq = ''.join(list(fa.seq))
    ID_list.append(ID)
    seq_list.append(seq)
    # PPI_seq_dict[ID] = seq
df = pd.DataFrame({"Entry": ID_list, "seq": seq_list})
df = df[df["seq"].str.len() > 50]
df = df[df["seq"].str.len() <= 800]

protein_list = list(df["Entry"])

data = pd.read_csv(pair_path, sep=' ', header=None).values.tolist()
data = pd.DataFrame(data)
data.columns = ["protein1", "protein2", "score"]
data = data.drop(index=0)
data = data[data["protein1"].isin(protein_list)]
data = data[data["protein2"].isin(protein_list)]
data = data.reset_index(drop=True)
label_list = []
for i in range(len(data)):
    if int(data["score"][i]) > 500:
        label_list.append(1)
    else:
        label_list.append(0)
data["label"] = label_list
pos = data[data["label"] != 0]
pos = pos.reset_index(drop=True)
pos = pos[:10000]
neg = data[data["label"] != 1]
neg = neg.reset_index(drop=True)
# neg = neg[:50000]
POS_protein_list = list(set(pos['protein1'].unique()).union(set(pos['protein2'].unique())))
neg_ppi_list = []
for i in tqdm(range(len(neg))):
    if len(neg_ppi_list) >= 90000:
        break
    if neg["protein1"][i] in POS_protein_list and neg["protein2"][i] in POS_protein_list:
        neg_ppi_list.append([neg["protein1"][i], neg["protein2"][i]])

neg_network = pd.DataFrame(neg_ppi_list, columns=["protein1", "protein2"])
os.makedirs("./cerevisiae", exist_ok=True)
pos.to_csv("./cerevisiae/pos_network_cerevisiae.tsv")
neg_network.to_csv("./cerevisiae/neg_network_cerevisiae.tsv")
f_save = open("./cerevisiae/Protein_list_cerevisiae.pkl", "wb")
pickle.dump(POS_protein_list, f_save)
f_save.close()
print("1")


pos = data[data["label"] != 0]
pos = pos.reset_index(drop=True)
pos = pos[20000:30000]
neg = data[data["label"] != 1]
neg = neg.reset_index(drop=True)
# neg = neg[:50000]
POS_protein_list = list(set(pos['protein1'].unique()).union(set(pos['protein2'].unique())))
neg_ppi_list = []
for i in tqdm(range(len(neg))):
    if len(neg_ppi_list) >= 90000:
        break
    if neg["protein1"][i] in POS_protein_list and neg["protein2"][i] in POS_protein_list:
        neg_ppi_list.append([neg["protein1"][i], neg["protein2"][i]])

neg_network = pd.DataFrame(neg_ppi_list, columns=["protein1", "protein2"])
os.makedirs("./cerevisiae_1", exist_ok=True)
pos.to_csv("./cerevisiae_1/pos_network_cerevisiae_1.tsv")
neg_network.to_csv("./cerevisiae_1/neg_network_cerevisiae_1.tsv")
f_save = open("./cerevisiae_1/Protein_list_cerevisiae_1.pkl", "wb")
pickle.dump(POS_protein_list, f_save)
f_save.close()
print("1")