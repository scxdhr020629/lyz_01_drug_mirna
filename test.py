# 进行预测代码编写
# 先假设输入的字符串 s 然后传换成三列csv 再转换成四列csv 然后进行model.eval 模式 进行预测 然后保存数据结果

# 这里的raw_data 之后从前端传进来
raw_data = 'FC1=CNC(=O)NC1=O'

# 一些processdata的函数代码

import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

import logging
import sys


import torch.nn as nn
from jinja2.lexer import TOKEN_DOT
from torch.utils.data import DataLoader, WeightedRandomSampler
from model.cnn_gcnmulti import GCNNetmuti
# from  model.cnn_gcn import GCNNet
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc




# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# 第一部分 对传入的数据进行处理
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def process_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return None
    except:
        return None

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        # print(line[0][0])
        # print(i,"========", ch)
        X[i] = smi_ch_ind[ch]
    return X

# 这里的seq_dict 在后面
def seq_cat(prot,max_seq_len):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, ">": 65, "<": 66}

CHARISOSMILEN = 66

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 下面这部分代码是 drug 和 mirna 的读取

# 读取整个 Excel 文件
drugs = pd.read_excel('data/drug_id_smiles.xlsx')
rna = pd.read_excel('data/miRNA_sequences.xlsx')




# 这里先处理一下 用process_smiles
my_process_smile = process_smiles(raw_data)

compound_iso_smiles = [my_process_smile] * len(rna)  # 创建一个和 Sequence 列一样长的列表


# 将 affinity 列全部设置为 0
affinity = [0] * len(rna)

# 创建 DataFrame，合并所有数据
final_df = pd.DataFrame({
    'compound_iso_smiles': compound_iso_smiles,
    'target_sequence': rna['Sequence'],
    'affinity': affinity
})

# 保存为 CSV 文件
output_file = 'data/processed/last/_mytest1.csv'
final_df.to_csv(output_file, index=False)

print(f"CSV 文件已保存至: {output_file}")
seq_voc = "ACGU"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)



#药物图处理过程
opts = ['mytest']
for i in range(1, 2):
    compound_iso_smiles = []
    for opt in opts:
        df = pd.read_csv('data/processed/last/' + '_' + opt +str(i)+ '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    # # convert to PyTorch data format
    df = pd.read_csv('data/processed/last/'+ '_mytest'+str(i)+'.csv')
    test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(df['affinity'])
    XT = [seq_cat(t,24) for t in test_prots]
    test_sdrugs=[label_smiles(t,CHARISOSMISET,100) for t in test_drugs]
    test_drugs, test_prots, test_Y,test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y),np.asarray(test_sdrugs)
    test_data = TestbedDataset(root='data', dataset='last/'+'_mytest' + str(i), xd=test_drugs, xt=test_prots, y=test_Y,
                               z=test_seqdrugs,
                               smile_graph=smile_graph)

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# 第二部分 进行预测
# ----------------------------------------------------------------------------------------------------------------------------------------------------


# 删去 rog auc

def predicting(model, device, loader):
    model.eval()
    total_probs = []
    sample_indices = []
    total_labels = []

    logging.info('Making predictions for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            data = data.to(device)
            output = model(data)
            probs = output.cpu().numpy()
            indices = np.arange(len(probs)) + batch_idx * loader.batch_size

            total_probs.extend(probs)
            sample_indices.extend(indices)
            total_labels.extend(data.y.view(-1, 1).cpu().numpy())

    total_probs = np.array(total_probs).flatten()
    sample_indices = np.array(sample_indices).flatten()
    total_labels = np.array(total_labels).flatten()

    # accuracy = accuracy_score(total_labels, (total_probs >= 0.5).astype(int))
    # precision = precision_score(total_labels, (total_probs >= 0.5).astype(int), zero_division=1)
    # recall = recall_score(total_labels, (total_probs >= 0.5).astype(int))
    # f1 = f1_score(total_labels, (total_probs >= 0.5).astype(int))

    # logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    #
    # roc_auc = roc_auc_score(total_labels, total_probs)
    # precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    # sorted_indices = np.argsort(recall_vals)
    # recall_vals = recall_vals[sorted_indices]
    # precision_vals = precision_vals[sorted_indices]
    # pr_auc = auc(recall_vals, precision_vals)
    #
    # logging.info(f"ROC AUC: {roc_auc:.4f}")

    # return total_probs, sample_indices, accuracy, precision, recall, f1, pr_auc, total_labels
    # return total_probs, sample_indices, accuracy, precision, recall, f1, pr_auc
    return total_probs, sample_indices


# 先随便写的
def save_top_30_predictions(probs, indices, file_name='top_30_predictions_01.csv'):
    # Sort by probability (in descending order)
    sorted_indices = np.argsort(probs)[::-1]  # sort in descending order
    sorted_probs = probs[sorted_indices]
    # sorted_labels = labels[sorted_indices]
    sorted_indices = indices[sorted_indices]

    # Create a DataFrame to save the top 30 predictions
    top_30_df = pd.DataFrame({
        'Index': sorted_indices[:30],
        'Probability': sorted_probs[:30],
        # 'True_Label': sorted_labels[:30]
    })

    # Save to CSV
    top_30_df.to_csv(file_name, index=False)
    logging.info(f'Top 30 predictions saved to {file_name}')










modeling =GCNNetmuti

model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[1]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 0.0005
LOG_INTERVAL = 160
NUM_EPOCHS =100

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets

print('\nrunning on ', model_st + '_')


log_filename = f'training_1.log'

logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)



processed_data_file_test = 'data/processed/last/' + '_mytest1'+'.pt'
if ( (not os.path.isfile(processed_data_file_test))):
    print('please run process_data_old.py to prepare data in pytorch format!')
else:
    # train_data = TestbedDataset(root='data', dataset='_train1')
    # test_data = TestbedDataset(root='data', dataset='_test1')

    test_data = TestbedDataset(root='data', dataset='last/_mytest1')

    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE,shuffle=False,drop_last=True)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    model_path = "model_GCNNetmuti_.model"
    model.load_state_dict(torch.load(model_path))




    loss_fn = nn.BCELoss()  # for classification
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # best_acc= 0
    # best_epoch = -1

    result_file_name = 'result_' + model_st + '_' + '.csv'


    probs, indices = predicting(model, device, test_loader)
    save_top_30_predictions(probs, indices)


