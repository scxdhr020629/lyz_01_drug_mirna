import torch
from model.cnn_gcnmulti import GCNNetmuti
from torch_geometric.data import Data

import pandas as pd
import numpy as np
import os
import json
from rdkit import Chem
import networkx as nx

# 原子的特征提取
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

# 独热编码函数
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# 从SMILES字符串提取图结构
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)  # 将SMILES字符串转为分子对象
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")

    c_size = mol.GetNumAtoms()  # 获取原子数量

    # 提取节点特征
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)  # 获取原子特征
        features.append(feature / sum(feature))  # 特征归一化

    # 提取边的信息
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])  # 获取边的两个原子的索引

    # 构建图
    g = nx.Graph(edges)  # 创建无向图
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, np.array(features), edge_index  # 返回原子数量、特征和边的索引

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNNetmuti().to(device)
model.load_state_dict(torch.load("model_GCNNetmuti_run_5.pt",weights_only=True))  # 更改为你的模型名称
model.eval()

# # 准备输入功能
# def smile_to_graph(smile):
#     # 实现你的smile_to_graph函数，获取节点特征和边的信息
#     pass  # 请确保实现此函数

def prepare_input(smile):
    node_features, edge_index = smile_to_graph(smile)
    return Data(x=torch.tensor(node_features, dtype=torch.float).to(device),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device))

# 进行预测
smile = "FC1=CNC(=O)NC1=O"  # 示例SMILES字符串
input_data = prepare_input(smile)

# 预测结果
with torch.no_grad():
    output = model(input_data)
    predictions = (output >= 0.5).float()  # 将输出进行二分类处理

print("预测结果:", predictions.cpu().numpy())
