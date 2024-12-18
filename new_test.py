import pandas as pd

# 读取 miRNA_sequences.xlsx 文件
rna = pd.read_excel('data/miRNA_sequences.xlsx')

# 创建 compound_iso_smiles 列，所有值设为固定的字符串 'FC1=CNC(=O)NC1=O'
raw_data = 'FC1=CNC(=O)NC1=O'
compound_iso_smiles = [raw_data] * len(rna)  # 创建一个和 Sequence 列一样长的列表

# 将 affinity 列全部设置为 0
affinity = [0] * len(rna)

# 创建 DataFrame，合并所有数据
final_df = pd.DataFrame({
    'compound_iso_smiles': compound_iso_smiles,
    'target_sequence': rna['Sequence'],
    'affinity': affinity
})

# 保存为 CSV 文件
output_file = 'data/processed/last/miRNA_affinity_data.csv'
final_df.to_csv(output_file, index=False)

print(f"CSV 文件已保存至: {output_file}")
