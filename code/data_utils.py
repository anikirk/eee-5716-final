import os
import numpy as np
import pandas as pd

def load_chip_data(folder_path):
    all_data, labels = [], []
    for i in range(1, 34):  # Chip1 to Chip33
        file_path = os.path.join(folder_path, f"Chip{i}.xlsx")
        df = pd.read_excel(file_path, header=None)
        tf_rows = df.iloc[[0, 24]].values   # Trojan-Free rows
        ti_rows = df.iloc[1:24].values      # Trojan-Inserted rows
        all_data.extend(tf_rows)
        labels.extend([0] * len(tf_rows))
        all_data.extend(ti_rows)
        labels.extend([1] * len(ti_rows))
    return np.array(all_data), np.array(labels)

def group_by_trojan_type(X, y):
    trojan_groups = {i: [] for i in range(23)}  # 23 Trojan types
    tf_data = []
    for i in range(0, len(X), 25):  # Each chip has 25 rows
        tf_data.append(X[i])        # Row 0 = TF
        tf_data.append(X[i+24])     # Row 24 = TF
        for j in range(23):         # Rows 1–23 = TI
            trojan_groups[j].append(X[i+1+j])
    return np.array(tf_data), trojan_groups