import pandas as pd
import matplotlib.pyplot as plt
from data_utils import load_chip_data, group_by_trojan_type
from models import evaluate_case1, evaluate_case2

def run_all(folder):
    X, y = load_chip_data(folder)
    tf_data, trojan_groups = group_by_trojan_type(X, y)
    sample_sizes = [6, 12, 24]
    classifiers = ['rf', 'ocsvm']
    results_table = []
    for size in sample_sizes:
        for clf in classifiers:
            acc1, tpr1, fpr1, time1 = evaluate_case1(tf_data, trojan_groups, size, clf)
            acc2, tpr2, fpr2, time2 = evaluate_case2(tf_data, trojan_groups, size, clf)
            results_table.append([size, clf.upper(), acc1, tpr1, fpr1, time1, acc2, tpr2, fpr2, time2])
    df = pd.DataFrame(results_table, columns=[
        "Sample Size","Classifier",
        "Case1_Acc","Case1_TPR","Case1_FPR","Case1_Time",
        "Case2_Acc","Case2_TPR","Case2_FPR","Case2_Time"
    ])
    print(df)
    return df

if __name__ == "__main__":
    folder = "dataset"
    results_df = run_all(folder)
    results_df.to_csv("comparative_results.csv", index=False)