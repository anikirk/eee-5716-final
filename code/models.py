import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score

def evaluate_case1(tf_data, trojan_groups, sample_size, classifier_type='rf'):
    tf_count = sample_size // 2
    ti_count = sample_size // 2
    results, tprs, fprs, times = [], [], [], []
    for _ in range(20):
        tf_train = tf_data[np.random.choice(len(tf_data), tf_count, replace=False)]
        ti_train, ti_eval = [], []
        for j in range(23):
            samples = np.array(trojan_groups[j])
            selected = samples[np.random.choice(len(samples), ti_count, replace=False)]
            ti_train.extend(selected)
            ti_eval.extend([s for s in samples if s.tolist() not in selected.tolist()])
        X_train = np.vstack([tf_train, ti_train])
        y_train = np.array([0]*tf_count + [1]*len(ti_train))
        X_test = np.vstack([tf_data, ti_eval])
        y_test = np.array([0]*len(tf_data) + [1]*len(ti_eval))
        start = time.time()
        clf = RandomForestClassifier(n_estimators=100) if classifier_type == 'rf' else OneClassSVM(gamma='auto')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if classifier_type == 'ocsvm':
            y_pred = np.where(y_pred == 1, 0, 1)
        end = time.time()
        acc = accuracy_score(y_test, y_pred)
        TP = np.sum((y_test == 1) & (y_pred == 1))
        FN = np.sum((y_test == 1) & (y_pred == 0))
        FP = np.sum((y_test == 0) & (y_pred == 1))
        TN = np.sum((y_test == 0) & (y_pred == 0))
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        results.append(acc); tprs.append(tpr); fprs.append(fpr); times.append(end - start)
    return np.mean(results), np.mean(tprs), np.mean(fprs), np.mean(times)

def evaluate_case2(tf_data, trojan_groups, sample_size, classifier_type='ocsvm'):
    results, tprs, fprs, times = [], [], [], []
    for _ in range(20):
        tf_train = tf_data[np.random.choice(len(tf_data), sample_size, replace=False)]
        ti_eval = []
        for j in range(23):
            ti_eval.extend(trojan_groups[j])
        X_test = np.vstack([tf_data, ti_eval])
        y_test = np.array([0]*len(tf_data) + [1]*len(ti_eval))
        start = time.time()
        clf = RandomForestClassifier(n_estimators=100) if classifier_type == 'rf' else OneClassSVM(gamma='auto')
        clf.fit(tf_train, [0]*len(tf_train)) if classifier_type == 'rf' else clf.fit(tf_train)
        y_pred = clf.predict(X_test)
        if classifier_type == 'ocsvm':
            y_pred = np.where(y_pred == 1, 0, 1)
        end = time.time()
        acc = accuracy_score(y_test, y_pred)
        TP = np.sum((y_test == 1) & (y_pred == 1))
        FN = np.sum((y_test == 1) & (y_pred == 0))
        FP = np.sum((y_test == 0) & (y_pred == 1))
        TN = np.sum((y_test == 0) & (y_pred == 0))
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        results.append(acc); tprs.append(tpr); fprs.append(fpr); times.append(end - start)
    return np.mean(results), np.mean(tprs), np.mean(fprs), np.mean(times)