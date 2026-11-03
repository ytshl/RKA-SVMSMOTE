# -*- coding: utf-8 -*-
"""
RKA + SVMSMOTE 基准测试（70/30 分层划分 × 20 次重复 · v5）
=============================================================
v5 相对 v4 的唯一改动：RKA 换成【正确版】rka_transform_v3（严格按论文 Section 3.1）：
  - 权重函数 w(x) = eps + 1 - (a/(a+||x||^2))^b（正则化不完全 Beta）
  - Matérn 5/2 核含长度尺度 ell
  - 度量放大因子 q(x) = (det(g)/D_tr)^(eta/(2d))（逐样本、位置相关）
  - 新增调参维度：a / b / ell / eta
其余协议与 v4 完全一致：70/30 分层 × 20 次重复，仅 SVM / DT，逐指标最优阈值，
SVMSMOTE sampling_strategy 0.1~2.0，150 次 Optuna 调参（TPESampler seed=42），
标签域 {-1,+1}（少数类=1），结果存档 Table_Results_{SVM,DT}_7030x20_v5.csv。
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from rka_transform_v3 import RiemannianKernelAmplifierV3
from imblearn.over_sampling import SVMSMOTE

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_RUNS = 20
TEST_SIZE = 0.3
N_TRIALS = 150
METRIC_WEIGHTS = {'auc': 0.10, 'f1': 0.30, 'g_mean': 0.30, 'acc': 0.30}
THRESH_GRID = np.arange(0.01, 1.00, 0.01)


def load_keel_dat(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('@'):
                continue
            data.append(line.split(','))
    df = pd.DataFrame(data)
    X_df = df.iloc[:, :-1]
    for col in X_df.columns:
        try:
            X_df[col] = X_df[col].astype(float)
        except ValueError:
            X_df[col] = LabelEncoder().fit_transform(X_df[col])
    X = X_df.values
    y_raw = df.iloc[:, -1].values
    y = LabelEncoder().fit_transform(y_raw)
    minority_class = np.argmin(np.bincount(y))
    y = np.where(y == minority_class, 1, -1)
    return X, y


def pick_threshold(prob, y, metric):
    best_thr, best_score = 0.5, -np.inf
    for thr in THRESH_GRID:
        y_pred = np.where(prob >= thr, 1, -1)
        if metric == 'f1':
            s = f1_score(y, y_pred, pos_label=1)
        elif metric == 'g_mean':
            s = geometric_mean_score(y, y_pred, pos_label=1)
        elif metric == 'acc':
            s = accuracy_score(y, y_pred)
        if s > best_score:
            best_score, best_thr = s, thr
    return best_thr


def suggest_classifier_params(clf_name, trial):
    if clf_name == 'SVM':
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 1e1, log=True)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        return {'C': C, 'gamma': gamma, 'kernel': kernel, 'class_weight': class_weight}
    elif clf_name == 'DT':
        max_depth = trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        return {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
                'min_samples_split': min_samples_split, 'criterion': criterion,
                'class_weight': class_weight}
    raise ValueError(clf_name)


def make_classifier(clf_name, params):
    if clf_name == 'SVM':
        return SVC(probability=True, random_state=42, C=params['C'], gamma=params['gamma'],
                   kernel=params['kernel'], class_weight=params['class_weight'])
    return DecisionTreeClassifier(random_state=42, max_depth=params['max_depth'],
                                  min_samples_leaf=params['min_samples_leaf'],
                                  min_samples_split=params['min_samples_split'],
                                  criterion=params['criterion'], class_weight=params['class_weight'])


def suggest_rka_params(trial, max_dim):
    a = trial.suggest_float('a', 0.01, 100.0, log=True)
    b = trial.suggest_float('b', 0.1, 10.0, log=True)
    ell = trial.suggest_float('ell', 0.05, 20.0, log=True)
    eta = trial.suggest_float('eta', 0.0, 2.0)
    n_comp = trial.suggest_int('n_components', 2, max(3, max_dim))
    return {'a': a, 'b': b, 'ell': ell, 'eta': eta, 'n_components': n_comp}


def optimize_for_classifier(X, y, clf_name):
    skf_opt = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    max_dim = X.shape[1]

    def objective(trial):
        rka_p = suggest_rka_params(trial, max_dim)
        k_neighbors = trial.suggest_int('k_neighbors', 3, 10)
        sampling_strategy = trial.suggest_float('sampling_strategy', 0.1, 2.0)
        clf_params = suggest_classifier_params(clf_name, trial)

        scores = []
        for tr_idx, val_idx in skf_opt.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            if sum(y_tr == 1) <= k_neighbors:
                return 0.0
            try:
                rka = RiemannianKernelAmplifierV3(n_components=rka_p['n_components'],
                                                  a=rka_p['a'], b=rka_p['b'],
                                                  ell=rka_p['ell'], eta=rka_p['eta'])
                X_tr_res = rka.fit_transform(X_tr)
                X_val_res = rka.transform(X_val, X_tr)

                sm = SVMSMOTE(random_state=42, k_neighbors=k_neighbors,
                              sampling_strategy=sampling_strategy)
                X_resampled, y_resampled = sm.fit_resample(X_tr_res, y_tr)

                clf = make_classifier(clf_name, clf_params)
                clf.fit(X_resampled, y_resampled)

                tr_prob = clf.predict_proba(X_tr_res)[:, 1]
                thr_f1 = pick_threshold(tr_prob, y_tr, 'f1')
                thr_g = pick_threshold(tr_prob, y_tr, 'g_mean')
                thr_acc = pick_threshold(tr_prob, y_tr, 'acc')

                y_prob = clf.predict_proba(X_val_res)[:, 1]
                f1 = f1_score(y_val, np.where(y_prob >= thr_f1, 1, -1), pos_label=1)
                g = geometric_mean_score(y_val, np.where(y_prob >= thr_g, 1, -1), pos_label=1)
                acc = accuracy_score(y_val, np.where(y_prob >= thr_acc, 1, -1))
                auc = roc_auc_score(y_val, y_prob)
                score = (METRIC_WEIGHTS['auc'] * auc + METRIC_WEIGHTS['f1'] * f1
                         + METRIC_WEIGHTS['g_mean'] * g + METRIC_WEIGHTS['acc'] * acc)
                scores.append(score)
            except Exception:
                return 0.0
        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params


def main():
    folder_path = 'datasets'
    if len(sys.argv) > 1:
        dat_files = []
        for name in sys.argv[1:]:
            if not name.endswith('.dat'):
                continue
            path = os.path.join(folder_path, name)
            if os.path.exists(path):
                dat_files.append(path)
        if not dat_files:
            print("命令行参数中没有任何可用的数据集文件！")
            return
    else:
        dat_files = glob.glob(os.path.join(folder_path, '*.dat'))
        if not dat_files:
            print("未在 datasets 找到数据集文件！")
            return

    classifiers = ["SVM", "DT"]
    print(f"共 {len(dat_files)} 个数据集（70/30 分层划分 × {N_RUNS} 次重复，正确版 RKA）...\n")
    all_results = []

    for file_path in dat_files:
        dataset_name = os.path.basename(file_path).replace('.dat', '')
        X, y = load_keel_dat(file_path)
        print(f"\n[{dataset_name}]")

        for clf_name in classifiers:
            print(f"  为 {clf_name} 定制空间参数...", end=" ")
            best_p = optimize_for_classifier(X, y, clf_name)
            print(f"ok (a={best_p['a']:.3f} b={best_p['b']:.3f} ell={best_p['ell']:.3f} "
                  f"eta={best_p['eta']:.3f} k={best_p['k_neighbors']} sampling={best_p['sampling_strategy']:.3f})")

            run_auc, run_f1, run_g, run_acc = [], [], [], []
            n_fail, last_err = 0, None
            for seed in range(N_RUNS):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
                    rka = RiemannianKernelAmplifierV3(n_components=best_p['n_components'],
                                                      a=best_p['a'], b=best_p['b'],
                                                      ell=best_p['ell'], eta=best_p['eta'])
                    X_train_res = rka.fit_transform(X_train)
                    X_test_res = rka.transform(X_test, X_train)
                    sm = SVMSMOTE(random_state=42, k_neighbors=best_p['k_neighbors'],
                                  sampling_strategy=best_p['sampling_strategy'])
                    X_resampled, y_resampled = sm.fit_resample(X_train_res, y_train)
                    clf = make_classifier(clf_name, best_p)
                    clf.fit(X_resampled, y_resampled)

                    train_prob = clf.predict_proba(X_train_res)[:, 1]
                    thr_f1 = pick_threshold(train_prob, y_train, 'f1')
                    thr_g = pick_threshold(train_prob, y_train, 'g_mean')
                    thr_acc = pick_threshold(train_prob, y_train, 'acc')

                    y_prob = clf.predict_proba(X_test_res)[:, 1]
                    run_auc.append(roc_auc_score(y_test, y_prob))
                    run_f1.append(f1_score(y_test, np.where(y_prob >= thr_f1, 1, -1), pos_label=1))
                    run_g.append(geometric_mean_score(y_test, np.where(y_prob >= thr_g, 1, -1), pos_label=1))
                    run_acc.append(accuracy_score(y_test, np.where(y_prob >= thr_acc, 1, -1)))
                except Exception as e:
                    n_fail += 1
                    last_err = e
                    continue

            n_success = N_RUNS - n_fail
            if n_success < N_RUNS:
                print(f"  ⚠ 仅 {n_success}/{N_RUNS} 次运行成功（最后错误: {type(last_err).__name__}: {last_err}）")

            all_results.append({
                'Dataset': dataset_name, 'Classifier': clf_name,
                'AUC_mean': np.mean(run_auc) if run_auc else 0.0,
                'AUC_std': np.std(run_auc, ddof=1) if len(run_auc) > 1 else 0.0,
                'F1_mean': np.mean(run_f1) if run_f1 else 0.0,
                'F1_std': np.std(run_f1, ddof=1) if len(run_f1) > 1 else 0.0,
                'G_mean_mean': np.mean(run_g) if run_g else 0.0,
                'G_mean_std': np.std(run_g, ddof=1) if len(run_g) > 1 else 0.0,
                'Acc_mean': np.mean(run_acc) if run_acc else 0.0,
                'Acc_std': np.std(run_acc, ddof=1) if len(run_acc) > 1 else 0.0,
            })

    df_results = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print(f" 跑分结束：70/30 分层划分 × {N_RUNS} 次重复（仅 SVM / DT，正确版 RKA）")
    print("=" * 60)

    for clf_name in classifiers:
        df_clf = df_results[df_results['Classifier'] == clf_name].copy()
        df_clf.drop(columns=['Classifier'], inplace=True)
        df_clf.set_index('Dataset', inplace=True)
        df_show = pd.DataFrame(index=df_clf.index)
        df_show['AUC'] = df_clf.apply(lambda r: f"{r['AUC_mean']:.4f}±{r['AUC_std']:.4f}", axis=1)
        df_show['F1'] = df_clf.apply(lambda r: f"{r['F1_mean']:.4f}±{r['F1_std']:.4f}", axis=1)
        df_show['G_mean'] = df_clf.apply(lambda r: f"{r['G_mean_mean']:.4f}±{r['G_mean_std']:.4f}", axis=1)
        df_show['Accuracy'] = df_clf.apply(lambda r: f"{r['Acc_mean']:.4f}±{r['Acc_std']:.4f}", axis=1)
        print(f"\n---> Table: Comparative results based on the {clf_name} classifier (70/30 x {N_RUNS}) <---")
        print(df_show.to_markdown())
        df_clf[['AUC_mean', 'AUC_std', 'F1_mean', 'F1_std',
                'G_mean_mean', 'G_mean_std', 'Acc_mean', 'Acc_std']].to_csv(
            f"Table_Results_{clf_name}_7030x20_v5.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
