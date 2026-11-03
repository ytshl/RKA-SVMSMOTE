# -*- coding: utf-8 -*-
"""
RKA + SVMSMOTE 基准测试（70/30 分层划分 × 20 次重复 · v5）
=============================================================
v5 相对 v4 的唯一改动：RKA 换成【正确版】RiemannianKernelAmplifierV3（严格按论文 Section 3.1）：
  - 权重函数 w(x) = eps + 1 - (a/(a+||x||^2))^b（正则化不完全 Beta）
  - Matérn 5/2 核含长度尺度 ell
  - 度量放大因子 q(x) = (det(g)/D_tr)^(eta/(2d))（逐样本、位置相关）
  - 新增调参维度：a / b / ell / eta
其余协议与 v4 一致：70/30 分层 × 20 次重复，仅 SVM / DT，逐指标最优阈值，
SVMSMOTE sampling_strategy 0.1~2.0，标签域 {-1,+1}（少数类=1），
结果存档 Table_Results_{SVM,DT}_7030x20_v5.csv。

调参协议（v5.1 起）：每次重复先在【该次 70% 训练集内部】再划出 TUNE_VAL_SIZE
比例作为验证集，做 N_TRIALS 次 Optuna 调参（TPESampler seed=42），验证集只用于
打分；调参结束后把验证集合并回训练集（直接用完整 X_train 重训），测试集全程不
参与调参。总调参开销约为原"全量 3 折 CV 调参一次"的 6~7 倍。
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from rka_transform import RiemannianKernelAmplifierV3
from imblearn.over_sampling import SVMSMOTE

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_RUNS = 20
TEST_SIZE = 0.3
N_TRIALS = 150            # 每次重复（每个 seed）的 Optuna 试验次数
TUNE_VAL_SIZE = 0.2       # 从训练集内部划出的调参验证集比例
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


def optimize_on_train(X_train, y_train, clf_name, seed):
    """只在【训练集】上调参：从训练集内部再划出验证集给 Optuna 打分。

    验证集不参与最终训练；调参结束后用 best_params 在完整 X_train 上重训，
    因此该 seed 的测试集从头到尾不参与任何参数选择。
    """
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=TUNE_VAL_SIZE, stratify=y_train,
        random_state=seed)
    max_dim = X_train.shape[1]

    def objective(trial):
        rka_p = suggest_rka_params(trial, max_dim)
        k_neighbors = trial.suggest_int('k_neighbors', 3, 10)
        sampling_strategy = trial.suggest_float('sampling_strategy', 0.1, 2.0)
        clf_params = suggest_classifier_params(clf_name, trial)

        if sum(y_sub == 1) <= k_neighbors:
            return 0.0
        try:
            rka = RiemannianKernelAmplifierV3(n_components=rka_p['n_components'],
                                              a=rka_p['a'], b=rka_p['b'],
                                              ell=rka_p['ell'], eta=rka_p['eta'])
            X_sub_res = rka.fit_transform(X_sub)
            X_val_res = rka.transform(X_val, X_sub)

            sm = SVMSMOTE(random_state=42, k_neighbors=k_neighbors,
                          sampling_strategy=sampling_strategy)
            X_resampled, y_resampled = sm.fit_resample(X_sub_res, y_sub)

            clf = make_classifier(clf_name, clf_params)
            clf.fit(X_resampled, y_resampled)

            sub_prob = clf.predict_proba(X_sub_res)[:, 1]
            thr_f1 = pick_threshold(sub_prob, y_sub, 'f1')
            thr_g = pick_threshold(sub_prob, y_sub, 'g_mean')
            thr_acc = pick_threshold(sub_prob, y_sub, 'acc')

            y_prob = clf.predict_proba(X_val_res)[:, 1]
            f1 = f1_score(y_val, np.where(y_prob >= thr_f1, 1, -1), pos_label=1)
            g = geometric_mean_score(y_val, np.where(y_prob >= thr_g, 1, -1), pos_label=1)
            acc = accuracy_score(y_val, np.where(y_prob >= thr_acc, 1, -1))
            auc = roc_auc_score(y_val, y_prob)
            return (METRIC_WEIGHTS['auc'] * auc + METRIC_WEIGHTS['f1'] * f1
                    + METRIC_WEIGHTS['g_mean'] * g + METRIC_WEIGHTS['acc'] * acc)
        except Exception:
            return 0.0

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
            print(f"  {clf_name}：逐次在训练集内调参（每次 {N_TRIALS} 次 Optuna）...")

            run_auc, run_f1, run_g, run_acc = [], [], [], []
            n_fail, last_err = 0, None
            for seed in range(N_RUNS):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)

                    # 本 seed 的调参只使用 X_train（内部再划验证集），测试集不参与
                    best_p = optimize_on_train(X_train, y_train, clf_name, seed)
                    print(f"    seed {seed:02d} 调参 ok (a={best_p['a']:.3f} b={best_p['b']:.3f} "
                          f"ell={best_p['ell']:.3f} eta={best_p['eta']:.3f} "
                          f"k={best_p['k_neighbors']} sampling={best_p['sampling_strategy']:.3f})")

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
