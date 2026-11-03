# -*- coding: utf-8 -*-
"""
RKA-SVMSMOTE-Fixed 主实验脚本（固定参数,不使用 Optuna）
========================================================
核参数固定：a = b = ell = 1,  eta = 0.5
其余配置与完整 RKA-SVMSMOTE 完全一致（run_ablation_6models.py 的 M6 管道口径）：
  StandardScaler -> RKA(w(x) + Matérn5/2 + q(x)) + KernelPCA(precomputed,
      n_components=min(d, n_train-1)) -> SVMSMOTE(k_neighbors=5,
      sampling_strategy=1.0, 内部 SVC(C=1, gamma='scale', rbf, random_state=42))
  -> 固定分类器 -> 训练集逐指标最优阈值(0.01~0.99) -> 测试指标 AUC/F1/G-mean/Acc
其余固定量：eps=0.01、A0=1.0、kpca_random_state=42、n_jobs=-1。

实验规模：
  20 个 KEEL 数据集 × {NB, LR, SVM, DT} × 70:30 分层划分 × 20 次重复(seed 0..19)
分类器（参数固定,与消融协议同口径）：
  NB  = GaussianNB()
  LR  = LogisticRegression(max_iter=1000, random_state=42)
  SVM = SVC(probability=True, C=1.0, gamma='scale', kernel='rbf', random_state=42)
  DT  = DecisionTreeClassifier(random_state=42)

输出：
  RKA_SVMSMOTE_Fixed_Raw.csv                     逐 run 增量落盘(中断后可续跑,自动跳过已完成)
  Table_Results_{NB,LR,SVM,DT}_7030x20_RKAFixed.csv   各分类器 mean±std 表
  Table_Results_All_7030x20_RKAFixed.csv         四分类器合并表

用法：
  python run_rka_svmsmote_fixed.py                                # 完整实验
  python run_rka_svmsmote_fixed.py --datasets wisconsin --runs 2  # 冒烟
  python run_rka_svmsmote_fixed.py --classifiers NB SVM           # 只跑部分分类器
"""
import os
import sys
import glob
import time
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SVMSMOTE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from rka_transform_v3 import RiemannianKernelAmplifierV3

warnings.filterwarnings('ignore')

N_RUNS = 20
TEST_SIZE = 0.3
THRESH_GRID = np.arange(0.01, 1.00, 0.01)

# RKA-SVMSMOTE-Fixed 固定参数
FIXED_A, FIXED_B, FIXED_ELL = 1.0, 1.0, 1.0
FIXED_ETA = 0.5
FIXED_EPS = 0.01
FIXED_A0 = 1.0

FIXED_K_NEIGHBORS = 5
FIXED_SAMPLING_STRATEGY = 1.0

ALL_CLASSIFIERS = ['NB', 'LR', 'SVM', 'DT']

RAW_PATH = 'RKA_SVMSMOTE_Fixed_Raw.csv'
RAW_COLUMNS = ['dataset', 'classifier', 'run', 'auc', 'f1', 'gmean', 'acc']


def load_keel_dat(file_path):
    """加载 KEEL .dat 文件；少数类编码为 +1，多数类为 -1（与 v5 / 消融一致）。"""
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
    return X, np.where(y == minority_class, 1, -1)


def pick_threshold(prob, y, metric):
    """在训练集预测概率上逐指标搜索最优阈值(0.01~0.99,与 v5 / 消融一致)。"""
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


def make_classifier(clf_name):
    """最终分类器:四组一致、参数固定。"""
    if clf_name == 'NB':
        return GaussianNB()
    if clf_name == 'LR':
        return LogisticRegression(max_iter=1000, random_state=42)
    if clf_name == 'SVM':
        return SVC(probability=True, C=1.0, gamma='scale', kernel='rbf', random_state=42)
    return DecisionTreeClassifier(random_state=42)


def run_pipeline(X_train, X_test, y_train, y_test, clf_name, eta):
    """RKA-SVMSMOTE-Fixed 流水线(全部模块只在训练集上拟合):
    StandardScaler -> RKA+KernelPCA -> SVMSMOTE -> 固定分类器 -> 逐指标阈值 -> 测试指标。"""
    if np.sum(y_train == 1) <= FIXED_K_NEIGHBORS:
        raise RuntimeError('少数类样本不足(<= k_neighbors=%d),无法 SVMSMOTE' % FIXED_K_NEIGHBORS)

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    n_comp = min(X_train.shape[1], X_train.shape[0] - 1)
    rka = RiemannianKernelAmplifierV3(n_components=n_comp, a=FIXED_A, b=FIXED_B,
                                      ell=FIXED_ELL, eta=eta, eps=FIXED_EPS,
                                      A0=FIXED_A0, n_jobs=-1, kpca_random_state=42)
    Xtr_t = rka.fit_transform(Xtr_s)
    Xte_t = rka.transform(Xte_s, Xtr_s)
    if not (np.all(np.isfinite(Xtr_t)) and np.all(np.isfinite(Xte_t))):
        raise RuntimeError('核空间特征包含 NaN/inf')

    sm = SVMSMOTE(k_neighbors=FIXED_K_NEIGHBORS,
                  sampling_strategy=FIXED_SAMPLING_STRATEGY, random_state=42,
                  svm_estimator=SVC(C=1.0, gamma='scale', kernel='rbf', random_state=42))
    X_res, y_res = sm.fit_resample(Xtr_t, y_train)

    clf = make_classifier(clf_name)
    clf.fit(X_res, y_res)

    # 阈值只在训练集上逐指标选取(原始未过采样的训练特征)
    train_prob = clf.predict_proba(Xtr_t)[:, 1]
    thr_f1 = pick_threshold(train_prob, y_train, 'f1')
    thr_g = pick_threshold(train_prob, y_train, 'g_mean')
    thr_acc = pick_threshold(train_prob, y_train, 'acc')

    y_prob = clf.predict_proba(Xte_t)[:, 1]
    return dict(
        auc=roc_auc_score(y_test, y_prob),
        f1=f1_score(y_test, np.where(y_prob >= thr_f1, 1, -1), pos_label=1),
        gmean=geometric_mean_score(y_test, np.where(y_prob >= thr_g, 1, -1), pos_label=1),
        acc=accuracy_score(y_test, np.where(y_prob >= thr_acc, 1, -1)),
    )


def append_raw_row(path, row):
    header = not (os.path.exists(path) and os.path.getsize(path) > 0)
    pd.DataFrame([row], columns=RAW_COLUMNS).to_csv(path, mode='a', header=header, index=False)


def load_done_keys(path):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return set()
    df = pd.read_csv(path)
    return set(zip(df['dataset'], df['classifier'], df['run']))


def summarize(raw_df, classifiers, eta):
    """按分类器汇总 mean±std,打印并落盘(与 v5 输出表同款列)。"""
    all_rows = []
    for clf_name in classifiers:
        df_clf = raw_df[raw_df['classifier'] == clf_name].copy()
        rows = []
        for ds, sub in df_clf.groupby('dataset'):
            rows.append({
                'Dataset': ds,
                'AUC_mean': sub['auc'].mean(), 'AUC_std': sub['auc'].std(ddof=1),
                'F1_mean': sub['f1'].mean(), 'F1_std': sub['f1'].std(ddof=1),
                'G_mean_mean': sub['gmean'].mean(), 'G_mean_std': sub['gmean'].std(ddof=1),
                'Acc_mean': sub['acc'].mean(), 'Acc_std': sub['acc'].std(ddof=1),
            })
        if not rows:
            continue
        df_tbl = pd.DataFrame(rows).set_index('Dataset')
        df_tbl.to_csv('Table_Results_%s_7030x20_RKAFixed_eta%.1f.csv' % (clf_name, eta),
                      float_format='%.4f')
        for r in rows:
            r['Classifier'] = clf_name
            all_rows.append(r)
        show = pd.DataFrame(index=df_tbl.index)
        show['AUC'] = df_tbl.apply(lambda r: '%.4f+-%.4f' % (r['AUC_mean'], r['AUC_std']), axis=1)
        show['F1'] = df_tbl.apply(lambda r: '%.4f+-%.4f' % (r['F1_mean'], r['F1_std']), axis=1)
        show['G_mean'] = df_tbl.apply(lambda r: '%.4f+-%.4f' % (r['G_mean_mean'], r['G_mean_std']), axis=1)
        show['Accuracy'] = df_tbl.apply(lambda r: '%.4f+-%.4f' % (r['Acc_mean'], r['Acc_std']), axis=1)
        print('\n---> Table: RKA-SVMSMOTE-Fixed (eta=%.1f) / %s (70/30 x %d) <---'
              % (eta, clf_name, N_RUNS))
        print(show.to_markdown())
    if all_rows:
        pd.DataFrame(all_rows)[['Dataset', 'Classifier', 'AUC_mean', 'AUC_std', 'F1_mean',
                                'F1_std', 'G_mean_mean', 'G_mean_std', 'Acc_mean',
                                'Acc_std']].to_csv(
            'Table_Results_All_7030x20_RKAFixed_eta%.1f.csv' % eta, float_format='%.4f', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='只跑指定数据集(不含 .dat 后缀),默认全部 20 个')
    parser.add_argument('--classifiers', nargs='*', default=None,
                        help='只跑指定分类器,默认 NB LR SVM DT')
    parser.add_argument('--runs', type=int, default=N_RUNS, help='重复次数(默认 20)')
    parser.add_argument('--eta', type=float, default=FIXED_ETA, help='放大指数 η(默认 0.5)')
    parser.add_argument('--fresh', action='store_true', help='删除已有 Raw 重新开始')
    args = parser.parse_args()

    raw_path = 'RKA_SVMSMOTE_Fixed_Raw_eta%.1f.csv' % args.eta
    if args.fresh and os.path.exists(raw_path):
        os.remove(raw_path)
        print('已删除旧 Raw:%s' % raw_path)

    classifiers = args.classifiers or ALL_CLASSIFIERS
    for c in classifiers:
        if c not in ALL_CLASSIFIERS:
            raise SystemExit('未知分类器:%s(可选 %s)' % (c, ' '.join(ALL_CLASSIFIERS)))

    if args.datasets:
        dat_files = []
        for name in args.datasets:
            p = os.path.join('datasets', name + ('' if name.endswith('.dat') else '.dat'))
            if os.path.exists(p):
                dat_files.append(p)
            else:
                print('[!] 数据集不存在,跳过:%s' % name)
        if not dat_files:
            raise SystemExit('没有可用的数据集文件')
    else:
        dat_files = sorted(glob.glob(os.path.join('datasets', '*.dat')))

    done = load_done_keys(raw_path)
    print('RKA-SVMSMOTE-Fixed  a=b=ell=1, eta=%.1f(固定,无 Optuna)' % args.eta)
    print('%d 个数据集 x %s x %d 次重复;已有完成记录 %d 个'
          % (len(dat_files), ' '.join(classifiers), args.runs, len(done)))

    for file_path in dat_files:
        ds = os.path.basename(file_path).replace('.dat', '')
        X, y = load_keel_dat(file_path)
        print('\n[%s] X=%s 少数类=%d' % (ds, X.shape, np.sum(y == 1)), flush=True)
        t0 = time.time()
        n_fail = 0
        for clf_name in classifiers:
            for seed in range(args.runs):
                key = (ds, clf_name, seed)
                if key in done:
                    continue
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
                    m = run_pipeline(X_train, X_test, y_train, y_test, clf_name, args.eta)
                    append_raw_row(raw_path, dict(dataset=ds, classifier=clf_name,
                                                  run=seed, **m))
                    print('  [%s | %s | run %d/%d] AUC=%.4f F1=%.4f G=%.4f Acc=%.4f'
                          % (ds, clf_name, seed, args.runs - 1, m['auc'], m['f1'],
                             m['gmean'], m['acc']), flush=True)
                except Exception as e:
                    n_fail += 1
                    print('  [%s | %s | run %d/%d] 失败: %s: %s'
                          % (ds, clf_name, seed, args.runs - 1,
                             type(e).__name__, e), flush=True)
        if n_fail:
            print('  [!] %s 共 %d 个 run 失败(其余已落盘)' % (ds, n_fail), flush=True)
        print('[%s] 本轮耗时 %.1fs' % (ds, time.time() - t0), flush=True)

    raw_df = pd.read_csv(raw_path)
    print('\n' + '=' * 60)
    print(' 跑分结束:20 数据集 x 4 分类器 x %d 次重复,共 %d 条记录'
          % (args.runs, len(raw_df)))
    print('=' * 60)
    summarize(raw_df, classifiers, args.eta)
    print('\n全部完成。')


if __name__ == '__main__':
    main()
