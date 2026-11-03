# -*- coding: utf-8 -*-
"""
六模型消融实验（M1–M6）· 70/30 分层划分 × 20 次重复 · NB / SVM
================================================================
模型链（每一步隔离一个模块的独立贡献）：
  M1  SMOTE                                              —— 基础过采样基准
  M2  SVMSMOTE                                           —— SVM 边界引导采样
  M3  平稳 Matérn 5/2 (w≡1, q≡1) + KernelPCA + SVMSMOTE  —— 基础核空间映射
  M4  非平稳核 w(x)（a=b=ell=1, η=0 → q≡1）+ KernelPCA + SVMSMOTE —— 非平稳位置权重
  M5  非平稳核（η=0，Optuna 调 {a,b,ell}）+ KernelPCA + SVMSMOTE —— 核参数调优
  M6  RKA（Optuna 调 P={a,b,ell,eta}）+ KernelPCA + SVMSMOTE —— 放大因子 + 最终完整算法

协议要点：
  1. 70:30 分层划分 × 20 次（random_state=0..19，与 v5 基准相同的 train_test_split
     写法），六个模型共享同一组 20 个划分 → 结果可逐次配对比较。
  2. 所有模块（StandardScaler / 核映射+KernelPCA / 过采样）只在训练集上拟合；
     M3–M6 的 RKA 内部自带 MinMaxScaler，对先行的 StandardScaler 是仿射不变的
     无操作，因此对六个模型统一先做 StandardScaler 不会改变 M3–M6 的核映射结果。
  3. M1–M4 固定参数（M5/M6 由 Optuna 调参，见下）：
       a=b=ell=1（M3/M4 统一）；η=0（M3/M4）；M3 平稳、M4 非平稳；
       k_neighbors=5、sampling_strategy=1.0、eps=0.01、A0=1.0、
       n_components=min(d, n_train-1)（M3–M6）、
       SVMSMOTE 内部 SVM = SVC(C=1, gamma='scale', rbf, random_state=42) 六组一致。
  4. M5/M6 用训练集内部 3 折分层 CV（shuffle, random_state=42）做 50 次 Optuna
     （TPESampler seed=42），搜索空间与 v5 一致：
       M5 调 {a,b,ell}；M6 调 {a,b,ell,eta}：a∈[0.01,100]log, b∈[0.1,10]log, ell∈[0.05,20]log, η∈[0,2]（M5 中 η 固定 0）；
     目标 = 0.1·AUC + 0.3·F1 + 0.3·G-mean + 0.3·Acc（阈值在各 CV 训练折上逐指标
     选取，0.01~0.99 网格，与 v5 相同）。
  5. 最终分类器六组完全一致且固定：
       NB  = GaussianNB()
       SVM = SVC(probability=True, C=1.0, gamma='scale', kernel='rbf', random_state=42)
  6. 测试集指标：AUC；F1 / G-mean / Acc 使用训练集上逐指标最优阈值。
  7. 输出：20 次均值±标准差、逐次配对差值（M4−M3 / M5−M4 / M6−M5，含 Wilcoxon
     符号秩检验 p 值）；原始逐次结果增量落盘，中断后已完成的运行不丢失。

用法：
  python run_ablation_6models.py     # 完整实验（5 数据集 × {NB,SVM} × 20 次 × 50 trials）
  python run_ablation_6models.py --models M2 M3 M4 M5   # 只跑指定模型子集（如已有 M1/M6 数据）
  python run_ablation_6models.py --datasets ecoli2 --classifiers NB --runs 1 --trials 2  # 冒烟
  python run_ablation_6models.py --fresh   # 先删除既有 Ablation_*.csv 再运行

注意：输出 CSV 为追加写。中断后重跑同一 (数据集, 分类器) 会重复追加已完成的
run 行（不会去重），因此正式运行请使用 --fresh，或先手动删除 Ablation_*.csv。
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, SVMSMOTE

from rka_transform_v3 import RiemannianKernelAmplifierV3

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# 实验配置
# ---------------------------------------------------------------------------
DATASET_SPEC = {
    'wisconsin':   'datasets/wisconsin.dat',
    'ecoli2':      'datasets/ecoli2.dat',
    'yeast3':      'datasets/yeast3.dat',
    'segment0':    'datasets/segment0.dat',
    'abalone9_18': 'datasets/abalone9-18.dat',  # 论文名 abalone9_18 对应文件 abalone9-18.dat
}
CLASSIFIERS = ['NB', 'SVM']
MODELS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
METRICS = ['auc', 'f1', 'gmean', 'acc']
DIFF_PAIRS = [('M4', 'M3'), ('M5', 'M4'), ('M6', 'M5')]  # 逐次配对差值

N_RUNS = 20
TEST_SIZE = 0.3
N_TRIALS = 50

FIXED_K_NEIGHBORS = 5
FIXED_SAMPLING_STRATEGY = 1.0
FIXED_EPS = 0.01
FIXED_A0 = 1.0
FIXED_A, FIXED_B, FIXED_ELL = 1.0, 1.0, 1.0   # M3/M4 统一固定核参数
M5_FIXED_ETA = 0.0                            # M5 固定 η=0（M5 只调 {a,b,ell}，不放大）

METRIC_WEIGHTS = {'auc': 0.10, 'f1': 0.30, 'g_mean': 0.30, 'acc': 0.30}  # 与 v5 一致
THRESH_GRID = np.arange(0.01, 1.00, 0.01)

SMOTE_KWARGS = dict(k_neighbors=FIXED_K_NEIGHBORS,
                    sampling_strategy=FIXED_SAMPLING_STRATEGY, random_state=42)
SVMSMOTE_KWARGS = dict(k_neighbors=FIXED_K_NEIGHBORS,
                       sampling_strategy=FIXED_SAMPLING_STRATEGY, random_state=42,
                       svm_estimator=SVC(C=1.0, gamma='scale', kernel='rbf',
                                         random_state=42))

# M3/M4 的固定核参数（M3 为平稳核 w≡1、q≡1，此时 a、b 是名义参数不进入计算，
# 实际生效的只有 ell 与 A0；M4 为非平稳核但 η=0 → q≡1）
FIXED_KERNEL_PARAMS = {
    'M3': dict(a=FIXED_A, b=FIXED_B, ell=FIXED_ELL, eta=0.0, stationary=True),
    'M4': dict(a=FIXED_A, b=FIXED_B, ell=FIXED_ELL, eta=0.0, stationary=False),
}

RAW_COLUMNS = ['dataset', 'classifier', 'model', 'run', 'auc', 'f1', 'gmean', 'acc']
DIFF_COLUMNS = ['dataset', 'classifier', 'pair', 'run',
                'd_auc', 'd_f1', 'd_gmean', 'd_acc']
BEST_COLUMNS = ['dataset', 'classifier', 'model', 'run', 'a', 'b', 'ell', 'eta', 'best_value']
TABLE_COLUMNS = ['dataset', 'classifier', 'model', 'n',
                 'auc_mean', 'auc_std', 'f1_mean', 'f1_std',
                 'gmean_mean', 'gmean_std', 'acc_mean', 'acc_std']
DIFFSTAT_COLUMNS = ['dataset', 'classifier', 'pair', 'n',
                    'd_auc_mean', 'd_auc_std', 'd_auc_p',
                    'd_f1_mean', 'd_f1_std', 'd_f1_p',
                    'd_gmean_mean', 'd_gmean_std', 'd_gmean_p',
                    'd_acc_mean', 'd_acc_std', 'd_acc_p']


def _path(kind, clf_name):
    return {'raw': f'Ablation_Raw_{clf_name}.csv',
            'diff': f'Ablation_Diff_{clf_name}.csv',
            'best': f'Ablation_BestParams_{clf_name}.csv',
            'table': f'Ablation_Table_{clf_name}.csv',
            'diffstat': f'Ablation_DiffStats_{clf_name}.csv'}[kind]


def _append_csv(path, rows, columns):
    """增量落盘：文件不存在或为空时写表头，否则追加（中断后已完成结果不丢失）。"""
    if not rows:
        return
    header = not (os.path.exists(path) and os.path.getsize(path) > 0)
    pd.DataFrame(rows, columns=columns).to_csv(path, mode='a', header=header, index=False)


def load_keel_dat(file_path):
    """加载 KEEL .dat 文件；少数类编码为 +1，多数类为 -1（与 v5 一致）。

    注：类别列的 LabelEncoder 在划分前于全量数据上拟合（与 v5 完全一致，保证
    两实验结果可直接对照）。已核实对 5 个选定数据集没有任何测试信息流入：
    唯一含类别列的是 abalone9-18.dat（Sex 列），其全部 3 个类别在 20 个划分的
    每个训练集中都出现，训练集单独拟合的编码映射与全量拟合 20/20 次完全一致。
    """
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
    """在给定（训练集）预测概率上逐指标搜索最优阈值（0.01~0.99，与 v5 一致）。"""
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
    """最终分类器：六组一致、参数固定。"""
    if clf_name == 'NB':
        return GaussianNB()
    return SVC(probability=True, C=1.0, gamma='scale', kernel='rbf', random_state=42)


def make_rka(n_components, a, b, ell, eta, stationary=False):
    # kpca_random_state=42：固定 arpack 初始向量，保证跨进程重跑逐位可复现
    # （下游 SVMSMOTE/SVC/GaussianNB 对特征列符号翻转不敏感，仅影响复现性）
    return RiemannianKernelAmplifierV3(n_components=n_components, a=a, b=b, ell=ell,
                                       eta=eta, eps=FIXED_EPS, A0=FIXED_A0,
                                       stationary=stationary, n_jobs=-1,
                                       kpca_random_state=42)


def run_pipeline(X_train, X_test, y_train, y_test, clf_name,
                 kernel_params=None, use_smote=False):
    """统一流水线（全部模块只在训练集上拟合）：
    StandardScaler -> [核映射+KernelPCA] -> SMOTE/SVMSMOTE -> 固定分类器 -> 测试指标。
    kernel_params=None 表示 M1/M2（不进入核空间）。返回四个指标字典。"""
    if np.sum(y_train == 1) <= FIXED_K_NEIGHBORS:
        raise RuntimeError('少数类样本不足（<= k_neighbors=%d），无法 SMOTE' % FIXED_K_NEIGHBORS)

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    if kernel_params is None:                      # M1 / M2：标准化特征空间
        Xtr_t, Xte_t = Xtr_s, Xte_s
    else:                                          # M3–M6：核空间
        n_comp = min(X_train.shape[1], X_train.shape[0] - 1)
        rka = make_rka(n_comp, **kernel_params)
        Xtr_t = rka.fit_transform(Xtr_s)
        Xte_t = rka.transform(Xte_s, Xtr_s)
        if not (np.all(np.isfinite(Xtr_t)) and np.all(np.isfinite(Xte_t))):
            raise RuntimeError('核空间特征包含 NaN/inf')

    if use_smote:                                  # M1
        sm = SMOTE(**SMOTE_KWARGS)
    else:                                          # M2–M6
        sm = SVMSMOTE(**SVMSMOTE_KWARGS)
    X_res, y_res = sm.fit_resample(Xtr_t, y_train)

    clf = make_classifier(clf_name)
    clf.fit(X_res, y_res)

    # 阈值只在训练集上逐指标选取（原始未过采样的训练特征）
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


def eval_model(model_key, X_train, X_test, y_train, y_test, clf_name, tuned=None):
    if model_key == 'M1':
        return run_pipeline(X_train, X_test, y_train, y_test, clf_name,
                            kernel_params=None, use_smote=True)
    if model_key == 'M2':
        return run_pipeline(X_train, X_test, y_train, y_test, clf_name,
                            kernel_params=None)
    if model_key in ('M5', 'M6'):
        return run_pipeline(X_train, X_test, y_train, y_test, clf_name,
                            kernel_params=tuned)  # tuned 只含 a,b,ell,eta
    return run_pipeline(X_train, X_test, y_train, y_test, clf_name,
                        kernel_params=FIXED_KERNEL_PARAMS[model_key])


def optimize_kernel(X_train, y_train, clf_name, n_trials, tune_eta=True):
    """训练集内部 3 折分层 CV × n_trials 次 Optuna 调核参数。
    tune_eta=True（M6）调 P={a,b,ell,eta}；tune_eta=False（M5）调 {a,b,ell}、η 固定 0。
    目标为 v5 同款复合加权（阈值在各 CV 训练折上逐指标选取）。
    n_components 按完整训练集口径 min(d, N_fit-1) 计算，与最终拟合完全一致。"""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    n_comp = min(X_train.shape[1], X_train.shape[0] - 1)

    def objective(trial):
        a = trial.suggest_float('a', 0.01, 100.0, log=True)
        b = trial.suggest_float('b', 0.1, 10.0, log=True)
        ell = trial.suggest_float('ell', 0.05, 20.0, log=True)
        eta = trial.suggest_float('eta', 0.0, 2.0) if tune_eta else M5_FIXED_ETA
        kp = dict(a=a, b=b, ell=ell, eta=eta)

        scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            if np.sum(y_tr == 1) <= FIXED_K_NEIGHBORS:
                return 0.0
            try:
                scaler = StandardScaler().fit(X_tr)
                rka = make_rka(n_comp, **kp)
                Xtr_t = rka.fit_transform(scaler.transform(X_tr))
                Xva_t = rka.transform(scaler.transform(X_va), scaler.transform(X_tr))
                sm = SVMSMOTE(**SVMSMOTE_KWARGS)
                X_res, y_res = sm.fit_resample(Xtr_t, y_tr)
                clf = make_classifier(clf_name)
                clf.fit(X_res, y_res)

                tr_prob = clf.predict_proba(Xtr_t)[:, 1]
                thr_f1 = pick_threshold(tr_prob, y_tr, 'f1')
                thr_g = pick_threshold(tr_prob, y_tr, 'g_mean')
                thr_acc = pick_threshold(tr_prob, y_tr, 'acc')

                va_prob = clf.predict_proba(Xva_t)[:, 1]
                f1 = f1_score(y_va, np.where(va_prob >= thr_f1, 1, -1), pos_label=1)
                g = geometric_mean_score(y_va, np.where(va_prob >= thr_g, 1, -1), pos_label=1)
                acc = accuracy_score(y_va, np.where(va_prob >= thr_acc, 1, -1))
                auc = roc_auc_score(y_va, va_prob)
                scores.append(METRIC_WEIGHTS['auc'] * auc + METRIC_WEIGHTS['f1'] * f1
                              + METRIC_WEIGHTS['g_mean'] * g + METRIC_WEIGHTS['acc'] * acc)
            except Exception:
                return 0.0
        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    # 全部 trial 失败时 best_params 会静默退化为首个随机抽样（best_value=0.0），
    # 此时抛异常让外层正确计入失败，避免把无效参数当作最优结果写入报告。
    if study.best_value <= 0.0:
        raise RuntimeError('Optuna 搜索全部 trial 失败或目标得分为 0 (best_value=%.4f)'
                           % study.best_value)
    params = dict(study.best_params)
    if not tune_eta:
        params['eta'] = 0.0
    return params, study.best_value


def _fmt_mean_std(mean, std):
    return '%.4f+-%.4f' % (mean, std)


def parse_args():
    p = argparse.ArgumentParser(description='RKA+SVMSMOTE 六模型消融实验 (M1-M6)')
    p.add_argument('--datasets', nargs='*', default=list(DATASET_SPEC),
                   choices=list(DATASET_SPEC))
    p.add_argument('--classifiers', nargs='*', default=CLASSIFIERS, choices=CLASSIFIERS)
    p.add_argument('--runs', type=int, default=N_RUNS)
    p.add_argument('--trials', type=int, default=N_TRIALS)
    p.add_argument('--models', nargs='*', default=MODELS, choices=MODELS,
                   help='要运行的模型子集（默认全部 M1-M6，例如 --models M2 M3 M4 M5）')
    p.add_argument('--fresh', action='store_true',
                   help='运行前删除所选分类器的既有 Ablation_*.csv，防止新旧结果混行')
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    if args.fresh:
        for clf_name in args.classifiers:
            for kind in ['raw', 'diff', 'best', 'table', 'diffstat']:
                path = _path(kind, clf_name)
                if os.path.exists(path):
                    os.remove(path)
                    print('已删除旧输出: %s' % path)

    # 本次实际运行的模型子集；差值对只保留两个模型都在子集内的
    selected_models = args.models
    diff_pairs = [(ma, mb) for (ma, mb) in DIFF_PAIRS
                  if ma in selected_models and mb in selected_models]
    print('模型子集: %s | 差值对: %s' % (','.join(selected_models),
                                       ','.join('%s-%s' % (a, b) for a, b in diff_pairs)),
          flush=True)

    for ds_key in args.datasets:
        file_path = DATASET_SPEC[ds_key]
        X, y = load_keel_dat(file_path)
        print('\n[%s] X=%s 少数类=%d' % (ds_key, str(X.shape), int(np.sum(y == 1))), flush=True)

        for clf_name in args.classifiers:
            t0 = time.time()
            raw_rows, diff_rows, best_rows, table_rows, diffstat_rows = [], [], [], [], []
            n_fail = {mk: 0 for mk in selected_models}

            for run in range(args.runs):
                i_raw, i_diff, i_best = len(raw_rows), len(diff_rows), len(best_rows)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=TEST_SIZE, stratify=y, random_state=run)
                per_model, parts = {}, []
                for mk in selected_models:
                    try:
                        if mk in ('M5', 'M6'):
                            best_p, best_val = optimize_kernel(
                                X_train, y_train, clf_name, args.trials,
                                tune_eta=(mk == 'M6'))
                            per_model[mk] = eval_model(mk, X_train, X_test, y_train, y_test,
                                                       clf_name, tuned=best_p)
                            best_rows.append(dict(dataset=ds_key, classifier=clf_name,
                                                  model=mk, run=run,
                                                  a=best_p['a'], b=best_p['b'], ell=best_p['ell'],
                                                  eta=best_p['eta'], best_value=best_val))
                            parts.append('%s(a=%.2f,b=%.2f,ell=%.2f,eta=%.2f)'
                                         % (mk, best_p['a'], best_p['b'], best_p['ell'],
                                            best_p['eta']))
                        else:
                            per_model[mk] = eval_model(mk, X_train, X_test, y_train, y_test,
                                                       clf_name)
                            parts.append('%s ok' % mk)
                    except Exception as e:
                        n_fail[mk] += 1
                        parts.append('%s FAIL(%s)' % (mk, type(e).__name__))
                        print('  [run %d] %s 失败: %s: %s'
                              % (run, mk, type(e).__name__, e), flush=True)
                print('  [%s | %s | run %d/%d] %s'
                      % (ds_key, clf_name, run, args.runs - 1, ' | '.join(parts)), flush=True)

                for mk in selected_models:
                    if mk in per_model:
                        raw_rows.append(dict(dataset=ds_key, classifier=clf_name,
                                             model=mk, run=run, **per_model[mk]))
                for ma, mb in diff_pairs:
                    if ma in per_model and mb in per_model:
                        diff_rows.append(dict(
                            dataset=ds_key, classifier=clf_name, pair='%s-%s' % (ma, mb),
                            run=run,
                            **{'d_%s' % m: per_model[ma][m] - per_model[mb][m] for m in METRICS}))
                _append_csv(_path('raw', clf_name), raw_rows[i_raw:], RAW_COLUMNS)
                _append_csv(_path('diff', clf_name), diff_rows[i_diff:], DIFF_COLUMNS)
                _append_csv(_path('best', clf_name), best_rows[i_best:], BEST_COLUMNS)

            # ---------------- 汇总当前 (数据集, 分类器) ----------------
            df_raw = pd.DataFrame(raw_rows, columns=RAW_COLUMNS)
            df_diff = pd.DataFrame(diff_rows, columns=DIFF_COLUMNS)

            print('\n  --- [%s | %s] 指标表 (mean+-std, n=%d) ---' % (ds_key, clf_name, args.runs))
            print('  %-6s | %-15s | %-15s | %-15s | %-15s'
                  % ('Model', 'AUC', 'F1', 'G-mean', 'Accuracy'))
            for mk in selected_models:
                sub = df_raw[df_raw.model == mk]
                if len(sub) == 0:
                    continue
                row = dict(dataset=ds_key, classifier=clf_name, model=mk, n=len(sub))
                for m in METRICS:
                    row['%s_mean' % m] = sub[m].mean()
                    row['%s_std' % m] = sub[m].std(ddof=1)
                table_rows.append(row)
                print('  %-6s | %-15s | %-15s | %-15s | %-15s'
                      % (mk, _fmt_mean_std(row['auc_mean'], row['auc_std']),
                         _fmt_mean_std(row['f1_mean'], row['f1_std']),
                         _fmt_mean_std(row['gmean_mean'], row['gmean_std']),
                         _fmt_mean_std(row['acc_mean'], row['acc_std'])))

            print('\n  --- [%s | %s] 逐次配对差值 (mean+-std, Wilcoxon p) ---'
                  % (ds_key, clf_name))
            print('  %-7s | %-24s | %-24s | %-24s | %-24s'
                  % ('Pair', 'd_AUC', 'd_F1', 'd_G-mean', 'd_Acc'))
            for ma, mb in diff_pairs:
                sub = df_diff[df_diff.pair == '%s-%s' % (ma, mb)]
                if len(sub) == 0:
                    continue
                row = dict(dataset=ds_key, classifier=clf_name,
                           pair='%s-%s' % (ma, mb), n=len(sub))
                cells = []
                for m in METRICS:
                    vals = sub['d_%s' % m].values
                    mean, std = vals.mean(), vals.std(ddof=1)
                    # n<2 或全零差值时检验无效：p 记为 NaN（避免被误读为 p=1.000）
                    p = np.nan
                    if len(vals) >= 2 and np.any(vals != 0):
                        try:
                            p = wilcoxon(vals)[1]
                        except Exception:
                            p = np.nan
                    row['d_%s_mean' % m] = mean
                    row['d_%s_std' % m] = std
                    row['d_%s_p' % m] = p
                    std_str = ('%.4f' % std) if np.isfinite(std) else 'n/a'
                    p_str = ('%.3f' % p) if np.isfinite(p) else 'n/a'
                    cells.append('%.4f+-%s (p=%s)' % (mean, std_str, p_str))
                diffstat_rows.append(row)
                print('  %-7s | %-24s | %-24s | %-24s | %-24s'
                      % ('%s-%s' % (ma, mb), cells[0], cells[1], cells[2], cells[3]))

            if best_rows:
                df_best = pd.DataFrame(best_rows, columns=BEST_COLUMNS)
                for bm in sorted(set(df_best['model'])):
                    sub_best = df_best[df_best['model'] == bm]
                    print('\n  --- [%s | %s] %s Optuna 最优参数 (n=%d) ---'
                          % (ds_key, clf_name, bm, len(sub_best)))
                    for p_name in ['a', 'b', 'ell', 'eta']:
                        print('  %-4s : mean=%.4f std=%.4f  [%.4f, %.4f]'
                              % (p_name, sub_best[p_name].mean(), sub_best[p_name].std(ddof=1),
                                 sub_best[p_name].min(), sub_best[p_name].max()))
                    print('  best_value : mean=%.4f' % sub_best['best_value'].mean())

            n_fail_str = ', '.join('%s:%d' % (mk, n_fail[mk]) for mk in selected_models if n_fail[mk])
            if n_fail_str:
                print('  [!] 失败次数 -> %s' % n_fail_str)
            print('  [%s | %s] 完成，耗时 %.1fs' % (ds_key, clf_name, time.time() - t0), flush=True)

            _append_csv(_path('table', clf_name), table_rows, TABLE_COLUMNS)
            _append_csv(_path('diffstat', clf_name), diffstat_rows, DIFFSTAT_COLUMNS)

    print('\n全部完成，总耗时 %.1fs。输出文件：' % (time.time() - t_start))
    for clf_name in args.classifiers:
        for kind in ['table', 'diffstat', 'raw', 'diff', 'best']:
            print('  %s' % _path(kind, clf_name))


if __name__ == '__main__':
    main()
