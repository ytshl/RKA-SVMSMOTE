import os
import sys
import glob
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
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
N_TRIALS = 50            # 每次重复（每个 seed）的 Optuna 试验次数
N_CV_SPLITS = 3          # 调参用 3 折分层交叉验证
FIXED_K_NEIGHBORS = 5
FIXED_SAMPLING_STRATEGY = 1.0   # 固定 1:1 采样
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


def make_classifier(clf_name):
    """最终分类器:参数固定,只优化 RKA 的四个参数 a,b,ell,eta。"""
    if clf_name == 'SVM':
        return SVC(probability=True, C=1.0, gamma='scale', kernel='rbf', random_state=42)
    return DecisionTreeClassifier(random_state=42)


def suggest_rka_params(trial):
    """只优化四个 RKA 参数:a, b, ell, eta。"""
    a = trial.suggest_float('a', 0.01, 100.0, log=True)
    b = trial.suggest_float('b', 0.1, 10.0, log=True)
    ell = trial.suggest_float('ell', 0.05, 20.0, log=True)
    eta = trial.suggest_float('eta', 0.0, 2.0)
    return {'a': a, 'b': b, 'ell': ell, 'eta': eta}


def optimize_on_train(X_train, y_train, clf_name, seed):
    """只在【训练集】上调参：训练集内部 3 折分层交叉验证给 Optuna 打分。

    只优化四个 RKA 参数 {a, b, ell, eta}；n_components、k_neighbors、
    采样策略(固定 1:1)与最终分类器参数全部固定。调参结束后用 best_params 在
    完整 X_train 上重训，因此该 seed 的测试集从头到尾不参与任何参数选择。
    """
    skf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=seed)
    n_comp = min(X_train.shape[1], X_train.shape[0] - 1)

    def objective(trial):
        rka_p = suggest_rka_params(trial)

        scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            if np.sum(y_tr == 1) <= FIXED_K_NEIGHBORS:
                return 0.0
            try:
                rka = RiemannianKernelAmplifierV3(n_components=n_comp,
                                                  a=rka_p['a'], b=rka_p['b'],
                                                  ell=rka_p['ell'], eta=rka_p['eta'])
                X_tr_res = rka.fit_transform(X_tr)
                X_va_res = rka.transform(X_va, X_tr)

                sm = SVMSMOTE(random_state=42, k_neighbors=FIXED_K_NEIGHBORS,
                              sampling_strategy=FIXED_SAMPLING_STRATEGY)
                X_resampled, y_resampled = sm.fit_resample(X_tr_res, y_tr)

                clf = make_classifier(clf_name)
                clf.fit(X_resampled, y_resampled)

                tr_prob = clf.predict_proba(X_tr_res)[:, 1]
                thr_f1 = pick_threshold(tr_prob, y_tr, 'f1')
                thr_g = pick_threshold(tr_prob, y_tr, 'g_mean')
                thr_acc = pick_threshold(tr_prob, y_tr, 'acc')

                va_prob = clf.predict_proba(X_va_res)[:, 1]
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
    study.optimize(objective, n_trials=N_TRIALS)
    if study.best_value <= 0.0:
        raise RuntimeError('Optuna 搜索全部 trial 失败或目标得分为 0 (best_value=%.4f)'
                           % study.best_value)
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

                    # 本 seed 的调参只使用 X_train（内部 3 折交叉验证），测试集不参与
                    best_p = optimize_on_train(X_train, y_train, clf_name, seed)
                    print(f"    seed {seed:02d} 调参 ok (a={best_p['a']:.3f} b={best_p['b']:.3f} "
                          f"ell={best_p['ell']:.3f} eta={best_p['eta']:.3f})")

                    n_comp = min(X_train.shape[1], X_train.shape[0] - 1)
                    rka = RiemannianKernelAmplifierV3(n_components=n_comp,
                                                      a=best_p['a'], b=best_p['b'],
                                                      ell=best_p['ell'], eta=best_p['eta'])
                    X_train_res = rka.fit_transform(X_train)
                    X_test_res = rka.transform(X_test, X_train)
                    sm = SVMSMOTE(random_state=42, k_neighbors=FIXED_K_NEIGHBORS,
                                  sampling_strategy=FIXED_SAMPLING_STRATEGY)
                    X_resampled, y_resampled = sm.fit_resample(X_train_res, y_train)
                    clf = make_classifier(clf_name)
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
