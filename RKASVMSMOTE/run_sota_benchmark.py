import os
import glob
import numpy as np
import pandas as pd
import optuna
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from rka_transform_v3 import RiemannianKernelAmplifierV3
from imblearn.over_sampling import SVMSMOTE

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_RUNS = 20        # 70/30 分层划分重复次数
TEST_SIZE = 0.3    # 测试集比例
N_TRIALS = 50      # 每个分类器 Optuna 调参次数
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

    # --- 升级版特征处理：自动识别并转换字符串特征 ---
    X_df = df.iloc[:, :-1]
    for col in X_df.columns:
        try:
            X_df[col] = X_df[col].astype(float)
        except ValueError:
            X_df[col] = LabelEncoder().fit_transform(X_df[col])
    X = X_df.values
    # ---------------------------------------------

    # 标签处理保持不变
    y_raw = df.iloc[:, -1].values
    y = LabelEncoder().fit_transform(y_raw)
    counts = np.bincount(y)
    minority_class = np.argmin(counts)
    y = np.where(y == minority_class, 1, -1)

    return X, y


def pick_threshold(prob, y, metric):
    """在给定概率与标签上，为某一指标挑选最优阈值（0.5 兜底）。"""
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


def make_classifier(clf_name, params):
    if clf_name == 'NaiveBayes':
        return GaussianNB(var_smoothing=params['var_smoothing'])
    return LogisticRegression(max_iter=1000, C=params['C'],
                              class_weight=params['class_weight'])


def suggest_classifier_params(clf_name, trial):
    if clf_name == 'NaiveBayes':
        return {'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True)}
    C = trial.suggest_float('C', 1e-3, 1e3, log=True)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    return {'C': C, 'class_weight': class_weight}


def optimize_for_classifier(X, y, clf_name):
    """
    让当前分类器寻找最适合它的 RKA 参数、SVMSMOTE 邻居数/采样比例与分类器参数。
    采用正确版 RKA（RiemannianKernelAmplifierV3，含 a/b/ell/eta），
    且对每个指标在训练集上选最优阈值（与 v5 同一协议）。
    """
    skf_opt = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    max_dim = X.shape[1]

    def objective(trial):
        a = trial.suggest_float('a', 0.01, 100.0, log=True)
        b = trial.suggest_float('b', 0.1, 10.0, log=True)
        ell = trial.suggest_float('ell', 0.05, 20.0, log=True)
        eta = trial.suggest_float('eta', 0.0, 2.0)
        n_comp = trial.suggest_int('n_components', 2, max(3, max_dim))
        k_neighbors = trial.suggest_int('k_neighbors', 3, 10)
        sampling_strategy = trial.suggest_float('sampling_strategy', 0.1, 1.0)
        clf_params = suggest_classifier_params(clf_name, trial)

        scores = []
        for tr_idx, val_idx in skf_opt.split(X, y):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            if sum(y_tr == 1) <= k_neighbors:
                return 0.0

            try:
                rka = RiemannianKernelAmplifierV3(n_components=n_comp, a=a, b=b,
                                                  ell=ell, eta=eta)
                X_tr_res = rka.fit_transform(X_tr)
                X_val_res = rka.transform(X_val, X_tr)

                bsmote = SVMSMOTE(random_state=42, k_neighbors=k_neighbors,
                                  sampling_strategy=sampling_strategy)
                X_resampled, y_resampled = bsmote.fit_resample(X_tr_res, y_tr)

                clf = make_classifier(clf_name, clf_params)
                clf.fit(X_resampled, y_resampled)

                # 训练集上选各指标最优阈值
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

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params


def main():
    folder_path = 'datasets'
    dat_files = glob.glob(os.path.join(folder_path, '*.dat'))
    if not dat_files:
        print(f"未在 {folder_path} 找到数据集文件！")
        return

    classifiers = ["NaiveBayes", "LR"]

    print(f"共 {len(dat_files)} 个数据集（70/30 分层划分 × {N_RUNS} 次重复，正确版 RKA，逐指标最优阈值）...\n")

    # [新增] 专门用来收集所有明细数据的列表
    all_results = []

    for file_path in dat_files:
        dataset_name = os.path.basename(file_path).replace('.dat', '')
        X, y = load_keel_dat(file_path)
        print(f"\n[{dataset_name}]")

        for clf_name in classifiers:
            print(f"  为 {clf_name} 定制空间参数...", end=" ")
            best_p = optimize_for_classifier(X, y, clf_name)
            print(f"ok (k={best_p['k_neighbors']} sampling={best_p['sampling_strategy']:.3f} "
                  f"a={best_p['a']:.3f} b={best_p['b']:.3f} ell={best_p['ell']:.3f} eta={best_p['eta']:.3f})")

            # 70/30 分层划分 × N_RUNS 次重复
            run_auc = []
            run_f1 = []
            run_g_mean = []
            run_acc = []

            for seed in range(N_RUNS):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)

                try:
                    rka = RiemannianKernelAmplifierV3(n_components=best_p['n_components'],
                                                      a=best_p['a'], b=best_p['b'],
                                                      ell=best_p['ell'], eta=best_p['eta'])
                    X_train_res = rka.fit_transform(X_train)
                    X_test_res = rka.transform(X_test, X_train)

                    bsmote = SVMSMOTE(random_state=42, k_neighbors=best_p['k_neighbors'],
                                      sampling_strategy=best_p['sampling_strategy'])
                    X_resampled, y_resampled = bsmote.fit_resample(X_train_res, y_train)

                    clf = make_classifier(clf_name, best_p)
                    clf.fit(X_resampled, y_resampled)

                    train_prob = clf.predict_proba(X_train_res)[:, 1]
                    thr_f1 = pick_threshold(train_prob, y_train, 'f1')
                    thr_g = pick_threshold(train_prob, y_train, 'g_mean')
                    thr_acc = pick_threshold(train_prob, y_train, 'acc')

                    y_prob = clf.predict_proba(X_test_res)[:, 1]
                    run_auc.append(roc_auc_score(y_test, y_prob))
                    run_f1.append(f1_score(y_test, np.where(y_prob >= thr_f1, 1, -1), pos_label=1))
                    run_g_mean.append(geometric_mean_score(y_test, np.where(y_prob >= thr_g, 1, -1), pos_label=1))
                    run_acc.append(accuracy_score(y_test, np.where(y_prob >= thr_acc, 1, -1)))
                except Exception:
                    # 如果这轮报错，则跳过
                    continue

            # 计算这 N_RUNS 次的均值与标准差
            ds_auc = np.mean(run_auc) if run_auc else 0.0
            ds_auc_std = np.std(run_auc, ddof=1) if len(run_auc) > 1 else 0.0
            ds_f1 = np.mean(run_f1) if run_f1 else 0.0
            ds_f1_std = np.std(run_f1, ddof=1) if len(run_f1) > 1 else 0.0
            ds_g_mean = np.mean(run_g_mean) if run_g_mean else 0.0
            ds_g_mean_std = np.std(run_g_mean, ddof=1) if len(run_g_mean) > 1 else 0.0
            ds_acc = np.mean(run_acc) if run_acc else 0.0
            ds_acc_std = np.std(run_acc, ddof=1) if len(run_acc) > 1 else 0.0

            # 记录到总表
            all_results.append({
                'Dataset': dataset_name,
                'Classifier': clf_name,
                'AUC': ds_auc,
                'AUC_std': ds_auc_std,
                'G_mean': ds_g_mean,
                'G_mean_std': ds_g_mean_std,
                'F1': ds_f1,
                'F1_std': ds_f1_std,
                'Accuracy': ds_acc,
                'Accuracy_std': ds_acc_std,
            })

    # 将收集到的结果转换为 DataFrame，方便进行排版和操作
    df_results = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print(f" 跑分结束：每个数据集在不同分类器下的具体表现（70/30 × {N_RUNS}，正确版 RKA，逐指标最优阈值）")
    print("=" * 60)

    # 按照分类器分组，并打印出类似论文中的明细表格
    for clf_name in classifiers:
        df_clf = df_results[df_results['Classifier'] == clf_name].copy()
        df_clf.drop(columns=['Classifier'], inplace=True)
        df_clf.set_index('Dataset', inplace=True)

        df_show = pd.DataFrame(index=df_clf.index)
        df_show['AUC'] = df_clf.apply(lambda r: f"{r['AUC']:.4f}±{r['AUC_std']:.4f}", axis=1)
        df_show['G_mean'] = df_clf.apply(lambda r: f"{r['G_mean']:.4f}±{r['G_mean_std']:.4f}", axis=1)
        df_show['F1'] = df_clf.apply(lambda r: f"{r['F1']:.4f}±{r['F1_std']:.4f}", axis=1)
        df_show['Accuracy'] = df_clf.apply(lambda r: f"{r['Accuracy']:.4f}±{r['Accuracy_std']:.4f}", axis=1)

        print(f"\n---> Table: Comparative results based on the {clf_name} classifier <---")
        print(df_show.to_markdown())

        df_clf.to_csv(f"Table_Results_{clf_name}.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
