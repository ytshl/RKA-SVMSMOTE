import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler


class RiemannianKernelAmplifierV3:
    def __init__(self, n_components=None, a=1.0, b=1.0, ell=1.0, eta=0.5,
                 eps=0.01, A0=1.0, stationary=False, n_jobs=None,
                 kpca_random_state=None):
        self.n_components = n_components
        self.a = a          # 径向尺度（Eq.2）
        self.b = b          # 权重形状（Eq.2）
        self.ell = ell      # Matérn 长度尺度（Eq.3）
        self.eta = eta      # 放大强度（Eq.7）
        self.eps = eps      # 权重下界（Eq.2，保证 w>0）
        self.A0 = A0        # 核振幅（Eq.3）
        # stationary=True：w(x)≡1、q(x)≡1，退化为平稳 Matérn 5/2 核（消融实验 M3 专用）。
        # 默认 False，与既有行为完全一致（v5 基准不受影响）。
        self.stationary = stationary
        self.n_jobs = n_jobs
        # kpca_random_state 仅影响 arpack 求解器的初始向量（n>200 且 n_components<10 时
        # 自动选用 arpack，否则走确定性 dense 求解）；固定它可保证跨进程重跑特征列
        # 符号一致（下游 SVMSMOTE/SVC/GaussianNB 对符号翻转不敏感，只影响逐位复现）。
        self.kpca = KernelPCA(n_components=n_components, kernel='precomputed',
                              n_jobs=n_jobs, random_state=kpca_random_state)

    def _weight(self, norms_sq):
        # Eq.2: w(x) = eps + 1 - (a/(a+||x||^2))^b
        return self.eps + 1.0 - (self.a / (self.a + norms_sq)) ** self.b

    def _matern(self, X, Y):
        # Eq.3: Matern 5/2，含长度尺度 ell
        r = cdist(X, Y, metric='euclidean') / self.ell
        s = np.sqrt(5.0) * r
        return self.A0 * (1.0 + s + (s ** 2) / 3.0) * np.exp(-s)

    def _log_det_g(self, w, grad_normsq):
        # Eq.6 的对数形式（避免数值溢出）：c = 5/(3 ell^2)
        c = 5.0 / (3.0 * self.ell ** 2)
        cw2 = c * w ** 2
        return (self.d_ * np.log(self.A0)
                + self.d_ * np.log(cw2)
                + np.log1p(grad_normsq / cw2))

    def _log_q(self, log_det_g):
        # q(x) = (det(g)/D_tr)^(eta/(2d))，D_tr 为训练样本 det(g) 的几何平均
        log_D_tr = np.mean(log_det_g)
        return (self.eta / (2.0 * self.d_)) * (log_det_g - log_D_tr), log_D_tr

    def fit_transform(self, X):
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.d_ = X_scaled.shape[1]

        norms_sq = np.sum(X_scaled ** 2, axis=1, keepdims=True)  # Nx1
        if self.stationary:
            w = np.ones_like(norms_sq)                           # 平稳核：w≡1
        else:
            w = self._weight(norms_sq)                           # Nx1

        K_M52 = self._matern(X_scaled, X_scaled)
        K_theta = (w @ w.T) * K_M52                              # Eq.4

        if self.stationary:
            q = np.ones_like(w)                                  # 平稳核：q≡1（无度量放大）
            log_D_tr = 0.0
        else:
            # grad w = 2 b a^b / (a+||x||^2)^{b+1} * x
            denom = (self.a + norms_sq) ** (self.b + 1.0)
            coef = 2.0 * self.b * (self.a ** self.b) / denom
            grad_w = coef * X_scaled                             # Nxd
            grad_normsq = np.sum(grad_w ** 2, axis=1, keepdims=True)  # Nx1

            log_det_g = self._log_det_g(w, grad_normsq)          # Nx1
            log_q, log_D_tr = self._log_q(log_det_g)
            q = np.exp(log_q)                                    # Nx1

        K_amp = (q @ q.T) * K_theta                              # Eq.7

        self.X_train_scaled_ = X_scaled
        self.w_train_ = w
        self.q_train_ = q
        self.log_D_tr_ = log_D_tr

        return self.kpca.fit_transform(K_amp)

    def transform(self, X_test, X_train):
        # 注意：X_train 参数仅为兼容既有调用接口而保留，本实现不使用它；
        # 测试样本的核与权重只依赖 fit_transform 阶段存储的训练状态
        # （self.scaler / self.X_train_scaled_ / self.w_train_ / self.q_train_ / self.log_D_tr_）。
        # 测试样本用训练阶段的 scaler / 参考行列式 / 训练侧权重
        X_test_scaled = self.scaler.transform(X_test)

        norms_sq_test = np.sum(X_test_scaled ** 2, axis=1, keepdims=True)
        if self.stationary:
            w_test = np.ones_like(norms_sq_test)
        else:
            w_test = self._weight(norms_sq_test)

        K_M52_test = self._matern(X_test_scaled, self.X_train_scaled_)  # Ntest x Ntrain
        K_theta_test = (w_test @ self.w_train_.T) * K_M52_test          # Eq.4

        if self.stationary:
            q_test = np.ones_like(norms_sq_test)                        # 平稳核：q≡1
        else:
            denom = (self.a + norms_sq_test) ** (self.b + 1.0)
            coef = 2.0 * self.b * (self.a ** self.b) / denom
            grad_w_test = coef * X_test_scaled
            grad_normsq_test = np.sum(grad_w_test ** 2, axis=1, keepdims=True)
            log_det_g_test = self._log_det_g(w_test, grad_normsq_test)
            log_q_test = (self.eta / (2.0 * self.d_)) * (log_det_g_test - self.log_D_tr_)
            q_test = np.exp(log_q_test)

        K_amp_test = (q_test @ self.q_train_.T) * K_theta_test          # Eq.7

        return self.kpca.transform(K_amp_test)
