import numpy as np
import random
from numpy.linalg import inv



def my_rbf_kernel(X_train, X_new=None, sigma=1.0):
    
    if X_new is None:
        X_new = X_train

    X1_sq = np.sum(X_train**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X_new**2, axis=1).reshape(1, -1)

    dist_sq = X1_sq + X2_sq - 2 * np.dot(X_train, X_new.T)

    return np.exp(-dist_sq / (2 * sigma**2))



def combine_kernels(K1, K2, alpha=0.5):
    
    return alpha * K1 + (1 - alpha) * K2



def _graph_nmf(
    Y_train,
    W_kernel,
    rank,
    lam_nmf,
    max_iter=10000,
    tol=1e-6,
    random_state=1949
):
   

    np.random.seed(random_state)
    random.seed(random_state)

    X = Y_train.copy()
    W = np.array(W_kernel)

    m, n = X.shape
    k = rank

    D = np.diag(W.sum(axis=1))
    L = D - W

    U = np.random.random((m, k))
    V = np.random.random((n, k))

    eps = 1e-8

    term1 = np.linalg.norm(X - U.dot(V.T)) ** 2
    term2 = lam_nmf * np.trace(V.T.dot(L).dot(V))

    obj0 = term1 + term2
    obj1 = obj0

    for _ in range(max_iter):

        # update U
        U *= (X.dot(V)) / (U.dot(V.T).dot(V) + eps)

        # update V
        num = X.T.dot(U) + lam_nmf * W.dot(V)
        den = V.dot(U.T).dot(U) + lam_nmf * D.dot(V) + eps
        V *= num / den

        term1 = np.linalg.norm(X - U.dot(V.T)) ** 2
        term2 = lam_nmf * np.trace(V.T.dot(L).dot(V))
        obj2 = term1 + term2

        if (obj1 - obj2) < (obj0 * tol):
            break

        obj1 = obj2

    return U, V



def _kr_weights_vkr(K_train, V_train, lam_kr):
   
    n = K_train.shape[0]
    reg = np.eye(n) * lam_kr

    W = inv(K_train.dot(K_train) + reg).dot(
        K_train.dot(V_train)
    )

    return W



def _vkr_predict(
    X_dgi_train,
    X_chem_train,
    Y_train,
    X_dgi_test,
    X_chem_test,
    sigma,
    lam_kr,
    lam_nmf,
    alpha,
    rank
):

    # Step 1
    K1_tr = my_rbf_kernel(X_dgi_train, sigma=sigma)
    K2_tr = my_rbf_kernel(X_chem_train, sigma=sigma)

    K_tr = combine_kernels(
        K1_tr,
        K2_tr,
        alpha
    )

    # Step 2
    U, V = _graph_nmf(
        Y_train,
        W_kernel=K_tr,
        rank=rank,
        lam_nmf=lam_nmf
    )

    # Step 3
    W_kr = _kr_weights_vkr(
        K_tr,
        V,
        lam_kr
    )

    # Step 4
    K1_te = my_rbf_kernel(
    X_dgi_test,
    X_dgi_train,
    sigma=sigma
)

    K2_te = my_rbf_kernel(
    X_chem_test,
    X_chem_train,
    sigma=sigma
)

    K_te = combine_kernels(
        K1_te,
        K2_te,
        alpha
    )

    # Step 5
    V_test = K_te.dot(W_kr)

    # Step 6
    Y_pred = U.dot(V_test.T)

    return Y_pred



class VKRNMFModel:

    def __init__(
        self,
        sigma=1.0,
        lam_kr=0.1,
        lam_nmf=0.1,
        alpha=0.5,
        rank=50
    ):
        self.sigma = sigma
        self.lam_kr = lam_kr
        self.lam_nmf = lam_nmf
        self.alpha = alpha
        self.rank = rank

        self.is_fitted = False

    def fit(
        self,
        dgi_train,
        fp_train,
        adr_train
    ):
       
        self.X_dgi_train = dgi_train
        self.X_chem_train = fp_train
        self.Y_train = adr_train.T

        self.is_fitted = True

    def predict(self,dgi_test, fp_test):
        dgi_test = np.array(dgi_test)
        fp_test = np.array(fp_test)
        
        if not self.is_fitted:
            raise ValueError(
                "Model not fitted. Call fit() first."
            )

        y_pred = _vkr_predict(
            self.X_dgi_train,
            self.X_chem_train,
            self.Y_train,
            dgi_test.reshape(1, -1),
            fp_test.reshape(1, -1),
            self.sigma,
            self.lam_kr,
            self.lam_nmf,
            self.alpha,
            self.rank
        )

        return y_pred.flatten()