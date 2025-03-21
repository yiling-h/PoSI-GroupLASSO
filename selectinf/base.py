from typing import NamedTuple

import numpy as np

import regreg.api as rr
import regreg.affine as ra

def restricted_estimator(loss, active, solve_args={'min_its':50, 'tol':1.e-10}):
    """
    Fit a restricted model using only columns `active`.

    Parameters
    ----------

    Mest_loss : objective function
        A GLM loss.

    active : ndarray
        Which columns to use.

    solve_args : dict
        Passed to `solve`.

    Returns
    -------

    soln : ndarray
        Solution to restricted problem.

    """
    X, Y = loss.data

    if not loss._is_transform and hasattr(loss, 'saturated_loss'): # M_est is a glm
        X_restricted = X[:,active]
        loss_restricted = rr.affine_smooth(loss.saturated_loss, X_restricted)
    else:
        I_restricted = ra.selector(active, ra.astransform(X).input_shape[0], ra.identity((active.sum(),)))
        loss_restricted = rr.affine_smooth(loss, I_restricted.T)
    beta_E = loss_restricted.solve(**solve_args)
    
    return beta_E


# functions construct targets of inference
# and covariance with score representation

class TargetSpec(NamedTuple):
    
    observed_target : np.ndarray
    cov_target : np.ndarray
    regress_target_score : np.ndarray
    alternatives : list
    
def selected_targets(loglike, 
                     solution,
                     features=None,
                     sign_info={}, 
                     dispersion=None,
                     solve_args={'tol': 1.e-12, 'min_its': 100},
                     hessian=None):

    if features is None:
        features = solution != 0

    X, y = loglike.data
    n, p = X.shape

    # OLS solution: \hat\beta_S = (X_E'X_E)^-1 X_E Y
    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    linpred = X[:, features].dot(observed_target)

    # Hfeat = _hessian_active = X'X_E
    Hfeat = _compute_hessian(loglike,
                             solution,
                             features)[1]
    # Qfeat = X_E'X_E
    Qfeat = Hfeat[features]
    _score_linear = -Hfeat

    """def pi_hess(x):
        return np.exp(x) / (1 + np.exp(x)) ** 2
    print("Qfeat rank:", np.linalg.matrix_rank(Qfeat))
    print("Qfeat dim:", Qfeat.shape[1])
    print("Xfeat rank:", np.linalg.matrix_rank(X[:, features]))
    print("Xfeat dim:", X[:, features].shape[1])

    if np.linalg.matrix_rank(Qfeat) != Qfeat.shape[1]:
        print(Qfeat)
        print(Hfeat)
        print("Hessian:", pi_hess(linpred))"""

    # cov_target: (X_E'X_E)^-1 = covariance of selected OLS estimator hat{beta_S}
    cov_target = np.linalg.inv(Qfeat)
    crosscov_target_score = _score_linear.dot(cov_target)
    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]

    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    if dispersion is None:  # use Pearson's X^2
        dispersion = _pearsonX2(y,
                                linpred,
                                loglike,
                                observed_target.shape[0])

    regress_target_score = np.zeros((cov_target.shape[0], p))
    # regress_target_score = [ (X_E'X_E)^-1  0_{-E} ]
    regress_target_score[:,features] = cov_target

    return TargetSpec(observed_target,
                      cov_target * dispersion,
                      regress_target_score,
                      alternatives)

def selected_targets_quasi(loglike,
                           solution,
                           cov_score,
                           features=None,
                           sign_info={},
                           dispersion=None,
                           solve_args={'tol': 1.e-12, 'min_its': 100},
                           hessian=None):
    """
    cov_score: the K matrix, estimated with the selected model
    loglike: log-likelihood object with the full X, Y
    solution: solution to the randomized objective
    """

    if features is None:
        features = solution != 0

    X, y = loglike.data
    n, p = X.shape

    # OLS solution: \hat\beta_S = (X_E'X_E)^-1 X_E Y
    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    linpred = X[:, features].dot(observed_target)

    # Hfeat = _hessian_active = H_{:,E}
    Hfeat = _compute_hessian(loglike,
                             solution,
                             features)[1]
    # Qfeat = H_{E,E}
    Qfeat = Hfeat[features]
    Qinv = np.linalg.inv(Qfeat)
    #print("Qinv shape", Qinv.shape)

    # Kfeat = K_{E,E}
    Kfeat = cov_score[:,features]
    Kfeat = Kfeat[features,:]
    #print("Kfeat shape", Kfeat.shape)

    # cov_target: \Sigma_E
    cov_target = Qinv @ Kfeat @ Qinv
    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]

    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    if dispersion is None:  # use Pearson's X^2
        dispersion = _pearsonX2(y,
                                linpred,
                                loglike,
                                observed_target.shape[0])

    regress_target_score = np.zeros((Qinv.shape[0], p))
    # regress_target_score = [ (X_E'X_E)^-1  0_{-E} ]
    regress_target_score[:,features] = Qinv

    return TargetSpec(observed_target,
                      cov_target * dispersion,
                      regress_target_score,
                      alternatives)

def full_targets_quasi(loglike,
                       solution,
                       cov_score,
                       features=None,
                       sign_info={},
                       dispersion=None,
                       solve_args={'tol': 1.e-12, 'min_its': 100},
                       hessian=None):
    """
    cov_score: the K matrix, estimated with the selected model
    loglike: log-likelihood object with the full X, Y
    solution: solution to the randomized objective
    """

    if features is None:
        features = solution != 0

    X, y = loglike.data
    n, p = X.shape

    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    full_target = loglike.solve(**solve_args)
    linpred = X.dot(full_target)

    # Hfeat = _hessian_active = H_{:,E}
    Hfeat = _compute_hessian(loglike,
                             full_target,
                             features)[1]
    # Qfeat = H_{E,E}
    Qfeat = Hfeat[features]
    Qinv = np.linalg.inv(Qfeat)
    #print("Qinv shape", Qinv.shape)

    # Kfeat = K_{E,E}
    Kfeat = cov_score[np.ix_(features, features)]
    #print("Kfeat shape", Kfeat.shape)

    print("H norm: ", np.linalg.norm(Qfeat, 'fro'))
    print("K norm: ", np.linalg.norm(Kfeat, 'fro'))
    print("H-K norm: ", np.linalg.norm(Qfeat - Kfeat, 'fro'))

    # cov_target: \Sigma_E
    cov_target = Qinv @ Kfeat @ Qinv
    print("Sigma_E norm: ", np.linalg.norm(cov_target, 'fro'))
    print("H^{-1} norm: ", np.linalg.norm(Qinv, 'fro'))
    print("H^{-1}-Sigma_E norm: ", np.linalg.norm(cov_target - Qinv, 'fro'))
    #print("Cov target shape", cov_target.shape)
    alternatives = ['twosided'] * features.sum()
    features_idx = np.arange(p)[features]

    for i in range(len(alternatives)):
        if features_idx[i] in sign_info.keys():
            alternatives[i] = sign_info[features_idx[i]]

    if dispersion is None:  # use Pearson's X^2
        dispersion = _pearsonX2(y,
                                linpred,
                                loglike,
                                observed_target.shape[0])

    regress_target_score = np.zeros((Qinv.shape[0], p))
    # regress_target_score = [ (X_E'X_E)^-1  0_{-E} ]
    regress_target_score[:,features] = Qinv

    return TargetSpec(observed_target,
                      cov_target * dispersion,
                      regress_target_score,
                      alternatives)

def full_targets(loglike, 
                 solution,
                 features=None,
                 dispersion=None,
                 solve_args={'tol': 1.e-12, 'min_its': 50},
                 hessian=None):
    
    if features is None:
        features = solution != 0

    X, y = loglike.data
    n, p = X.shape
    features_bool = np.zeros(p, np.bool)
    features_bool[features] = True
    features = features_bool

    # target is one-step estimator

    # Solve inherited from env3/lib/python3.8/site-packages/regreg/problems/composite.py
    full_estimator = loglike.solve(**solve_args)
    linpred = X.dot(full_estimator)
    Qfull = _compute_hessian(loglike,
                             full_estimator)

    Qfull_inv = np.linalg.inv(Qfull)
    cov_target = Qfull_inv[features][:, features]
    observed_target = full_estimator[features]
    crosscov_target_score = np.zeros((p, cov_target.shape[0]))
    crosscov_target_score[features] = -np.identity(cov_target.shape[0])

    if dispersion is None:  # use Pearson's X^2
        dispersion = _pearsonX2(y,
                                linpred,
                                loglike,
                                p)

    alternatives = ['twosided'] * features.sum()
    regress_target_score = Qfull_inv[features] # weights missing?

    return TargetSpec(observed_target,
                      cov_target * dispersion,
                      regress_target_score,
                      alternatives)

def form_targets(target, 
                 loglike, 
                 solution,
                 features, 
                 **kwargs):
    _target = {'full':full_targets,
               'selected':selected_targets}[target]
    return _target(loglike,
                   solution,
                   features,
                   **kwargs)

def _compute_hessian(loglike,
                     beta_bar,
                     *bool_indices):

    X, y = loglike.data
    linpred = X.dot(beta_bar)
    n = linpred.shape[0]

    if hasattr(loglike.saturated_loss, "hessian"): # a GLM -- all we need is W
        # W is all ones for the lasso
        W = loglike.saturated_loss.hessian(linpred)
        # Active idx, then unpenalized idx
        parts = [np.dot(X.T, X[:, bool_idx] * W[:, None]) for bool_idx in bool_indices]
        _hessian = np.dot(X.T, X * W[:, None]) # CAREFUL -- this will be big
    elif hasattr(loglike.saturated_loss, "hessian_mult"):
        parts = []
        for bool_idx in bool_indices:
            _right = np.zeros((n, bool_idx.sum()))
            for i, j in enumerate(np.nonzero(bool_idx)[0]):
                _right[:,i] = loglike.saturated_loss.hessian_mult(linpred, 
                                                                       X[:,j], 
                                                                       case_weights=loglike.saturated_loss.case_weights)
            parts.append(X.T.dot(_right))
        _hessian = np.zeros_like(X)
        for i in range(X.shape[1]):
            _hessian[:,i] = loglike.saturated_loss.hessian_mult(linpred, 
                                                                     X[:,i], 
                                                                     case_weights=loglike.saturated_loss.case_weights)
        _hessian = X.T.dot(_hessian)
    else:
        raise ValueError('saturated_loss has no hessian or hessian_mult method')

    if bool_indices:
        # Returns _hessian, _hessian_active, _hessian_unpen
        return (_hessian,) + tuple(parts)
    else:
        return _hessian

def _pearsonX2(y,
               linpred,
               loglike,
               df_fit):

    W = loglike.saturated_loss.hessian(linpred)
    n = y.shape[0]
    resid = y - loglike.saturated_loss.mean_function(linpred)
    return (resid ** 2 / W).sum() / (n - df_fit)

def target_query_Interactspec(query_spec,
                              regress_target_score,
                              cov_target):

    QS = query_spec
    prec_target = np.linalg.inv(cov_target)

    U1 = regress_target_score.T.dot(prec_target)
    U2 = U1.T.dot(QS.M2.dot(U1))
    U3 = U1.T.dot(QS.M3.dot(U1))
    U4 = QS.M1.dot(QS.opt_linear).dot(QS.cond_cov).dot(QS.opt_linear.T.dot(QS.M1.T.dot(U1)))
    U5 = U1.T.dot(QS.M1.dot(QS.opt_linear))

    return U1, U2, U3, U4, U5
