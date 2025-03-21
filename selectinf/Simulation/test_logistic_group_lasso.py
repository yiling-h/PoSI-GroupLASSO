import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import multiprocessing
# from multiprocess import Pool

import regreg.api as rr

from selectinf.group_lasso_query import (group_lasso,
                                         split_group_lasso)

from selectinf.base import selected_targets
from selectinf.Simulation.instance import (logistic_group_instance)

from selectinf.base import restricted_estimator
import scipy.stats
from scipy.stats import norm
from scipy.linalg import qr

def rank_deficiency_qr(A, tol=None):
    # Perform QR decomposition with column pivoting
    Q, R, P = qr(A, mode='economic', pivoting=True)
    # If tolerance is not provided, use a default one based on numerical precision
    if tol is None:
        tol = np.max(A.shape) * np.abs(np.diag(R)).max() * np.finfo(R.dtype).eps
    # Rank is determined by the number of diagonal elements larger than the tolerance
    rank = np.sum(np.abs(np.diag(R)) > tol)
    # Rank deficiency is the difference between the matrix size and its rank
    rank_deficiency = A.shape[1] - rank
    return rank_deficiency


def calculate_F1_score(beta_true, selection):
    p = len(beta_true)
    nonzero_true = (beta_true != 0)

    # precision & recall
    if selection.sum() > 0:
        precision = (nonzero_true * selection).sum() / selection.sum()
    else:
        precision = 0
    recall = (nonzero_true * selection).sum() / nonzero_true.sum()
    print("precision:", precision, "recall", recall)
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0

def naive_inference(X, Y, groups, beta, const,
                    n, weight_frac=1, level=0.9, p_val=False):

    p = X.shape[1]
    sigma_ = np.std(Y)
    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
    print("Naive l1 weights:", weights)

    conv = const(X=X,
                 successes=Y,
                 trials=np.ones(n),
                 groups=groups,
                 weights=weights,
                 useJacobian=True,
                 perturb=np.zeros(p),
                 ridge_term=0.)

    signs, _ = conv.fit()
    nonzero = signs != 0

    # print('Naive selection', conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():
        def pi(x):
            return 1 / (1 + np.exp(-x))

        Y_mean = pi(X.dot(beta))

        loglike_Mean = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike_Mean,
                                                 nonzero)
        return _beta_unpenalized

    target = solve_target_restricted()

    if nonzero.sum() > 0:
        # E: nonzero flag

        X_E = X[:, nonzero]

        def pi_hess(x):
            return np.exp(x) / (1 + np.exp(x)) ** 2

        loglike = rr.glm.logistic(X, successes=Y, trials=np.ones(n))
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE = restricted_estimator(loglike, nonzero)

        # Calculation the asymptotic covariance of the MLE
        W = np.diag(pi_hess(X_E @ beta_MLE))

        f_info = X_E.T @ W @ X_E

        if np.isnan(f_info).any():
            if p_val:
                # If no variable selected, no inference
                return None, None, None, None, None, None, None
            else:
                return None, None, None, None, None, None

        if rank_deficiency_qr(f_info):
            if p_val:
                # If no variable selected, no inference
                return None, None, None, None, None, None, None
            else:
                return None, None, None, None, None, None

        cov = np.linalg.inv(f_info)

        # Standard errors
        sd = np.sqrt(np.diag(cov))

        # Normal quantiles
        z_low = scipy.stats.norm.ppf((1 - level) / 2)
        z_up = scipy.stats.norm.ppf(1 - (1 - level) / 2)
        assert np.abs(np.abs(z_low) - np.abs(z_up)) < 10e-6

        # Construct confidence intervals
        intervals_low = beta_MLE + z_low * sd
        intervals_up = beta_MLE + z_up * sd

        coverage = (target > intervals_low) * (target < intervals_up)

        p_vals = 2 * np.min([norm.cdf(beta_MLE / sd),
                             1 - norm.cdf(beta_MLE / sd)], axis=0)

        if p_val:
            return (coverage, intervals_up - intervals_low, nonzero,
                    intervals_low, intervals_up, target, p_vals)
        else:
            return (coverage, intervals_up - intervals_low, nonzero,
                    intervals_low, intervals_up, target)
    if p_val:
        # If no variable selected, no inference
        return None, None, None, None, None, None, None
    else:
        return None, None, None, None, None, None

def randomization_inference(X, Y, n, p, beta,
                            groups, hess=None, proportion=0.5,
                            weight_frac=1, level=0.9, solve_only = False,
                            p_val=False):

    ## solve_only: bool variable indicating whether
    ##              1) we only need the solver's output
    ##              or
    ##              2) we also want inferential results

    def estimate_hess():
        loglike = rr.glm.logistic(X, successes=Y, trials=np.ones(n))
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_full = restricted_estimator(loglike, np.array([True] * p))
        def pi_hess(x):
            return np.exp(x) / (1 + np.exp(x)) ** 2

        # Calculation the asymptotic covariance of the MLE
        W = np.diag(pi_hess(X @ beta_full))

        return X.T @ W @ X * (1 - proportion) / proportion

    if hess is None:
        hess = estimate_hess()

    sigma_ = np.std(Y)
    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = group_lasso.logistic(X=X,
                                successes=Y,
                                trials=np.ones(n),
                                groups=groups,
                                weights=weights,
                                useJacobian=True,
                                ridge_term=0.,
                                cov_rand=hess)

    signs, _ = conv.fit()
    nonzero = (signs != 0)

    #print("MLE selection:", conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():
        def pi(x):
            return 1 / (1 + np.exp(-x))

        Y_mean = pi(X.dot(beta))

        loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    # Return the selected variables if we only want to solve the problem
    if solve_only:
        return None,None,solve_target_restricted(),nonzero,None,None

    if nonzero.sum() > 0:
        conv.setup_inference(dispersion=1)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=1)

        result,_ = conv.inference(target_spec,
                                method='selective_MLE',
                                level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        if p_val:
            return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
                   nonzero, intervals[:, 0], intervals[:, 1], pval
        else:
            return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
                nonzero, intervals[:, 0], intervals[:, 1]
    if p_val:
        return None, None, None, None, None, None, None
    else:
        return None, None, None, None, None, None

def randomization_inference_fast(X, Y, n, p, beta, groups, proportion = 0.5,
                                 hess=None, weight_frac=1, level=0.9,
                                 p_val=False):

    ## Use split group lasso to solve the hessian-randomized MLE problem efficiently
    ## Selection is consistent with the MLE method with the hessian randomization covariance
    ## but inference is carried out as if data carving were intended
    ## The two inference approaches are asymptotically the same
    def estimate_hess():
        loglike = rr.glm.logistic(X, successes=Y, trials=np.ones(n))
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_full = restricted_estimator(loglike, np.array([True] * p))
        def pi_hess(x):
            return np.exp(x) / (1 + np.exp(x)) ** 2

        # Calculation the asymptotic covariance of the MLE
        W = np.diag(pi_hess(X @ beta_full))

        return X.T @ W @ X * (1 - proportion) / proportion

    if hess is None:
        hess = estimate_hess()

    sigma_ = np.std(Y)
    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac / np.sqrt(proportion) * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
    print("MLE l1 weights:", weights)

    conv = split_group_lasso.logistic(X=X,
                                      successes=Y,
                                      trials=np.ones(n),
                                      groups=groups,
                                      weights=weights,
                                      useJacobian=True,
                                      proportion=proportion,
                                      cov_rand=hess,
                                      )

    signs, _ = conv.fit()
    nonzero = (signs != 0)

    # print("MLE selection:", conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():
        def pi(x):
            return 1 / (1 + np.exp(-x))

        Y_mean = pi(X.dot(beta))

        loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    if nonzero.sum() > 0:
        conv.setup_inference(dispersion=1)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=1)

        result,_ = conv.inference(target_spec,
                                method='selective_MLE',
                                level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        if p_val:
            return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
                nonzero, intervals[:, 0], intervals[:, 1], pval
        else:
            return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
                nonzero, intervals[:, 0], intervals[:, 1]

    if p_val:
        return None, None, None, None, None, None, None
    else:
        return None, None, None, None, None, None

def split_inference(X, Y, n, p, beta, groups, const,
                    weight_frac=1, proportion=0.5, level=0.9):

    ## selective inference with data carving

    sigma_ = np.std(Y)
    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = const(X=X,
                 successes=Y,
                 groups=groups,
                 weights=weights,
                 proportion=proportion, # proportion of data used for selection (n1/n)
                 useJacobian=True)

    signs, _ = conv.fit()
    nonzero = signs != 0

    print("Carving selection", conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():
        def pi(x):
            return 1 / (1 + np.exp(-x))

        Y_mean = pi(X.dot(beta))

        loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    if nonzero.sum() > 0:
        conv.setup_inference(dispersion=1)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=1)

        result,_ = conv.inference(target_spec,
                                'selective_MLE',
                                level=level)
        estimate = result['MLE']
        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        hess = ((1 - proportion) / proportion) * conv._unscaled_cov_score  # hessian

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
               nonzero, conv._selection_idx, hess, intervals[:, 0], intervals[:, 1]

    return None, None, None, None, None, None, None, None

def data_splitting(X, Y, n, p, beta, groups, proportion=0.5, weight_frac=1,
                   nonzero=None, subset_select=None, level=0.9, p_val=False,
                   count_rank_deficiency=False):

    if (nonzero is None) or (subset_select is None):
        # print("(Poisson Data Splitting) Selection done without carving")
        pi_s = proportion
        subset_select = np.zeros(n, np.bool)
        subset_select[:int(pi_s * n)] = True
        n1 = subset_select.sum()
        n2 = n - n1
        np.random.shuffle(subset_select)
        X_S = X[subset_select, :]
        Y_S = Y[subset_select]

        # Selection on the first subset of data
        p = X.shape[1]
        sigma_ = np.std(Y_S)
        # weights = dict([(i, 0.5) for i in np.unique(groups)])
        weights = dict([(i, np.sqrt(proportion) * weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
        #print("DS l1 weights:", weights)

        conv = group_lasso.logistic(X=X_S,
                                    successes=Y_S,
                                    trials=np.ones(n1),
                                    groups=groups,
                                    weights=weights,
                                    useJacobian=True,
                                    perturb=np.zeros(p),
                                    ridge_term=0.)

        signs, _ = conv.fit(solve_only=True)
        # print("signs",  signs)
        nonzero = signs != 0

    n1 = subset_select.sum()
    n2 = n - n1

    if nonzero.sum() > 0:
        # Solving the inferential target
        def solve_target_restricted():
            def pi(x):
                return 1 / (1 + np.exp(-x))

            Y_mean = pi(X.dot(beta))

            loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
            # For LASSO, this is the OLS solution on X_{E,U}
            _beta_unpenalized = restricted_estimator(loglike,
                                                     nonzero)
            return _beta_unpenalized

        target = solve_target_restricted()

        X_notS = X[~subset_select, :]
        Y_notS = Y[~subset_select]

        # E: nonzero flag

        X_notS_E = X_notS[:, nonzero]

        # Solve for the unpenalized MLE
        def pi_hess(x):
            return np.exp(x) / (1 + np.exp(x)) ** 2

        loglike = rr.glm.logistic(X_notS, successes=Y_notS, trials=np.ones(n2))
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE_notS = restricted_estimator(loglike, nonzero)

        # Calculation the asymptotic covariance of the MLE
        W = np.diag(pi_hess(X_notS_E @ beta_MLE_notS))

        f_info = X_notS_E.T @ W @ X_notS_E

        if np.isnan(f_info).any():
            if p_val:
                if count_rank_deficiency:
                    return True, None, None, None, None, nonzero, None, None
                # If no variable selected, no inference
                return None, None, None, None, nonzero, None, None
            else:
                if count_rank_deficiency:
                    return True, None, None, None, None, nonzero, None
                return None, None, None, None, nonzero, None

        if rank_deficiency_qr(f_info):
            if p_val:
                if count_rank_deficiency:
                    return True, None, None, None, None, nonzero, None, None
                # If no variable selected, no inference
                return None, None, None, None, nonzero, None, None
            else:
                if count_rank_deficiency:
                    return True, None, None, None, None, nonzero, None
                return None, None, None, None, nonzero, None
        cov = np.linalg.inv(f_info)

        # Standard errors
        sd = np.sqrt(np.diag(cov))

        # Normal quantiles
        z_low = scipy.stats.norm.ppf((1 - level) / 2)
        z_up = scipy.stats.norm.ppf(1 - (1 - level) / 2)
        assert np.abs(np.abs(z_low) - np.abs(z_up)) < 10e-6

        # Construct confidence intervals
        intervals_low = beta_MLE_notS + z_low * sd
        intervals_up = beta_MLE_notS + z_up * sd

        coverage = (target > intervals_low) * (target < intervals_up)

        p_vals = 2 * np.min([norm.cdf(beta_MLE_notS/sd),
                             1-norm.cdf(beta_MLE_notS/sd)], axis=0)

        if p_val:
            if count_rank_deficiency:
                return (False, coverage, intervals_up - intervals_low,
                        intervals_low, intervals_up, nonzero, target, p_vals)
            return (coverage, intervals_up - intervals_low,
                    intervals_low, intervals_up, nonzero, target, p_vals)
        else:
            if count_rank_deficiency:
                return (False, coverage, intervals_up - intervals_low,
                        intervals_low, intervals_up, nonzero, target)
            return (coverage, intervals_up - intervals_low,
                    intervals_low, intervals_up, nonzero, target)
    if p_val:
        if count_rank_deficiency:
            # If no variable selected, no inference
            return False, None, None, None, None, nonzero, None, None
        else:
            return None, None, None, None, nonzero, None, None
    else:
        if count_rank_deficiency:
            return False, None, None, None, None, nonzero, None
        return None, None, None, None, nonzero, None

def test_comparison_logistic_group_lasso(n=500,
                                         p=200,
                                         signal_fac=0.1,
                                         s=5,
                                         rho=0.3,
                                         randomizer_scale=1.,
                                         level=0.90,
                                         iter=10):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["beta size"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    #oper_char["runtime"] = []

    confint_df = pd.DataFrame()

    for signal_fac in [0.01, 0.03, 0.06, 0.1]: #[0.01, 0.03, 0.06, 0.1]:
        for i in range(iter):

            #np.random.seed(i)

            inst  = logistic_group_instance
            const = group_lasso.logistic
            const_split = split_group_lasso.logistic

            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal,decimals=2))

            while True:  # run until we get some selection
                groups = np.arange(50).repeat(4)
                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  sgroup=s,
                                  groups=groups,
                                  ndiscrete=4,
                                  nlevels=5,
                                  sdiscrete=2, # How many discrete rvs are not null
                                  equicorrelated=False,
                                  rho=rho,
                                  random_signs=True)[:3]

                n, p = X.shape

                noselection = False    # flag for a certain method having an empty selected set

                if not noselection:
                    # carving
                    coverage_s, length_s, beta_target_s, nonzero_s, \
                    selection_idx_s, hessian, conf_low_s, conf_up_s = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.5)
                    noselection = (coverage_s is None)

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=X, Y=Y, n=n, p=p, proportion=0.5,
                                                     beta=beta, groups=groups, hess=hessian)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, nonzero=nonzero_s,
                                       subset_select=selection_idx_s, level=0.9)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive,\
                        beta_target_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level)
                    noselection = (coverage_naive is None)

                if not noselection:
                    # F1 scores
                    F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_s)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)

                    # Hessian MLE coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')
                    df_MLE = pd.concat([pd.DataFrame(np.ones(nonzero.sum())*i),
                                        pd.DataFrame(beta_target),
                                        pd.DataFrame(conf_low),
                                        pd.DataFrame(conf_up),
                                        pd.DataFrame(beta[nonzero] != 0),
                                        pd.DataFrame([signal_str] * nonzero.sum()),
                                        pd.DataFrame(np.ones(nonzero.sum()) * F1),
                                        pd.DataFrame(["MLE"] * nonzero.sum())
                                        ], axis=1)

                    confint_df = pd.concat([confint_df, df_MLE], axis=0)
                    #oper_char["runtime"].append(0)

                    # Carving coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_s))
                    oper_char["avg length"].append(np.mean(length_s))
                    oper_char["F1 score"].append(F1_s)
                    oper_char["method"].append('Carving')
                    df_s = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                        pd.DataFrame(beta_target_s),
                                        pd.DataFrame(conf_low_s),
                                        pd.DataFrame(conf_up_s),
                                        pd.DataFrame(beta[nonzero_s] != 0),
                                        pd.DataFrame([signal_str] * nonzero_s.sum()),
                                        pd.DataFrame(np.ones(nonzero_s.sum()) * F1_s),
                                        pd.DataFrame(["Carving"] * nonzero_s.sum())
                                        ], axis=1)
                    confint_df = pd.concat([confint_df, df_s], axis=0)

                    # Data splitting coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')
                    df_ds = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                      pd.DataFrame(beta_target_s),
                                      pd.DataFrame(conf_low_ds),
                                      pd.DataFrame(conf_up_ds),
                                      pd.DataFrame(beta[nonzero_s] != 0),
                                      pd.DataFrame([signal_str] * nonzero_s.sum()),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * F1_ds),
                                      pd.DataFrame(["Data splitting"] * nonzero_s.sum())
                                      ], axis=1)
                    confint_df = pd.concat([confint_df, df_ds], axis=0)

                    # Naive coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')
                    df_naive = pd.concat([pd.DataFrame(np.ones(nonzero_naive.sum()) * i),
                                       pd.DataFrame(beta_target_naive),
                                       pd.DataFrame(conf_low_naive),
                                       pd.DataFrame(conf_up_naive),
                                       pd.DataFrame(beta[nonzero_naive] != 0),
                                       pd.DataFrame([signal_str] * nonzero_naive.sum()),
                                       pd.DataFrame(np.ones(nonzero_naive.sum()) * F1_naive),
                                       pd.DataFrame(["Naive"] * nonzero_naive.sum())
                                       ], axis=1)
                    confint_df = pd.concat([confint_df, df_naive], axis=0)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('selectinf/randomized/Tests/logis_vary_signal.csv', index=False)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['beta size'] + ['F1'] + ['Method']
    confint_df.columns = colnames
    confint_df.to_csv('selectinf/randomized/Tests/logis_CI_vary_signal.csv', index=False)

    #sns.histplot(oper_char_df["beta size"])
    #plt.show()

    print("Mean coverage rate/length:")
    print(oper_char_df.groupby(['beta size', 'method']).mean())

    #cov_plot = \
    sns.boxplot(y=oper_char_df["coverage rate"],
                x=oper_char_df["beta size"],
                hue=oper_char_df["method"],
                showmeans=True,
                orient="v")
    plt.show()

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                           x=oper_char_df["beta size"],
                           hue=oper_char_df["method"],
                           showmeans=True,
                           orient="v")
    len_plot.set_ylim(5,17)
    plt.show()

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                           x=oper_char_df["beta size"],
                           hue=oper_char_df["method"],
                           showmeans=True,
                           orient="v")
    F1_plot.set_ylim(0, 1)
    plt.show()


def test_comparison_logistic_lasso_vary_s(n=500,
                                           p=200,
                                           signal_fac=0.1,
                                           s=5,
                                           sigma=2,
                                           rho=0.3,
                                           randomizer_scale=1.,
                                           full_dispersion=True,
                                           level=0.90,
                                           iter=30):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["sparsity size"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    # oper_char["runtime"] = []

    confint_df = pd.DataFrame()

    for s in [5,8,10]: #[0.01, 0.03, 0.06, 0.1]:
        for i in range(iter):
            #np.random.seed(i)

            inst, const, const_split = logistic_group_instance, group_lasso.logistic, \
                                       split_group_lasso.logistic
            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal, decimals=2))

            while True:  # run until we get some selection
                groups = np.arange(50).repeat(4)
                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  sgroup=s,
                                  groups=groups,
                                  ndiscrete=20,
                                  nlevels=5,
                                  sdiscrete=s-3,#s-3, # How many discrete rvs are not null
                                  equicorrelated=False,
                                  rho=rho,
                                  random_signs=True)[:3]
                #print(X)

                n, p = X.shape

                noselection = False  # flag for a certain method having an empty selected set

                """if not noselection:
                    # carving
                    coverage_s, length_s, beta_target_s, nonzero_s, \
                    selection_idx_s, hessian, conf_low_s, conf_up_s = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.67)

                    noselection = (coverage_s is None)"""

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=X, Y=Y, n=n, p=p, proportion=0.67,
                                                     beta=beta, groups=groups)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds, nonzero_ds, beta_target_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, groups=groups,
                                       proportion=0.67, level=0.9)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive, \
                    beta_target_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level)
                    noselection = (coverage_naive is None)

                if not noselection:
                    # F1 scores
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_ds)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)

                    # MLE coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')
                    df_MLE = pd.concat([pd.DataFrame(np.ones(nonzero.sum()) * i),
                                        pd.DataFrame(beta_target),
                                        pd.DataFrame(conf_low),
                                        pd.DataFrame(conf_up),
                                        pd.DataFrame(beta[nonzero] != 0),
                                        pd.DataFrame(np.ones(nonzero.sum()) * s),
                                        pd.DataFrame(np.ones(nonzero.sum()) * F1),
                                        pd.DataFrame(["MLE"] * nonzero.sum())
                                        ], axis=1)
                    confint_df = pd.concat([confint_df, df_MLE], axis=0)


                    """# Carving coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_s))
                    oper_char["avg length"].append(np.mean(length_s))
                    oper_char["F1 score"].append(F1_s)
                    oper_char["method"].append('Carving')
                    #oper_char["runtime"].append(0)
                    df_s = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                      pd.DataFrame(beta_target_s),
                                      pd.DataFrame(conf_low_s),
                                      pd.DataFrame(conf_up_s),
                                      pd.DataFrame(beta[nonzero_s] != 0),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * s),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * F1_s),
                                      pd.DataFrame(["Carving"] * nonzero_s.sum())
                                      ], axis=1)
                    confint_df = pd.concat([confint_df, df_s], axis=0)"""
                    

                    # Data splitting coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')
                    #oper_char["runtime"].append(0)
                    df_ds = pd.concat([pd.DataFrame(np.ones(nonzero_ds.sum()) * i),
                                       pd.DataFrame(beta_target_ds),
                                       pd.DataFrame(conf_low_ds),
                                       pd.DataFrame(conf_up_ds),
                                       pd.DataFrame(beta[nonzero_ds] != 0),
                                       pd.DataFrame(np.ones(nonzero_ds.sum()) * s),
                                       pd.DataFrame(np.ones(nonzero_ds.sum()) * F1_ds),
                                       pd.DataFrame(["Data splitting"] * nonzero_ds.sum())
                                       ], axis=1)
                    confint_df = pd.concat([confint_df, df_ds], axis=0)

                    # Naive coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')
                    df_naive = pd.concat([pd.DataFrame(np.ones(nonzero_naive.sum()) * i),
                                          pd.DataFrame(beta_target_naive),
                                          pd.DataFrame(conf_low_naive),
                                          pd.DataFrame(conf_up_naive),
                                          pd.DataFrame(beta[nonzero_naive] != 0),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * s),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * F1_naive),
                                          pd.DataFrame(["Naive"] * nonzero_naive.sum())
                                          ], axis=1)
                    confint_df = pd.concat([confint_df, df_naive], axis=0)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('selectinf/randomized/Tests/logis_vary_sparsity.csv',index=False)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['sparsity size'] + ['F1'] + ['Method']
    confint_df.columns = colnames
    confint_df.to_csv('selectinf/randomized/Tests/logis_CI_vary_sparsity.csv', index=False)

    #sns.histplot(oper_char_df["sparsity size"])
    #plt.show()

    """print("Mean coverage rate/length:")
    print(oper_char_df.groupby(['sparsity size', 'method']).mean())

    sns.boxplot(y=oper_char_df["coverage rate"],
                x=oper_char_df["sparsity size"],
                hue=oper_char_df["method"],
                orient="v")
    plt.show()

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                           x=oper_char_df["sparsity size"],
                           hue=oper_char_df["method"],
                           showmeans=True,
                           orient="v")
    len_plot.set_ylim(5,17)
    plt.show()

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                          x=oper_char_df["sparsity size"],
                          hue=oper_char_df["method"],
                          showmeans=True,
                          orient="v")
    F1_plot.set_ylim(0, 1)
    plt.show()"""

def comparison_logistic(range):
    """
        Compare to R randomized lasso
        """
    n = 500
    p = 200
    signal_fac = 0.1
    sigma = 2
    rho = 0.5
    randomizer_scale = 1.
    full_dispersion = True
    level = 0.90

    # Operating characteristics
    oper_char = {}
    oper_char["sparsity size"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    # oper_char["runtime"] = []

    confint_df = pd.DataFrame()

    for s in [5, 8, 10]:  # [0.01, 0.03, 0.06, 0.1]:
        for i in range:
            # np.random.seed(i)

            inst, const, const_split = logistic_group_instance, group_lasso.logistic, \
                                       split_group_lasso.logistic
            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal, decimals=2))

            while True:  # run until we get some selection
                groups = np.arange(50).repeat(4)
                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  sgroup=s,
                                  groups=groups,
                                  ndiscrete=20,
                                  nlevels=5,
                                  sdiscrete=s - 3,  # s-3, # How many discrete rvs are not null
                                  equicorrelated=False,
                                  rho=rho,
                                  random_signs=True)[:3]
                # print(X)

                n, p = X.shape

                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    # carving
                    coverage_s, length_s, beta_target_s, nonzero_s, \
                    selection_idx_s, hessian, conf_low_s, conf_up_s = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.67)

                    noselection = (coverage_s is None)

                if not noselection:
                    # MLE inference
                    start = time.perf_counter()
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=X, Y=Y, n=n, p=p, proportion=0.67,
                                                     beta=beta, groups=groups, hess=hessian)
                    end = time.perf_counter()
                    MLE_runtime = end - start
                    # print(MLE_runtime)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, nonzero=nonzero_s,
                                       subset_select=selection_idx_s, level=0.9)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive, \
                    beta_target_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level)
                    noselection = (coverage_naive is None)

                if not noselection:
                    # F1 scores
                    F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_s)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)

                    # MLE coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')
                    df_MLE = pd.concat([pd.DataFrame(np.ones(nonzero.sum()) * i),
                                        pd.DataFrame(beta_target),
                                        pd.DataFrame(conf_low),
                                        pd.DataFrame(conf_up),
                                        pd.DataFrame(beta[nonzero] != 0),
                                        pd.DataFrame(np.ones(nonzero.sum()) * s),
                                        pd.DataFrame(np.ones(nonzero.sum()) * F1),
                                        pd.DataFrame(["MLE"] * nonzero.sum())
                                        ], axis=1)
                    confint_df = pd.concat([confint_df, df_MLE], axis=0)

                    # Carving coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_s))
                    oper_char["avg length"].append(np.mean(length_s))
                    oper_char["F1 score"].append(F1_s)
                    oper_char["method"].append('Carving')
                    # oper_char["runtime"].append(0)
                    df_s = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                      pd.DataFrame(beta_target_s),
                                      pd.DataFrame(conf_low_s),
                                      pd.DataFrame(conf_up_s),
                                      pd.DataFrame(beta[nonzero_s] != 0),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * s),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * F1_s),
                                      pd.DataFrame(["Carving"] * nonzero_s.sum())
                                      ], axis=1)
                    confint_df = pd.concat([confint_df, df_s], axis=0)

                    # Data splitting coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')
                    # oper_char["runtime"].append(0)
                    df_ds = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                       pd.DataFrame(beta_target_s),
                                       pd.DataFrame(conf_low_ds),
                                       pd.DataFrame(conf_up_ds),
                                       pd.DataFrame(beta[nonzero_s] != 0),
                                       pd.DataFrame(np.ones(nonzero_s.sum()) * s),
                                       pd.DataFrame(np.ones(nonzero_s.sum()) * F1_ds),
                                       pd.DataFrame(["Data splitting"] * nonzero_s.sum())
                                       ], axis=1)
                    confint_df = pd.concat([confint_df, df_ds], axis=0)

                    # Naive coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')
                    df_naive = pd.concat([pd.DataFrame(np.ones(nonzero_naive.sum()) * i),
                                          pd.DataFrame(beta_target_naive),
                                          pd.DataFrame(conf_low_naive),
                                          pd.DataFrame(conf_up_naive),
                                          pd.DataFrame(beta[nonzero_naive] != 0),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * s),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * F1_naive),
                                          pd.DataFrame(["Naive"] * nonzero_naive.sum())
                                          ], axis=1)
                    confint_df = pd.concat([confint_df, df_naive], axis=0)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['sparsity size'] + ['F1'] + ['Method']
    confint_df.columns = colnames

    print("task done")
    return oper_char_df, confint_df

def test_comparison_logistic_group_lasso_vary_s_parallel(n=500,
                                                         p=200,
                                                         signal_fac=0.1,
                                                         s=5,
                                                         sigma=2,
                                                         rho=0.3,
                                                         randomizer_scale=1.,
                                                         full_dispersion=True,
                                                         level=0.90,
                                                         iter=2,
                                                         ncore=2):
    print(iter)
    print(ncore)
    def n_range_to_k(n, k):
        l = []
        for i in range(k):
            if i == 0:
                start = 0
                end = int(n / k)
            elif i == k - 1:
                start = end
                end = n
            else:
                start = end
                end = int((i + 1) * n / k)
            range_i = range(start, end)
            l.append(range_i)
        return l

    range_list = n_range_to_k(n=iter,k=ncore)

    print("ranges:", range_list)


    pool = multiprocessing.Pool(processes=ncore)
    pool_outputs = pool.map(comparison_logistic, range_list)

    """with Pool(ncore) as pool:
        pool_outputs = list(
            tqdm(
                pool.imap(comparison_logistic, range_list),
                total=len(range_list)
            )
        )"""

    oper_char_df = pd.DataFrame()
    confint_df = pd.DataFrame()

    for i in range(ncore):
        oper_char_df = pd.concat([oper_char_df, pool_outputs[i][0]], axis=0)
        confint_df = pd.concat([confint_df, pool_outputs[i][1]], axis=0)

    oper_char_df.to_csv('selectinf/randomized/Tests/logis_vary_sparsity.csv', index=False)
    confint_df.to_csv('selectinf/randomized/Tests/logis_CI_vary_sparsity.csv', index=False)

    def print_results(oper_char_df):
        print("Mean coverage rate/length:")
        print(oper_char_df.groupby(['sparsity size', 'method']).mean())

        sns.boxplot(y=oper_char_df["coverage rate"],
                    x=oper_char_df["sparsity size"],
                    hue=oper_char_df["method"],
                    orient="v")
        plt.show()

        len_plot = sns.boxplot(y=oper_char_df["avg length"],
                               x=oper_char_df["sparsity size"],
                               hue=oper_char_df["method"],
                               showmeans=True,
                               orient="v")
        len_plot.set_ylim(5, 17)
        plt.show()

        F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                              x=oper_char_df["sparsity size"],
                              hue=oper_char_df["method"],
                              showmeans=True,
                              orient="v")
        F1_plot.set_ylim(0, 1)
        plt.show()

    #print_results(oper_char_df)

def test_plotting(path='selectinf/randomized/Tests/logis_vary_sparsity.csv'):
    oper_char_df = pd.read_csv(path)
    # sns.histplot(oper_char_df["sparsity size"])
    # plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    print("Mean coverage rate/length:")
    print(oper_char_df.groupby(['sparsity size', 'method']).mean())

    cov_plot = sns.boxplot(y=oper_char_df["coverage rate"],
                           x=oper_char_df["sparsity size"],
                           hue=oper_char_df["method"],
                           palette="pastel",
                           orient="v", ax=ax1,
                           linewidth=1)
    cov_plot.set(title='Coverage')
    cov_plot.set_ylim(0.6, 1.05)
    # plt.tight_layout()
    cov_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
    # ax1.set_ylabel("")  # remove y label, but keep ticks

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                           x=oper_char_df["sparsity size"],
                           hue=oper_char_df["method"],
                           palette="pastel",
                           orient="v", ax=ax2,
                           linewidth=1)
    len_plot.set(title='Length')
    # len_plot.set_ylim(0, 100)
    # len_plot.set_ylim(0, 8)
    # plt.tight_layout()
    # ax2.set_ylabel("")  # remove y label, but keep ticks

    handles, labels = ax2.get_legend_handles_labels()
    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)
    fig.subplots_adjust(bottom=0.2)
    fig.legend(handles, labels, loc='lower center', ncol=4)

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                          x=oper_char_df["sparsity size"],
                          hue=oper_char_df["method"],
                          palette="pastel",
                          orient="v", ax=ax3,
                          linewidth=1)
    F1_plot.set(title='F1 score')

    cov_plot.legend_.remove()
    len_plot.legend_.remove()
    F1_plot.legend_.remove()

    plt.show()

def test_plotting_separate(path='selectinf/randomized/Tests/logis_vary_sparsity.csv'):
    oper_char_df = pd.read_csv(path)

    #sns.histplot(oper_char_df["sparsity size"])
    #plt.show()

    def plot_naive():
        naive_flag = oper_char_df["method"] == 'Naive'
        print(np.sum(naive_flag))

        print("Mean coverage rate/length:")
        print(oper_char_df.groupby(['sparsity size', 'method']).mean())

        cov_plot = sns.boxplot(y=oper_char_df.loc[naive_flag, "coverage rate"],
                               x=oper_char_df.loc[naive_flag, "beta size"],
                               # hue=oper_char_df["method"],
                               #palette="pastel",
                               color='lightcoral',
                               orient="v",
                               linewidth=1)
        cov_plot.set(title='Coverage of Naive Inference')
        cov_plot.set_ylim(0.5, 1.05)
        # plt.tight_layout()
        cov_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
        plt.show()

    def plot_comparison():
        cov_plot = sns.boxplot(y=oper_char_df["coverage rate"],
                               x=oper_char_df["sparsity size"],
                               hue=oper_char_df["method"],
                               palette="pastel",
                               orient="v",
                               linewidth=1)
        cov_plot.set(title='Coverage')
        cov_plot.set_ylim(0.5, 1.05)
        # plt.tight_layout()
        cov_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
        cov_plot.legend(loc='lower center', ncol=3)
        plt.tight_layout()

        """
        for i in [2,5,8,11]:
            mybox = cov_plot.artists[i]
            mybox.set_facecolor('lightcoral')
        """
        leg = cov_plot.get_legend()
        #leg.legendHandles[2].set_color('lightcoral')
        plt.show()

    def plot_len_comparison():
        len_plot = sns.boxplot(y=oper_char_df["avg length"],
                               x=oper_char_df["sparsity size"],
                               hue=oper_char_df["method"],
                               palette="pastel",
                               orient="v",
                               linewidth=1)
        len_plot.set(title='Length')
        # len_plot.set_ylim(0, 100)
        len_plot.legend(loc='lower center', ncol=3)
        len_plot.set_ylim(5, 20)
        plt.tight_layout()

        """
        for i in [2,5,8,11]:
            mybox = len_plot.artists[i]
            mybox.set_facecolor('lightcoral')
        """
        leg = len_plot.get_legend()
        #leg.legendHandles[2].set_color('lightcoral')
        plt.show()

    def plot_F1_comparison():
        F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                               x=oper_char_df["sparsity size"],
                               hue=oper_char_df["method"],
                               palette="pastel",
                               orient="v",
                               linewidth=1)
        F1_plot.set(title='F1 score')
        # len_plot.set_ylim(0, 100)
        F1_plot.legend(loc='lower center', ncol=3)
        F1_plot.set_ylim(0, 1)
        plt.tight_layout()

        """
        for i in [2,5,8,11]:
            mybox = len_plot.artists[i]
            mybox.set_facecolor('lightcoral')
        """
        leg = F1_plot.get_legend()
        #leg.legendHandles[2].set_color('lightcoral')
        plt.show()

    def plot_MLE_runtime():
        plt.figure(figsize=(8, 5))
        MLE_flag = oper_char_df["method"] == 'MLE'

        runtime_plot = sns.boxplot(y=oper_char_df.loc[MLE_flag, "runtime"],
                                   x=oper_char_df.loc[MLE_flag, "sparsity size"],
                                   # hue=oper_char_df["method"],
                                   # palette="pastel",
                                   #color='lightcoral',
                                   color='lightskyblue',
                                   orient="v",
                                   linewidth=1)
        runtime_plot.set(title='Runtime in Seconds for MLE')
        runtime_plot.set_ylim(0, 1.)
        # plt.tight_layout()
        #runtime_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
        plt.show()

    #plot_naive()
    plot_comparison()
    plot_len_comparison()
    plot_F1_comparison()
    #plot_MLE_runtime()
