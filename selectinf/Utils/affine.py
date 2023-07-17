"""
This module contains the core code needed for post selection
inference based on affine selection procedures as
described in the papers `Kac Rice`_, `Spacings`_, `covTest`_
and `post selection LASSO`_.

.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238
.. _sample carving: http://arxiv.org/abs/1410.2597

"""

from warnings import warn
from copy import copy

import numpy as np

from selectinf.Utils.pvalue import truncnorm_cdf
from selectinf.Utils.gaussian_truncated import truncated_gaussian_old

WARNINGS = False

class constraints(object):

    r"""
    This class is the core object for affine selection procedures.
    It is meant to describe sets of the form $C$
    where

    .. math::

       C = \left\{z: Az\leq b \right \}

    Its main purpose is to consider slices through $C$
    and the conditional distribution of a Gaussian $N(\mu,\Sigma)$
    restricted to such slices.

    Notes
    -----

    In this parameterization, the parameter `self.mean` corresponds
    to the *reference measure* that is being truncated. It is not the
    mean of the truncated Gaussian.

    >>> positive = constraints(-np.identity(2), np.zeros(2))
    >>> Y = np.array([3, 4.4])
    >>> eta = np.array([1, 1], np.float)
    >>> list(positive.interval(eta, Y))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [4.62...,  10.17...]
    >>> positive.pivot(eta, Y) # doctest: +ELLIPSIS 
    5.187...-07
    >>> list(positive.bounds(eta, Y)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [1.399..., 7.4..., inf, 1.414...]  
    >>> 

    """

    def __init__(self, 
                 linear_part,
                 offset,
                 covariance=None,
                 mean=None,
                 rank=None):
        r"""
        Create a new inequality. 

        Parameters
        ----------

        linear_part : np.float((q,p))
            The linear part, $A$ of the affine constraint
            $\{z:Az \leq b\}$. 

        offset: np.float(q)
            The offset part, $b$ of the affine constraint
            $\{z:Az \leq b\}$. 

        covariance : np.float((p,p))
            Covariance matrix of Gaussian distribution to be 
            truncated. Defaults to `np.identity(self.dim)`.

        mean : np.float(p)
            Mean vector of Gaussian distribution to be 
            truncated. Defaults to `np.zeros(self.dim)`.

        rank : int
            If not None, this should specify
            the rank of the covariance matrix. Defaults
            to self.dim.

        """

        self.linear_part, self.offset = \
            linear_part, np.asarray(offset)
        
        if self.linear_part.ndim == 2:
            self.dim = self.linear_part.shape[1]
        else:
            self.dim = self.linear_part.shape[0]

        if rank is None:
            self.rank = self.dim
        else:
            self.rank = rank

        if covariance is None:
            covariance = np.identity(self.dim)
        self.covariance = covariance

        if mean is None:
            mean = np.zeros(self.dim)
        self.mean = mean

    def _repr_latex_(self):
        r"""
        >>> A = np.array([[0.32,0.27,0.19],
        ... [0.59,0.98,0.71],
        ... [0.34,0.15,0.17,0.25],
        ... [0.34,0.15,0.17,0.25]])
        >>> B = np.array([ 0.51,  0.74,  0.72 ,  0.82])
        >>> C = constraints(A, B)
        >>> C._repr_latex_()
        '$$Z \\sim N(\\mu,\\Sigma) | AZ \\leq b$$'
        """
        return r"""$$Z \sim N(\mu,\Sigma) | AZ \leq b$$"""

    def __copy__(self):
        r"""
        A copy of the constraints.

        Also copies _sqrt_cov, _sqrt_inv if attributes are present.
        """
        con = constraints(copy(self.linear_part),
                          copy(self.offset),
                          mean=copy(self.mean),
                          covariance=copy(self.covariance),
                          rank=self.rank)
        if hasattr(self, "_sqrt_cov"):
            con._sqrt_cov = self._sqrt_cov.copy()
            con._sqrt_inv = self._sqrt_inv.copy()
            con._rowspace = self._rowspace.copy()
        return con

    def __call__(self, Y, tol=1.e-3):
        r"""
        Check whether Y satisfies the linear
        inequality constraints.
        >>> A = np.array([[1., -1.], [1., -1.]])
        >>> B = np.array([1., 1.])
        >>> con = constraints(A, B)
        >>> Y = np.array([-1., 1.])
        >>> con(Y)
        True
        """
        V1 = self.value(Y)
        return np.all(V1 < tol * np.fabs(V1).max(0))

    def value(self, Y):
        r"""
        Compute $\max(Ay-b)$.
        """
        return (self.linear_part.dot(Y) - self.offset).max()

    def conditional(self, linear_part, value,
                    rank=None):
        """
        Return an equivalent constraint 
        after having conditioned on a linear equality.
        
        Let the inequality constraints be specified by
        `(A,b)` and the equality constraints be specified
        by `(C,d)`. We form equivalent inequality constraints by 
        considering the residual

        .. math::
           
           AY - E(AY|CY=d)

        Parameters
        ----------

        linear_part : np.float((k,q))
             Linear part of equality constraint, `C` above.

        value : np.float(k)
             Value of equality constraint, `b` above.

        rank : int
            If not None, this should specify
            the rank of `linear_part`. Defaults
            to `min(k,q)`.

        Returns
        -------

        conditional_con : `constraints`
             Affine constraints having applied equality constraint.

        """

        S = self.covariance
        C, d = linear_part, value

        M1 = S.dot(C.T)
        M2 = C.dot(M1)

        if M2.shape:
            M2i = np.linalg.pinv(M2)
            delta_cov = M1.dot(M2i.dot(M1.T))
            delta_mean = M1.dot(M2i.dot(C.dot(self.mean) - d))
        else:
            delta_cov = np.multiply.outer(M1, M1) / M2
            delta_mean = M1 * (C.dot(self.mean) - d) / M2

        if rank is None:
            if len(linear_part.shape) == 2:
                rank = min(linear_part.shape)
            else:
                rank = 1

        return constraints(self.linear_part,
                           self.offset,
                           covariance=self.covariance - delta_cov,
                           mean=self.mean - delta_mean,
                           rank=self.rank - rank)

    def bounds(self, direction_of_interest, Y):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$.

        Parameters
        ----------

        direction_of_interest: np.float
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float
            A realization of $N(\mu,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        Returns
        -------

        L : np.float
            Lower truncation bound.

        Z : np.float
            The observed $\eta^TY$

        U : np.float
            Upper truncation bound.

        S : np.float
            Standard deviation of $\eta^TY$.

        
        """
        return interval_constraints(self.linear_part,
                                    self.offset,
                                    self.covariance,
                                    Y,
                                    direction_of_interest)

    def pivot(self, 
              direction_of_interest, 
              Y,
              null_value=None,
              alternative='greater'):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.

        Parameters
        ----------

        direction_of_interest: np.float
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float
            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------

        P : np.float
            $p$-value of corresponding test.

        Notes
        -----

        All of the Tests are based on the exact pivot $F$ given
        by the truncated Gaussian distribution for the
        given direction $\eta$. If the alternative is 'greater'
        then we return $1-F$; if it is 'less' we return $F$
        and if it is 'twosided' we return $2 \min(F,1-F)$.

        """

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")
        L, Z, U, S = self.bounds(direction_of_interest, Y)

        if null_value is None:
            meanZ = (direction_of_interest * self.mean).sum()
        else:
            meanZ = null_value

        P = truncnorm_cdf((Z-meanZ)/S, (L-meanZ)/S, (U-meanZ)/S)

        if alternative == 'greater':
            return 1 - P
        elif alternative == 'less':
            return P
        else:
            return max(2 * min(P, 1-P), 0)

    def interval(self, direction_of_interest, Y,
                 alpha=0.05, UMAU=False):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.
        
        Parameters
        ----------

        direction_of_interest: np.float

            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float

            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        alpha : float

            What level of confidence?

        UMAU : bool

            Use the UMAU intervals?

        Returns
        -------

        [U,L] : selection interval

        
        """
        ## THE DOCUMENTATION IS NOT GOOD ! HAS TO BE CHANGED !

        return selection_interval( \
            self.linear_part,
            self.offset,
            self.covariance,
            Y,
            direction_of_interest,
            alpha=alpha,
            UMAU=UMAU)

    def covariance_factors(self, force=True):
        """
        Factor `self.covariance`,
        finding a possibly non-square square-root.

        Parameters
        ----------

        force : bool
            If True, force a recomputation of
            the covariance. If not, assumes that
            covariance has not changed.

        """
        if not hasattr(self, "_sqrt_cov") or force:

            # original matrix is np.dot(U, (D**2 * U).T)

            U, D = np.linalg.svd(self.covariance)[:2]
            D = np.sqrt(D[:self.rank])
            U = U[:,:self.rank]
        
            self._sqrt_cov = U * D[None,:]
            self._sqrt_inv = (U / D[None,:]).T
            self._rowspace = U

        return self._sqrt_cov, self._sqrt_inv, self._rowspace

    def estimate_mean(self, observed):
        """
        Softmax estimator based on an observed data point.
        
        Makes a whitened copy 
        then returns softmax estimate.

        TODO: what if self.mean != 0 before hand?

        """

        inverse_map, forward_map, white_con = self.whiten()
        W = white_con
        A, b = W.linear_part, W.offset

        white_observed = forward_map(observed)
        slack = b - A.dot(white_observed) 
        dslack = 1. / (slack + 1.) - 1. / slack
        return inverse_map(white_observed - A.T.dot(dslack))

    def whiten(self):
        """

        Return a whitened version of constraints in a different
        basis, and a change of basis matrix.

        If `self.covariance` is rank deficient, the change-of
        basis matrix will not be square.

        Returns
        -------

        inverse_map : callable

        forward_map : callable

        white_con : `constraints`

        """
        sqrt_cov, sqrt_inv = self.covariance_factors()[:2]

        new_A = self.linear_part.dot(sqrt_cov)
        den = np.sqrt((new_A**2).sum(1))
        new_b = self.offset - self.linear_part.dot(self.mean)
        new_con = constraints(new_A / den[:,None], new_b / den)

        mu = self.mean.copy()

        def inverse_map(Z): 
            if Z.ndim == 2:
                return sqrt_cov.dot(Z) + mu[:,None]
            else:
                return sqrt_cov.dot(Z) + mu

        forward_map = lambda W: sqrt_inv.dot(W - mu)

        return inverse_map, forward_map, new_con

    def project_rowspace(self, direction):
        """
        Project a vector onto rowspace
        of the covariance.
        """
        rowspace = self.covariance_factors()[-1]
        return rowspace.dot(rowspace.T.dot(direction))

    def solve(self, direction):
        """
        Compute the inverse of the covariance
        times a direction vector.
        """
        sqrt_inv = self.covariance_factors()[1]
        return sqrt_inv.T.dot(sqrt_inv.dot(direction))

def stack(*cons):
    """
    Combine constraints into a large constaint
    by intersection. 

    Parameters
    ----------

    cons : [`selection.affine.constraints`_]
         A sequence of constraints.

    Returns
    -------

    intersection : `selection.affine.constraints`_

    Notes
    -----

    Resulting constraint will have mean 0 and covariance $I$.

    """
    ineq, ineq_off = [], []
    for con in cons:
        ineq.append(con.linear_part)
        ineq_off.append(con.offset)

    intersection = constraints(np.vstack(ineq), 
                               np.hstack(ineq_off))
    return intersection

def interval_constraints(support_directions, 
                         support_offsets,
                         covariance,
                         observed_data, 
                         direction_of_interest,
                         tol = 1.e-4):
    r"""
    Given an affine constraint $\{z:Az \leq b \leq \}$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $\eta$, and
    an `observed_data` is Gaussian vector $Z \sim N(\mu,\Sigma)$ 
    with `covariance` matrix $\Sigma$, this
    function returns $\eta^TZ$ as well as an interval
    bounding this value. 

    The interval constructed is such that the endpoints are 
    independent of $\eta^TZ$, hence the $p$-value
    of `Kac Rice`_
    can be used to form an exact pivot.

    Parameters
    ----------

    support_directions : np.float
         Matrix specifying constraint, $A$.

    support_offsets : np.float
         Offset in constraint, $b$.

    covariance : np.float
         Covariance matrix of `observed_data`.

    observed_data : np.float
         Observations.

    direction_of_interest : np.float
         Direction in which we're interested for the
         contrast.

    tol : float
         Relative tolerance parameter for deciding 
         sign of $Az-b$.

    Returns
    -------

    lower_bound : float

    observed : float

    upper_bound : float

    sigma : float

    """

    # shorthand
    A, b, S, X, w = (support_directions,
                     support_offsets,
                     covariance,
                     observed_data,
                     direction_of_interest)

    U = A.dot(X) - b
    if not np.all(U  < tol * np.fabs(U).max()) and WARNINGS:
        warn('constraints not satisfied: %s' % repr(U))

    Sw = S.dot(w)
    sigma = np.sqrt((w*Sw).sum())
    alpha = A.dot(Sw) / sigma**2
    V = (w*X).sum() # \eta^TZ

    # adding the zero_coords in the denominator ensures that
    # there are no divide-by-zero errors in RHS
    # these coords are never used in upper_bound or lower_bound

    zero_coords = alpha == 0
    RHS = (-U + V * alpha) / (alpha + zero_coords)
    RHS[zero_coords] = np.nan

    pos_coords = alpha > tol * np.fabs(alpha).max()
    if np.any(pos_coords):
        upper_bound = RHS[pos_coords].min()
    else:
        upper_bound = np.inf
    neg_coords = alpha < -tol * np.fabs(alpha).max()
    if np.any(neg_coords):
        lower_bound = RHS[neg_coords].max()
    else:
        lower_bound = -np.inf

    return lower_bound, V, upper_bound, sigma

def selection_interval(support_directions, 
                       support_offsets,
                       covariance,
                       observed_data, 
                       direction_of_interest,
                       tol = 1.e-4,
                       alpha = 0.05,
                       UMAU=True):
    """
    Given an affine in cone constraint $\{z:Az+b \leq 0\}$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $\eta$, and
    an `observed_data` is Gaussian vector $Z \sim N(\mu,\Sigma)$ 
    with `covariance` matrix $\Sigma$, this
    function returns a confidence interval
    for $\eta^T\mu$.

    Parameters
    ----------

    support_directions : np.float
         Matrix specifying constraint, $A$.

    support_offset : np.float
         Offset in constraint, $b$.

    covariance : np.float
         Covariance matrix of `observed_data`.

    observed_data : np.float
         Observations.

    direction_of_interest : np.float
         Direction in which we're interested for the
         contrast.

    tol : float
         Relative tolerance parameter for deciding 
         sign of $Az-b$.

    UMAU : bool
         Use the UMAU interval, or twosided pivot.

    Returns
    -------

    selection_interval : (float, float)

    """

    lower_bound, V, upper_bound, sigma = interval_constraints( \
        support_directions, 
        support_offsets,
        covariance,
        observed_data, 
        direction_of_interest,
        tol=tol)

    truncated = truncated_gaussian_old([(lower_bound, upper_bound)], scale=sigma)
    if UMAU:
        _selection_interval = truncated.UMAU_interval(V, alpha)
    else:
        _selection_interval = truncated.equal_tailed_interval(V, alpha)
    
    return _selection_interval
