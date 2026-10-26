"""
BGL (Brand-Goy-Lemke 2025) Macro-Finance Model — India Adaptation
=====================================================================
Jointly estimates r*, trend inflation, output gap, yield curve factors,
and term premia using Bayesian Gibbs sampling + Durbin-Koopman smoother.

Model:
  Macro: IS curve (output gap ~ real rate gap), Phillips curve (inflation gap),
         r* = 4*c*g + z (c: estimated growth-to-r* scaling, trend growth, non-growth component)
  Finance: Arbitrage-free Nelson-Siegel yield curve with non-stationary level

Data required (quarterly, in CSV, or via load_india_data() for Data.xlsx):
  - gdp_real: log real GDP (seasonally adjusted), e.g. from MOSPI/RBI
  - cpi_yoy: year-on-year CPI inflation (%), e.g. CPI-Combined from MOSPI
  - y_1, y_2, y_3, y_4: zero-coupon yields (%) for 1Q, 2Q, 3Q, 4Q
  - y_8, y_12, y_16, y_20, y_24, y_28, y_32, y_36, y_40: ZC yields 2Y-10Y
    Source: FBIL/CCIL G-sec zero coupon yield curve (quarterly end-of-period)
  - survey_pi_star: long-run inflation expectations (optional, e.g. RBI household survey)
  - survey_i_4q: 1-year ahead short rate expectations (optional, e.g. consensus)

All yields and inflation in annualized percent.
"""

import copy
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numba
import numpy as np
from numpy.linalg import inv, solve, eigvals
from scipy.stats import invgamma, invwishart, norm, multivariate_normal
from scipy.linalg import block_diag
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

@dataclass
class BGLConfig:
    """Model configuration and prior hyperparameters."""
    # Maturities (in quarters) for yield data
    maturities: list = field(default_factory=lambda: [1,2,3,4,8,12,16,20,24,28,32,36,40])
    
    # DNS decay parameter. Prior centred on 0.095, the value minimizing
    # DNS-OLS residuals on the Indian G-sec curve across all 13 maturities
    # (RMSE 7.5bp, vs 8.2bp at the paper's US value 0.1821 and 10.3bp at
    # 0.29888). India's curve is steeper for longer than the US curve, so the
    # curvature hump sits far out (~1.79/0.095 = 19 quarters).
    # lambda_prior_mean doubles as the fixed value when estimate_lambda is False.
    estimate_lambda: bool = True
    lambda_prior_mean: float = 0.095
    lambda_prior_sd: float = 0.05
    lambda_mh_scale: float = 0.004  # MH proposal scale, tuned for 20-40% acceptance

    # MH proposal scales, tuned for 20-40% acceptance.
    c_mh_scale: float = 0.10      # growth-to-r* scaling c (r* = 4*c*g + z)
    omega_mh_scale: float = 0.15  # real-rate-gap wedge omega

    # Set to False to fix omega at 0, which recovers the paper's model
    # exactly (mean(r*) == mean ex-ante real rate).
    estimate_omega: bool = True


    # MCMC settings, per chain; N_CHAINS chains are pooled for reporting
    # (paper: one chain of 100,000 / 90,000 / 10, plus a multi-chain check).
    # India needs heavier thinning than the paper: lambda mixes slowly because
    # it is only weakly separated from the S_bar/C_bar block drawn by Gibbs,
    # and at thin=10 its Geweke z-score was 3.3.
    n_draws: int = 40000
    n_burnin: int = 20000
    thin: int = 20
    
    # State dimension
    n_states: int = 14  # L_tilde, S, C, pi*, x*, g, z, pi_tilde, x_tilde, + 5 lags


# ============================================================
# 2. NELSON-SIEGEL LOADINGS
# ============================================================

def ns_loadings(lam, tau):
    """Nelson-Siegel factor loadings for maturity tau (in quarters)."""
    if tau == 0:
        return 1.0, 1.0, 0.0
    lt = lam * tau
    exp_lt = np.exp(-lt)
    theta_s = (1 - exp_lt) / lt
    theta_c = theta_s - exp_lt
    return 1.0, theta_s, theta_c


# Largest unconditional slope/curvature the sampler will accept, in percentage
# points. SC_bar = (I - A_sc)^-1 a0 diverges as |eig(A_sc)| -> 1 while the VAR
# is still formally stationary: draws at |eig| = 0.986 to 0.9993 produced
# |C_bar| of 27pp up to 166pp, and hence A(40) of -12pp to -81pp. Those
# intercepts are not an economically meaningful yield curve, and they corrupt
# the FOLLOWING iteration's state draw as well as the draw that produced them.
# Cross-sectional DNS-OLS on the Indian curve puts S in [-6.4, +1.2] and C in
# [-9.9, +7.8] over 2000-2025, so a long-run mean past 20pp is the
# (I - A_sc)^-1 singularity, not something the data support.
SC_BAR_MAX = 20.0


def sc_unconditional_mean(p):
    """Unconditional mean (S_bar, C_bar) of the stationary slope-curvature VAR."""
    A_sc = np.array([[p['a11'], p['a12']], [p['a21'], p['a22']]])
    if np.abs(eigvals(A_sc)).max() < 1:
        SC_bar = solve(np.eye(2) - A_sc, np.array([p['a10'], p['a20']]))
    else:
        SC_bar = np.array([0.0, 0.0])
    return SC_bar[0], SC_bar[1]


# Unit conversion between the affine pricing recursion and the data.
# The recursion (paper eq. B.13-B.14) prices bonds using a *per-period, decimal*
# short rate, but delta_u, Omega and the state factors in this code are all in
# *annualized percent*. Writing a_dec = a_ann/RATE_SCALE and b_dec = b_ann/RATE_SCALE
# and substituting into B.13 leaves every term unchanged except the quadratic
# convexity term 0.5*b'Omega*b, which picks up a single factor 1/RATE_SCALE.
# Without it A(40) comes out around -180pp instead of +0.9pp (a -45bp convexity),
# which destroys the Kalman filter. See tests/test_run_model.py::test_arbitrage_adj_scale.
RATE_SCALE = 400.0  # 4 quarters per year x 100 percentage points

# Measurement-error sd used for series the model matches "without error"
# (the 1Q yield, log GDP and inflation). A true zero is not usable and 1e-6
# left S = C P C' + R too ill-conditioned to update stably; 1e-3 (0.1bp) is
# negligible economically while keeping the filter well behaved.
EXACT_MEAS_SD = 1e-3

# Data.xlsx stores log10(real GDP); the model needs 100*ln(GDP) so that the
# output gap is in percentage points. 100*ln(10) = 230.2585.
LOG10_TO_PCT_LOG = 100.0 * np.log(10.0)


@numba.njit(cache=True)
def _arbitrage_a_path(lam, tau_max, om_L, om_S, om_C, S_bar, C_bar):
    """
    a(tau) for every tau = 1..tau_max in a single recursion.

    The recursion for maturity tau passes through every shorter maturity, so
    one pass to tau_max yields A(tau) for the whole maturity vector; running it
    separately per maturity (as this used to) repeats sum(taus) steps instead
    of max(taus). Returns an array indexed by tau, with a[0] unused.
    """
    exp_l = np.exp(-lam)
    ts1 = (1.0 - exp_l) / lam
    tc1 = ts1 - exp_l

    # delta_u for the short rate, and the Q-measure autoregressive matrix.
    delta_u = np.array([1.0, ts1, tc1])
    Phi_Q_T = np.array([[1.0, 0.0, 0.0],
                        [0.0, exp_l, 0.0],
                        [0.0, lam*exp_l, exp_l]])   # Phi_Q transposed

    # delta_0 = -theta_s(1)*S_bar - theta_c(1)*C_bar (Appendix B, eq. B.13).
    # With c^Q = 0, a(1) = -delta_0 and b_u(1) = -delta_u.
    delta_0 = -ts1 * S_bar - tc1 * C_bar
    a = -delta_0
    bu = -delta_u.copy()

    a_vals = np.zeros(tau_max + 1)
    a_vals[1] = a
    for t in range(1, tau_max):
        # c^Q = 0, so the b_u(tau)'c^Q_u term drops out. Omega_uu is diagonal,
        # so the quadratic form b_u'Omega b_u is a weighted sum of squares.
        quad = om_L*bu[0]**2 + om_S*bu[1]**2 + om_C*bu[2]**2
        a = a + 0.5 * quad / RATE_SCALE - delta_0
        bu = Phi_Q_T @ bu - delta_u
        a_vals[t+1] = a

    return a_vals


def _omega_uu_diag(sigma_L, sigma_S, sigma_C, sigma_pi, sigma_g, sigma_z, c):
    """Diagonal of Omega_uu (see arbitrage_adj)."""
    return (sigma_L**2 + sigma_pi**2 + (4*c)**2*sigma_g**2 + sigma_z**2,
            sigma_S**2, sigma_C**2)


def arbitrage_adj(lam, tau, sigma_L, sigma_S, sigma_C, sigma_pi, sigma_g, sigma_z, S_bar, C_bar, c=1.0):
    """
    Compute A(tau) — the no-arbitrage adjustment term.
    Uses recursive difference equation from Christensen et al. (2011).
    Omega_uu = diag(sigma_L^2 + sigma_pi^2 + (4*c)^2*sigma_g^2 + sigma_z^2, sigma_S^2, sigma_C^2)
    (r* = 4*c*g + z, so the level factor L = pi* + 4*c*g + z + L_tilde loads on g with weight 4*c)
    """
    if tau <= 0:
        return 0.0
    om = _omega_uu_diag(sigma_L, sigma_S, sigma_C, sigma_pi, sigma_g, sigma_z, c)
    a_vals = _arbitrage_a_path(float(lam), int(tau), om[0], om[1], om[2],
                               float(S_bar), float(C_bar))
    return -a_vals[tau] / tau


def arbitrage_adj_many(lam, taus, sigma_L, sigma_S, sigma_C, sigma_pi, sigma_g,
                       sigma_z, S_bar, C_bar, c=1.0):
    """A(tau) for a whole maturity vector, from one shared recursion."""
    om = _omega_uu_diag(sigma_L, sigma_S, sigma_C, sigma_pi, sigma_g, sigma_z, c)
    a_vals = _arbitrage_a_path(float(lam), int(max(taus)), om[0], om[1], om[2],
                               float(S_bar), float(C_bar))
    return np.array([-a_vals[t] / t if t > 0 else 0.0 for t in taus])


# ============================================================
# 3. STATE SPACE CONSTRUCTION
# ============================================================

def build_state_space(params, lam, config):
    """
    Build F (transition), mu (intercept), C (measurement), gamma (meas. intercept),
    G (state noise), D (meas. noise) matrices.
    
    State vector (14 x 1):
    [L_tilde, S, C, pi*, x*, g, z, pi_tilde, x_tilde,
     L_tilde_{-1}, S_{-1}, C_{-1}, pi_tilde_{-1}, x_tilde_{-1}]
    
    params dict keys:
      a1, a2, a3, b1, b2, aL,
      a10, a11, a12, a13, a14, a20, a21, a22, a23, a24,
      sig_Lc, sig_S, sig_C, sig_pi_star, sig_x_star, sig_g, sig_z,
      sig_pi_tilde, sig_x_tilde, sig_y (array for K-1 yields), sig_s_pi, sig_s_sr
    """
    p = params
    N = config.n_states  # 14
    taus = config.maturities
    K = len(taus)
    
    # Slope/curvature unconditional means
    S_bar, C_bar = sc_unconditional_mean(p)
    
    _, ts1, tc1 = ns_loadings(lam, 1)
    
    # --- Transition matrix F (14x14) ---
    F = np.zeros((N, N))
    
    # L_tilde
    F[0, 0] = p['aL']
    # S
    F[1, 1] = p['a11']; F[1, 2] = p['a12']; F[1, 7] = p['a13']; F[1, 8] = p['a14']
    # C
    F[2, 1] = p['a21']; F[2, 2] = p['a22']; F[2, 7] = p['a23']; F[2, 8] = p['a24']
    # pi*: random walk
    F[3, 3] = 1.0
    # x*: x*_{t-1} + g_{t-1}
    F[4, 4] = 1.0; F[4, 5] = 1.0
    # g: random walk
    F[5, 5] = 1.0
    # z: random walk
    F[6, 6] = 1.0
    # pi_tilde (Phillips curve)
    F[7, 7] = p['b1']; F[7, 8] = p['b2']
    # x_tilde (IS curve with real rate gap substituted)
    a3 = p['a3']
    F[8, 0] = a3;  F[8, 1] = a3*ts1;  F[8, 2] = a3*tc1
    F[8, 7] = -a3*p['b1']; F[8, 8] = p['a1'] - a3*p['b2']
    # lagged terms (row 8 cont.)
    F[8, 9] = a3;  F[8, 10] = a3*ts1;  F[8, 11] = a3*tc1
    F[8, 12] = -a3*p['b1']; F[8, 13] = p['a2'] - a3*p['b2']
    # Identity for lags
    F[9, 0] = 1.0   # L_tilde_{-1} = L_tilde
    F[10, 1] = 1.0  # S_{-1}
    F[11, 2] = 1.0  # C_{-1}
    F[12, 7] = 1.0  # pi_tilde_{-1}
    F[13, 8] = 1.0  # x_tilde_{-1}
    
    # --- Intercept mu ---
    # The real rate gap enters the IS curve twice (at t-1 and t-2), so the
    # constant -a3*(theta_s(1)*S_bar + theta_c(1)*C_bar) is picked up twice.
    # (The paper's Appendix A.1 prints mu without the factor 2, but the
    # expansion of x_tilde two paragraphs earlier has it. Without the 2 the
    # model implies an unconditional mean output gap of ~+9pp for India.)
    #
    # omega is the constant real-rate-gap wedge, omega = mean(r_t - r*_t): the
    # free IS intercept a0 = -2*a3*omega is what allows mean(r_tilde) = omega
    # while keeping mean(x_tilde) = 0. omega = 0 recovers the paper exactly.
    omega_wedge = p.get('omega', 0.0)
    mu = np.zeros(N)
    mu[1] = p['a10']
    mu[2] = p['a20']
    mu[8] = -2.0 * a3 * (ts1 * S_bar + tc1 * C_bar) - 2.0 * a3 * omega_wedge
    
    # --- State noise G (14 x 14 diagonal, only first 9 have shocks) ---
    G_diag = np.zeros(N)
    G_diag[0] = p['sig_Lc']
    G_diag[1] = p['sig_S']
    G_diag[2] = p['sig_C']
    G_diag[3] = p['sig_pi_star']
    G_diag[4] = p['sig_x_star']
    G_diag[5] = p['sig_g']
    G_diag[6] = p['sig_z']
    G_diag[7] = p['sig_pi_tilde']
    G_diag[8] = p['sig_x_tilde']
    G = np.diag(G_diag)
    Q = G @ G.T  # State covariance
    
    # --- Measurement equation ---
    # Observables: K yields, x_t (GDP), pi_t (inflation), [survey_i, survey_pi]
    n_obs_base = K + 2  # yields + GDP + inflation
    has_survey_pi = 'sig_s_pi' in p and p['sig_s_pi'] > 0
    has_survey_sr = 'sig_s_sr' in p and p['sig_s_sr'] > 0
    has_pi_anchor = 'sig_anchor' in p and p['sig_anchor'] > 0
    n_surveys = int(has_survey_pi) + int(has_survey_sr) + int(has_pi_anchor)
    n_obs = n_obs_base + n_surveys
    
    C_mat = np.zeros((n_obs, N))
    gamma = np.zeros(n_obs)
    D_diag = np.zeros(n_obs)
    
    # Yield equations: y(tau) = A(tau) + L_tilde + L* + theta_s*S + theta_c*C + error
    # L* = pi* + r* = pi* + 4*c*g + z, so loadings on pi*(1), g(4*c), z(1)
    A_tau = arbitrage_adj_many(lam, taus,
                               p['sig_Lc'], p['sig_S'], p['sig_C'],
                               p['sig_pi_star'], p['sig_g'], p['sig_z'],
                               S_bar, C_bar, p['c'])
    for k, tau in enumerate(taus):
        _, ts, tc = ns_loadings(lam, tau)
        # L*_t = pi*_t + 4c*g_t + z_t + omega: the wedge shifts the long-run
        # level anchor of the whole curve, so it enters every yield intercept.
        gamma[k] = omega_wedge + A_tau[k]
        C_mat[k, 0] = 1.0    # L_tilde
        C_mat[k, 1] = ts     # S
        C_mat[k, 2] = tc     # C
        C_mat[k, 3] = 1.0    # pi*
        C_mat[k, 5] = 4.0 * p['c']  # 4*c*g (part of L* = pi* + 4*c*g + z)
        C_mat[k, 6] = 1.0    # z
        
        # The 1Q yield is matched (near-)exactly. EXACT_MEAS_SD rather than a
        # true zero keeps S = C P C' + R invertible; 1e-6 made it so
        # ill-conditioned that the covariance updates lost symmetry.
        if k == 0:
            D_diag[k] = EXACT_MEAS_SD
        else:
            D_diag[k] = p['sig_y'][k-1] if 'sig_y' in p else 0.1

    # GDP: x_t = x* + x_tilde
    C_mat[K, 4] = 1.0      # x*
    C_mat[K, 8] = 1.0      # x_tilde
    D_diag[K] = EXACT_MEAS_SD   # measured without error

    # Inflation: pi_t = pi* + pi_tilde
    C_mat[K+1, 3] = 1.0    # pi*
    C_mat[K+1, 7] = 1.0    # pi_tilde
    D_diag[K+1] = EXACT_MEAS_SD

    # Surveys (if present). Both survey rows are 4-quarter-ahead model
    # expectations, so they share the structure of the paper's eq. (17):
    # E_t xi_{t+4} = (I + F + F^2 + F^3) mu + F^4 xi_t.
    idx = K + 2
    if has_survey_pi or has_survey_sr:
        F4 = np.linalg.matrix_power(F, 4)
        F_cum = np.eye(N) + F + F @ F + F @ F @ F

    if has_survey_pi:
        # The IESH series is a 1-year-ahead household inflation forecast, not a
        # long-run measure, so it is matched to the model-implied
        # E_t pi_{t+4} = E_t pi*_{t+4} + E_t pi~_{t+4} rather than to pi* itself.
        # bias_s_pi is the households' systematic over-prediction, estimated
        # inside the model instead of being subtracted from the data up front.
        C_mat[idx, :] = F4[3, :] + F4[7, :]
        gamma[idx] = p.get('bias_s_pi', 0.0) + (F_cum @ mu)[3] + (F_cum @ mu)[7]
        D_diag[idx] = p['sig_s_pi']
        idx += 1
    if has_survey_sr:
        # E_t y_{t+4}(1) ≈ C1 @ F^4 @ xi_t + constants
        C1 = C_mat[0, :]  # first yield row
        C_mat[idx, :] = C1 @ F4
        gamma[idx] = gamma[0] + C1 @ F_cum @ mu
        D_diag[idx] = p['sig_s_sr']
        idx += 1
    if has_pi_anchor:
        # Direct anchor on trend inflation: the RBI's 4% flexible-inflation-
        # targeting midpoint, observable from 2016Q3. Unlike the IESH survey
        # (a 1-year-ahead forecast) this speaks to pi* itself, which is what
        # the level of r* = i* - pi* actually needs.
        C_mat[idx, 3] = 1.0
        D_diag[idx] = p['sig_anchor']
    
    D = np.diag(D_diag)
    R = D @ D.T
    
    return F, mu, Q, G, C_mat, gamma, R, n_obs


# ============================================================
# 4. KALMAN FILTER & DURBIN-KOOPMAN SIMULATION SMOOTHER
# ============================================================

@numba.njit(cache=True)
def _kf_core(Y, F, mu, Q, C, gamma, R, xi0, P0, store):
    """
    Kalman filter core. With store=False the T x N x N filtered/predicted
    histories are not allocated or written — the MH steps only need the
    log-likelihood, and those four arrays are the bulk of the filter's
    memory traffic.

    Two other things keep this cheap: S is factorized once per period
    (a Cholesky supplies both the solve and the log-determinant, where
    inv + slogdet cost two factorizations), and the sliced C_t/R_t/gamma_t
    are rebuilt only when the missing-data pattern actually changes — India's
    sample has 3 distinct patterns across 102 quarters, in long runs.
    """
    T, n_obs = Y.shape
    N = F.shape[0]

    ns = T if store else 1
    xi_filt = np.zeros((ns, N))
    P_filt = np.zeros((ns, N, N))
    xi_pred = np.zeros((ns, N))
    P_pred = np.zeros((ns, N, N))
    log_lik = 0.0

    xi_t = xi0.copy()
    P_t = P0.copy()
    eye_N = np.eye(N)

    # Placeholders, sliced with an all-False mask so numba fixes their types.
    prev_mask = np.zeros(n_obs, np.bool_)
    C_t = C[prev_mask]
    R_t = R[prev_mask][:, prev_mask]
    gamma_t = gamma[prev_mask]
    first = True

    for t in range(T):
        # Predict
        xi_p = mu + F @ xi_t
        P_p = F @ P_t @ F.T + Q

        if store:
            xi_pred[t] = xi_p
            P_pred[t] = P_p

        # Handle missing obs
        y_t = Y[t]
        obs_mask = ~np.isnan(y_t)

        n_t = 0
        changed = first
        for j in range(n_obs):
            if obs_mask[j]:
                n_t += 1
            if obs_mask[j] != prev_mask[j]:
                changed = True
        if changed:
            C_t = C[obs_mask]
            R_t = R[obs_mask][:, obs_mask]
            gamma_t = gamma[obs_mask]
            prev_mask = obs_mask
            first = False

        if n_t > 0:
            y_obs = y_t[obs_mask]

            # Innovation
            v = y_obs - gamma_t - C_t @ xi_p
            S = C_t @ P_p @ C_t.T + R_t + 1e-10 * np.eye(n_t)

            # One Cholesky serves the gain, the quadratic form and the
            # log-determinant.
            L_S = np.linalg.cholesky(S)
            logdet = 2.0 * np.sum(np.log(np.diag(L_S)))

            M = P_p @ C_t.T                      # (N x n_t)
            rhs = np.empty((n_t, N + 1))
            rhs[:, :N] = M.T
            rhs[:, N] = v
            sol = np.linalg.solve(S, rhs)
            # Contiguous: K feeds three matmuls below, and numba is much
            # slower on transposed (non-contiguous) operands.
            K = np.ascontiguousarray(sol[:, :N].T)       # P_p C' S^-1
            Sinv_v = sol[:, N]

            # Update. Joseph form: stays symmetric and PSD even when R is
            # near-singular (the 1Q yield / GDP / inflation rows are matched
            # almost exactly), unlike the shorter (I - K C) P form.
            xi_t = xi_p + K @ v
            IKC = eye_N - K @ C_t
            P_t = IKC @ P_p @ IKC.T + K @ R_t @ K.T

            # Log-likelihood
            log_lik += -0.5 * (n_t * np.log(2*np.pi) + logdet + v @ Sinv_v)
        else:
            xi_t = xi_p
            P_t = P_p

        if store:
            xi_filt[t] = xi_t
            P_filt[t] = P_t

    return xi_filt, P_filt, xi_pred, P_pred, log_lik


def kalman_filter(Y, F, mu, Q, C, gamma, R, xi0, P0):
    """Standard Kalman filter. Returns log-likelihood and filtered states."""
    return _kf_core(Y, F, mu, Q, C, gamma, R, xi0, P0, True)


def kalman_loglik(Y, F, mu, Q, C, gamma, R, xi0, P0):
    """Log-likelihood only — skips the filtered-state bookkeeping."""
    return _kf_core(Y, F, mu, Q, C, gamma, R, xi0, P0, False)[4]


@numba.njit(cache=True)
def _simulate_forward(F, mu, C, gamma, xi_start, eta_all, eps_all):
    """Simulate the state-space forward from xi_start (Durbin-Koopman step 1)."""
    T = eta_all.shape[0]
    N = F.shape[0]
    xi_plus = np.zeros((T + 1, N))
    xi_plus[0] = xi_start
    Y_plus = np.zeros((T, C.shape[0]))
    for t in range(T):
        xi_plus[t+1] = mu + F @ xi_plus[t] + eta_all[t]
        Y_plus[t] = gamma + C @ xi_plus[t+1] + eps_all[t]
    return xi_plus, Y_plus


def durbin_koopman_smoother(Y, F, mu, Q, C, gamma, R, xi0, P0):
    """
    Durbin-Koopman simulation smoother.
    Returns a single draw from p(xi_{1:T} | Y, params).
    """
    T, n_obs = Y.shape
    N = F.shape[0]

    # Step 1: Generate synthetic states and observations
    xi_start = xi0 + np.random.multivariate_normal(np.zeros(N), P0)

    Q_reg = Q + 1e-12 * np.eye(N)  # regularize
    R_reg = R + 1e-12 * np.eye(n_obs)

    # Draw all T innovations at once: Q_reg/R_reg are constant across t, so a
    # single batched draw does one covariance decomposition instead of T.
    eta_all = np.random.multivariate_normal(np.zeros(N), Q_reg, size=T)
    eps_all = np.random.multivariate_normal(np.zeros(n_obs), R_reg, size=T)

    xi_plus, Y_plus = _simulate_forward(F, mu, C, gamma, xi_start, eta_all, eps_all)

    # The simulated data must be missing wherever the real data are: the
    # Durbin-Koopman identity requires both smoothers to condition on the same
    # information set. Left dense, the Y_plus filter would use observations the
    # Y filter drops (8 quarters of CPI, 39 of the survey) and run different
    # Kalman gains, so the two smoothed paths would no longer cancel.
    Y_plus[np.isnan(Y)] = np.nan

    # Step 2: smooth both series in one pass — see _rts_smoother2.
    xi_smooth, xi_smooth_plus = _rts_smoother2(Y, Y_plus, F, mu, Q, C, gamma, R, xi0, P0)

    # Step 3: draw = smooth(Y) - smooth(Y_plus) + xi_plus[1:]
    xi_draw = xi_smooth - xi_smooth_plus + xi_plus[1:]

    return xi_draw


@numba.njit(cache=True)
def _rts_smoother2(Y1, Y2, F, mu, Q, C, gamma, R, xi0, P0):
    """
    Rauch-Tung-Striebel smoother run on two observation sets at once.

    Y1 and Y2 share F, mu, Q, C, gamma, R and — required — the same pattern of
    missing values, so the entire covariance side of the recursion (P_pred,
    P_filt, the gain K, the smoothing matrix L) is identical for both and is
    computed once here. Only the two mean recursions differ. That covariance
    work dominates the filter, so this costs far less than two passes.
    """
    T = Y1.shape[0]
    n_obs = Y1.shape[1]
    N = F.shape[0]

    xi1 = xi0.copy()
    xi2 = xi0.copy()
    P_t = P0.copy()
    eye_N = np.eye(N)

    xi_filt1 = np.zeros((T, N))
    xi_filt2 = np.zeros((T, N))
    xi_pred1 = np.zeros((T, N))
    xi_pred2 = np.zeros((T, N))
    P_filt = np.zeros((T, N, N))
    P_pred = np.zeros((T, N, N))

    prev_mask = np.zeros(n_obs, np.bool_)
    C_t = C[prev_mask]
    R_t = R[prev_mask][:, prev_mask]
    gamma_t = gamma[prev_mask]
    first = True

    # --- Forward filter (shared covariance recursion, two mean recursions) ---
    for t in range(T):
        xi_p1 = mu + F @ xi1
        xi_p2 = mu + F @ xi2
        P_p = F @ P_t @ F.T + Q

        xi_pred1[t] = xi_p1
        xi_pred2[t] = xi_p2
        P_pred[t] = P_p

        y_t = Y1[t]
        obs_mask = ~np.isnan(y_t)

        n_t = 0
        changed = first
        for j in range(n_obs):
            if obs_mask[j]:
                n_t += 1
            if obs_mask[j] != prev_mask[j]:
                changed = True
        if changed:
            C_t = C[obs_mask]
            R_t = R[obs_mask][:, obs_mask]
            gamma_t = gamma[obs_mask]
            prev_mask = obs_mask
            first = False

        if n_t > 0:
            S = C_t @ P_p @ C_t.T + R_t + 1e-10 * np.eye(n_t)
            K = np.ascontiguousarray(np.linalg.solve(S, (P_p @ C_t.T).T).T)

            v1 = y_t[obs_mask] - gamma_t - C_t @ xi_p1
            v2 = Y2[t][obs_mask] - gamma_t - C_t @ xi_p2
            xi1 = xi_p1 + K @ v1
            xi2 = xi_p2 + K @ v2

            IKC = eye_N - K @ C_t
            P_t = IKC @ P_p @ IKC.T + K @ R_t @ K.T
        else:
            xi1 = xi_p1
            xi2 = xi_p2
            P_t = P_p

        xi_filt1[t] = xi1
        xi_filt2[t] = xi2
        P_filt[t] = P_t

    # --- Backward smoothing pass (L is shared too) ---
    xi_smooth1 = np.zeros((T, N))
    xi_smooth2 = np.zeros((T, N))
    xi_smooth1[-1] = xi_filt1[-1]
    xi_smooth2[-1] = xi_filt2[-1]

    eye_reg = 1e-10 * eye_N
    for t in range(T-2, -1, -1):
        L = np.linalg.solve((P_pred[t+1] + eye_reg).T, (P_filt[t] @ F.T).T).T
        xi_smooth1[t] = xi_filt1[t] + L @ (xi_smooth1[t+1] - xi_pred1[t+1])
        xi_smooth2[t] = xi_filt2[t] + L @ (xi_smooth2[t+1] - xi_pred2[t+1])

    return xi_smooth1, xi_smooth2


def _rts_smoother(Y, F, mu, Q, C, gamma, R, xi0, P0):
    """Rauch-Tung-Striebel smoother (single observation set)."""
    return _rts_smoother2(Y, Y, F, mu, Q, C, gamma, R, xi0, P0)[0]


# ============================================================
# 5. PRIOR SPECIFICATION
# ============================================================

def default_priors():
    """Prior hyperparameters adapted for India."""
    return {
        # IS curve: N(mean, sd)
        'a1': (1.40, 0.25),
        'a2': (-0.55, 0.25),
        'a3': (-0.15, 0.05),  # Indian rates may transmit less to output
        # Phillips curve
        'b1': (0.60, 0.20),
        'b2': (0.12, 0.05),
        # Cyclical level AR
        'aL': (0.50, 0.25),
        # Growth-to-r* scaling: r* = 4*c*g + z (c<1 since not all trend
        # growth feeds into the real natural rate for a converging economy).
        # Truncated to (0, 1) in check_stationarity. The marginal likelihood
        # profile is shallow: it peaks near c = 0.4-0.5 and clearly rejects
        # large c (dlogL about -3.5 at c = 1.5), but c = 1 (the LW value)
        # costs only about 1 log point, which is not a decisive rejection, and
        # the profile is nearly flat below 0.35. The prior does real work at
        # the low end. Report profile_likelihood(Y, results, 'c', ...)
        # alongside the posterior rather than the posterior alone.
        'c': (0.40, 0.15),
        # Constant real-rate-gap wedge omega = mean(r_t - r*_t), so that
        # mean(r*) = mean(r) - omega. omega < 0 for India: financial
        # repression (18% SLR, HTM carve-outs, a largely closed G-sec market)
        # holds the observed real rate below the natural rate. Weakly
        # identified -- see the note in profile_omega().
        'omega': (-1.0, 0.5),
        # Survey bias: households' systematic over-prediction of 1-year-ahead
        # inflation. Loose; the raw IESH-vs-realized gap is about 5pp.
        'bias_s_pi': (5.0, 2.0),
        # Slope/curvature VAR — prior means from OLS on Indian DNS factors
        # at lambda = 0.095, regressing S_t, C_t on
        # [const, S_{t-1}, C_{t-1}, pi_tilde_{t-1}, x_tilde_{t-1}].
        # (These were previously the paper's US values from Table 3.1
        # despite the comment claiming otherwise.)
        'SC_mean': np.array([-0.313, 0.806, -0.054, 0.007, 0.032,
                             -0.425, -0.332, 0.571, 0.093, 0.055]),
        'SC_var': 0.5 * np.eye(10),
        # Variances: Inverse-Gamma(shape, scale). Back to the paper's flat
        # Gamma^-1(4,2) (mean disturbance variance 0.66, sd 0.47). These were
        # previously tightened to keep A(tau) from exploding, which was a
        # symptom of the RATE_SCALE unit bug in arbitrage_adj, not a real
        # need for informative priors.
        'sig_Lc': (4, 2),
        # Slope/curvature innovation sds. The paper draws these jointly from an
        # Inverse-Wishart W^-1(2, I2); G is diagonal here, so the equivalent
        # marginal is an Inverse-Gamma per equation. Left flat -- India's DNS-OLS
        # residual sds (0.74 and 1.38) are well inside this prior's mass.
        'sig_S': (4, 2),
        'sig_C': (4, 2),
        'sig_pi_star': (4, 2),
        'sig_x_star': (4, 2),
        'sig_g': (3, 0.003),   # tight prior for smooth trend growth
        'sig_z': (4, 2),
        'sig_pi_tilde': (4, 2),
        'sig_x_tilde': (4, 2),
        # Yield measurement errors: Inverse-Wishart
        'sig_y_df': 15,
        'sig_y_scale': np.eye(12),
        # Survey measurement errors
        'sig_s_pi': (4, 2),
        'sig_s_sr': (4, 2),
        # Measurement sd (not a Gamma pair) for the optional trend-inflation
        # anchor: how tightly pi* is tied to the announced inflation target.
        # 0.5pp is loose enough to let the data move pi* around the target.
        'sig_anchor': 0.5,
    }


# ============================================================
# 6. GIBBS SAMPLER
# ============================================================

def draw_ig(shape_prior, scale_prior, resid):
    """Draw from Inverse-Gamma posterior given residuals."""
    n = len(resid)
    shape_post = shape_prior + n / 2
    scale_post = scale_prior + 0.5 * np.sum(resid**2)
    return np.sqrt(1.0 / np.random.gamma(shape_post, 1.0 / scale_post))


_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _norm_logpdf(x, loc, scale):
    """Scalar normal log-density. scipy's norm.logpdf costs ~20us a call in
    argument validation alone, and the MH steps call it six times an iteration."""
    z = (x - loc) / scale
    return -0.5 * z * z - np.log(scale) - _LOG_SQRT_2PI


def _mvn_draw(mean, cov):
    """
    Draw from N(mean, cov) via a Cholesky factor, falling back to an
    eigendecomposition if cov is only positive semi-definite. Equivalent in
    distribution to scipy's multivariate_normal.rvs and ~25x faster, which
    matters because the stationarity rejection loop calls it up to 200 times
    per Gibbs iteration.
    """
    mean = np.atleast_1d(mean)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        L = V * np.sqrt(np.clip(w, 0.0, None))
    return mean + L @ np.random.randn(len(mean))


def draw_normal_regression(y, X, prior_mean, prior_var, sig2):
    """Draw coefficients from Normal posterior (conjugate)."""
    prior_prec = inv(prior_var)
    data_prec = X.T @ X / sig2
    post_var = inv(prior_prec + data_prec)
    post_mean = post_var @ (prior_prec @ prior_mean + X.T @ y / sig2)
    return _mvn_draw(post_mean, post_var)


def initialize_params(priors):
    """Draw initial parameters from priors."""
    p = {}
    p['a1'] = np.random.normal(*priors['a1'])
    p['a2'] = np.random.normal(*priors['a2'])
    p['a3'] = np.random.normal(*priors['a3'])
    p['b1'] = np.random.normal(*priors['b1'])
    p['b2'] = np.random.normal(*priors['b2'])
    p['aL'] = np.random.normal(*priors['aL'])
    p['c'] = np.random.normal(*priors['c'])
    p['omega'] = np.random.normal(*priors['omega'])
    p['bias_s_pi'] = np.random.normal(*priors['bias_s_pi'])

    sc = _mvn_draw(priors['SC_mean'], priors['SC_var'])
    p['a10'], p['a11'], p['a12'], p['a13'], p['a14'] = sc[:5]
    p['a20'], p['a21'], p['a22'], p['a23'], p['a24'] = sc[5:]
    
    for key in ['sig_Lc','sig_S','sig_C','sig_pi_star','sig_x_star','sig_g','sig_z',
                'sig_pi_tilde','sig_x_tilde','sig_s_pi','sig_s_sr']:
        shape, scale = priors[key]
        p[key] = np.sqrt(scale / (shape - 1)) if shape > 1 else 0.5

    p['sig_y'] = 0.1 * np.ones(12)  # for 12 yields (excluding 1Q)
    
    return p


def check_stationarity(params, lam):
    """
    Check that cyclical dynamics are stationary.

    Checks both the isolated 2x2 slope-curvature VAR and, crucially, the
    full coupled {L_tilde, S, C, pi_tilde, x_tilde} block (+ their lags):
    the IS-curve real-rate-gap substitution (Appendix A.1) couples these
    together via a1, a2, a3, b1, b2, and this coupled block can be
    explosive even when the isolated 2x2 sub-VAR and aL are each
    individually stable. (Verified empirically: ~71% of draws that passed
    only the old, narrower check had a genuinely explosive eigenvalue.)
    The four unit-root states (pi*, x*, g, z) are intentionally excluded
    here — they are random walks by design and must stay at modulus 1.
    """
    A_sc = np.array([[params['a11'], params['a12']],
                     [params['a21'], params['a22']]])
    if np.abs(eigvals(A_sc)).max() >= 1:
        return False
    if np.abs(params['aL']) >= 1:
        return False
    # Formal stationarity is not enough: the unconditional mean the VAR implies
    # has to be a yield curve, not the (I - A_sc)^-1 singularity. See SC_BAR_MAX.
    S_bar, C_bar = sc_unconditional_mean(params)
    if abs(S_bar) > SC_BAR_MAX or abs(C_bar) > SC_BAR_MAX:
        return False
    # c is restricted to (0, 1): negative growth loadings are economically
    # meaningless, and c >= 1 is the LW restriction the data reject for India.
    if not (0.0 < params['c'] < 1.0):
        return False

    p = params
    _, ts1, tc1 = ns_loadings(lam, 1)
    a3 = p['a3']
    # Coupled block, same order as build_state_space's F rows/cols
    # [0,1,2,7,8,9,10,11,12,13]: L_tilde, S, C, pi_tilde, x_tilde, + lags
    Fc = np.zeros((10, 10))
    Fc[0, 0] = p['aL']
    Fc[1, 1] = p['a11']; Fc[1, 2] = p['a12']; Fc[1, 3] = p['a13']; Fc[1, 4] = p['a14']
    Fc[2, 1] = p['a21']; Fc[2, 2] = p['a22']; Fc[2, 3] = p['a23']; Fc[2, 4] = p['a24']
    Fc[3, 3] = p['b1']; Fc[3, 4] = p['b2']
    Fc[4, 0] = a3; Fc[4, 1] = a3*ts1; Fc[4, 2] = a3*tc1
    Fc[4, 3] = -a3*p['b1']; Fc[4, 4] = p['a1'] - a3*p['b2']
    Fc[4, 5] = a3; Fc[4, 6] = a3*ts1; Fc[4, 7] = a3*tc1
    Fc[4, 8] = -a3*p['b1']; Fc[4, 9] = p['a2'] - a3*p['b2']
    Fc[5, 0] = 1.0; Fc[6, 1] = 1.0; Fc[7, 2] = 1.0; Fc[8, 3] = 1.0; Fc[9, 4] = 1.0

    if np.abs(eigvals(Fc)).max() >= 1 - 1e-8:
        return False
    return True


def hp_filter(y, lam_hp=1600.0):
    """Two-sided Hodrick-Prescott filter. Returns (trend, cycle)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    # Second-difference penalty matrix
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i], D[i, i+1], D[i, i+2] = 1.0, -2.0, 1.0
    trend = solve(np.eye(n) + lam_hp * D.T @ D, y)
    return trend, y - trend


def initial_state(Y, params, lam, config):
    """
    Build (xi0, P0) for the simulation smoother from a two-sided HP filter,
    following the paper's Appendix D ("Initial states for the simulation
    smoother are based on estimates from a two-sided HP filter").

    Doing this properly matters more here than in the paper: log GDP is around
    1400 in 100*ln units, so a poorly centred x*_0 is a huge initial surprise,
    and India's trend growth (~1.5/quarter) is three times the US value.
    """
    N = config.n_states
    xi0 = np.zeros(N)
    P0 = np.eye(N)

    gdp = pd.Series(Y['gdp_real'].values).interpolate(limit_direction='both').values
    x_trend, x_cycle = hp_filter(gdp, 1600.0)
    growth = np.diff(x_trend)

    # Yield-curve factors at t=0 from a DNS cross-sectional OLS on the first
    # quarter with yields, so L_tilde/S/C do not start far from the data.
    taus = config.maturities
    y0 = np.array([Y[f'y_{t}'].values[0] if f'y_{t}' in Y.columns else np.nan
                   for t in taus], dtype=float)
    obs = ~np.isnan(y0)
    S0 = C0 = 0.0
    L0 = np.nanmean(y0) if obs.any() else 8.0
    if obs.sum() >= 3:
        X = np.column_stack([np.ones(obs.sum())] +
                            [np.array([ns_loadings(lam, t)[i] for t in np.array(taus)[obs]])
                             for i in (1, 2)])
        L0, S0, C0 = np.linalg.lstsq(X, y0[obs], rcond=None)[0]

    pi = Y['cpi_yoy'].values
    pi0 = np.nanmean(pi[:20]) if np.isfinite(pi[:20]).any() else 5.0

    xi0[0] = 0.0                 # L_tilde: zero-mean cyclical component
    xi0[1] = S0
    xi0[2] = C0
    xi0[3] = pi0                 # pi*: early-sample mean inflation
    xi0[4] = x_trend[0]          # x*: HP trend of log GDP
    xi0[5] = growth[0]           # g: HP trend growth, per quarter
    # z: residual so that r*_0 = 4*c*g_0 + z_0 matches the early-sample
    # ex-ante real short rate.
    r_early = np.nanmean(Y['y_1'].values[:20]) - pi0
    xi0[6] = r_early - 4.0 * params['c'] * xi0[5]
    xi0[7] = pi[0] - pi0 if np.isfinite(pi[0]) else 0.0
    xi0[8] = x_cycle[0]
    xi0[9:12] = xi0[[0, 1, 2]]   # lagged L_tilde, S, C
    xi0[12] = xi0[7]
    xi0[13] = xi0[8]

    # Moderate uncertainty on the cyclical factors; wide on the non-stationary
    # trends, whose levels are genuinely uncertain at t=0.
    P0[0, 0] = P0[1, 1] = P0[2, 2] = 1.0
    P0[3, 3] = 4.0     # pi*
    P0[4, 4] = 25.0    # x* (100*ln units, so 5pp sd)
    P0[5, 5] = 0.25    # g
    P0[6, 6] = 4.0     # z
    P0[7, 7] = P0[8, 8] = 4.0
    P0[9:, 9:] = np.eye(N - 9)
    return xi0, P0


def gibbs_sampler(Y, config=None, priors=None, verbose=True):
    """
    Main estimation routine.
    
    Y: pd.DataFrame with columns matching config.maturities as 'y_{tau}',
       plus 'gdp_real', 'cpi_yoy', and optionally 'survey_pi_star', 'survey_i_4q'.
    
    Returns dict with posterior draws of states and parameters.
    """
    if config is None:
        config = BGLConfig()
    if priors is None:
        priors = default_priors()
    
    T = len(Y)
    taus = config.maturities
    K = len(taus)
    N = config.n_states
    
    # Build observation matrix
    has_survey_pi = 'survey_pi_star' in Y.columns
    has_survey_sr = 'survey_i_4q' in Y.columns
    has_pi_anchor = 'pi_star_anchor' in Y.columns
    Y_mat = _observation_matrix(Y, config)
    n_obs = Y_mat.shape[1]

    lam = config.lambda_prior_mean

    # Initialize (redraw until the cyclical dynamics are stationary)
    params = initialize_params(priors)
    while not check_stationarity(params, lam):
        params = initialize_params(priors)
    if not has_survey_pi:
        params['sig_s_pi'] = 0.0
    if not has_survey_sr:
        params['sig_s_sr'] = 0.0
    params['sig_anchor'] = priors['sig_anchor'] if has_pi_anchor else 0.0

    # build_state_space appends optional rows in OPTIONAL_OBS order, so the
    # packed columns and the measurement rows must agree.
    assert n_obs == build_state_space(params, lam, config)[7], \
        "observable columns and measurement rows disagree"

    # Initial state, from a two-sided HP filter as in the paper's Appendix D.
    xi0, P0 = initial_state(Y, params, lam, config)

    # Storage
    n_keep = (config.n_draws - config.n_burnin) // config.thin
    xi_store = np.zeros((n_keep, T, N))
    rstar_store = np.zeros((n_keep, T))
    pi_star_store = np.zeros((n_keep, T))
    param_store = []
    lam_store = np.zeros(n_keep)
    
    save_idx = 0
    n_accept_lam = 0
    n_accept_c = 0
    n_accept_omega = 0

    # Adaptive MH proposal scales. Tuned during burn-in only (Robbins-Monro on
    # log-scale towards a 30% acceptance rate) and then frozen, so the retained
    # draws come from a time-homogeneous chain and ergodicity is preserved.
    mh_scale = {'lambda': config.lambda_mh_scale,
                'c': config.c_mh_scale,
                'omega': config.omega_mh_scale}
    mh_recent = {'lambda': [], 'c': [], 'omega': []}
    MH_TARGET, MH_TUNE_EVERY = 0.30, 100

    def _tune_scales():
        for key, hist in mh_recent.items():
            if len(hist) >= MH_TUNE_EVERY:
                rate = float(np.mean(hist))
                mh_scale[key] *= float(np.exp(rate - MH_TARGET))
                mh_scale[key] = float(np.clip(mh_scale[key], 1e-4, 5.0))
                hist.clear()
    
    for it in range(config.n_draws):
        # c as used for the state draw below. The MH update of c later in the
        # iteration changes params['c'], but g and z were drawn under this
        # value, so r* = 4*c*g + z must be stored with it for (xi, c) to be a
        # joint draw.
        c_used = params['c']
        # --- Step 1: Draw states ---
        try:
            F, mu_vec, Q, G, C_mat, gamma_vec, R, _ = build_state_space(params, lam, config)
            xi_draw = durbin_koopman_smoother(Y_mat, F, mu_vec, Q, C_mat, gamma_vec, R, xi0, P0)
        except Exception:
            continue
        
        # Extract states
        L_tilde = xi_draw[:, 0]
        S = xi_draw[:, 1]
        C = xi_draw[:, 2]
        pi_star = xi_draw[:, 3]
        x_star = xi_draw[:, 4]
        g = xi_draw[:, 5]
        z = xi_draw[:, 6]
        pi_tilde = xi_draw[:, 7]
        x_tilde = xi_draw[:, 8]
        
        # --- Step 2: Draw parameters (conditional on states) ---
        params_prev = copy.deepcopy(params)

        # 2a-2d draw the structural coefficients that feed the coupled
        # stationarity check (check_stationarity's Fc block). Each conjugate
        # regression's posterior has no stationarity truncation built in, so
        # a single draw is often rejected. Rather than draw once and revert
        # the whole iteration on failure (which — verified empirically —
        # rejects ~100% of proposals here and freezes every structural
        # parameter at its initial value for the entire chain), redraw the
        # 2a-2d bundle from the same conditional posteriors (rejection
        # sampling from a truncated distribution) until one satisfies the
        # coupled stationarity constraint.
        _, ts1, tc1 = ns_loadings(lam, 1)
        S_bar, C_bar = sc_unconditional_mean(params_prev)
        # The level part of the real rate gap. The b1/b2 terms are added inside
        # the attempt loop, after the Phillips curve is drawn, because r_gap
        # depends on them.
        r_gap_level = L_tilde + ts1*(S - S_bar) + tc1*(C - C_bar) + params_prev['omega']

        is_prior_mean = np.array([priors['a1'][0], priors['a2'][0], priors['a3'][0]])
        is_prior_var = np.diag([priors['a1'][1]**2, priors['a2'][1]**2, priors['a3'][1]**2])

        y_pc = pi_tilde[1:]
        X_pc = np.column_stack([pi_tilde[:-1], x_tilde[:-1]])
        pc_prior_mean = np.array([priors['b1'][0], priors['b2'][0]])
        pc_prior_var = np.diag([priors['b1'][1]**2, priors['b2'][1]**2])

        y_sc = np.column_stack([S[1:], C[1:]])
        X_sc = np.column_stack([
            np.ones(T-1), S[:-1], C[:-1], pi_tilde[:-1], x_tilde[:-1]
        ])

        accepted = False
        for _attempt in range(200):
            params_try = copy.deepcopy(params_prev)

            # 2a. Phillips curve. Drawn BEFORE the IS curve because the real
            # rate gap r_tilde = ... - b1*pi_tilde - b2*x_tilde depends on
            # b1, b2; drawing the IS curve first would condition it on stale
            # coefficients.
            resid_pc = y_pc - X_pc @ np.array([params_try['b1'], params_try['b2']])
            sig2_pc = draw_ig(*priors['sig_pi_tilde'], resid_pc)**2
            pc_coefs = draw_normal_regression(y_pc, X_pc, pc_prior_mean, pc_prior_var, sig2_pc)
            params_try['b1'], params_try['b2'] = pc_coefs
            params_try['sig_pi_tilde'] = np.sqrt(sig2_pc)

            # 2b. IS curve: x_tilde_t = a1*x_tilde_{t-1} + a2*x_tilde_{t-2}
            #                           + a3*(r_gap_{t-1}+r_gap_{t-2}) + eps
            r_gap = (r_gap_level - params_try['b1']*pi_tilde
                     - params_try['b2']*x_tilde)
            y_is = x_tilde[2:]
            X_is = np.column_stack([
                x_tilde[1:-1],
                x_tilde[:-2],
                r_gap[1:-1] + r_gap[:-2]
            ])
            resid_is = y_is - X_is @ np.array([params_try['a1'], params_try['a2'], params_try['a3']])
            sig2_is = draw_ig(*priors['sig_x_tilde'], resid_is)**2
            is_coefs = draw_normal_regression(y_is, X_is, is_prior_mean, is_prior_var, sig2_is)
            params_try['a1'], params_try['a2'], params_try['a3'] = is_coefs
            params_try['sig_x_tilde'] = np.sqrt(sig2_is)

            # 2c. Cyclical level AR(1)
            resid_Lc = L_tilde[1:] - params_try['aL'] * L_tilde[:-1]
            params_try['sig_Lc'] = draw_ig(*priors['sig_Lc'], resid_Lc)
            aL_draw = draw_normal_regression(
                L_tilde[1:], L_tilde[:-1].reshape(-1,1),
                np.array([priors['aL'][0]]), np.array([[priors['aL'][1]**2]]),
                params_try['sig_Lc']**2
            )
            if np.abs(aL_draw[0]) < 1:
                params_try['aL'] = aL_draw[0]

            # 2d. Slope-curvature VAR. sig_S/sig_C were previously a plug-in
            # np.var() of the residuals evaluated at the PRIOR MEAN
            # coefficients, which is neither a posterior draw nor even
            # centred on the current coefficients. They feed Omega_uu and
            # hence A(tau), so they are now drawn properly: condition the
            # variance on the current coefficients, then draw the
            # coefficients given that variance.
            for eq_idx, sig_key in enumerate(['sig_S', 'sig_C']):
                offset = eq_idx * 5
                y_eq = y_sc[:, eq_idx]
                prior_m = priors['SC_mean'][offset:offset+5]
                prior_v = priors['SC_var'][offset:offset+5, offset:offset+5]
                cur = np.array([params_try['a10'], params_try['a11'], params_try['a12'],
                                params_try['a13'], params_try['a14']]) if eq_idx == 0 else \
                      np.array([params_try['a20'], params_try['a21'], params_try['a22'],
                                params_try['a23'], params_try['a24']])
                sig_eq = draw_ig(*priors[sig_key], y_eq - X_sc @ cur)
                coefs_eq = draw_normal_regression(y_eq, X_sc, prior_m, prior_v, sig_eq**2)
                params_try[sig_key] = sig_eq
                if eq_idx == 0:
                    params_try['a10'], params_try['a11'], params_try['a12'], params_try['a13'], params_try['a14'] = coefs_eq
                else:
                    params_try['a20'], params_try['a21'], params_try['a22'], params_try['a23'], params_try['a24'] = coefs_eq

            if check_stationarity(params_try, lam):
                params = params_try
                accepted = True
                break

        if not accepted:
            params = params_prev

        # 2e. Random walk variances
        params['sig_pi_star'] = draw_ig(*priors['sig_pi_star'], np.diff(pi_star))
        params['sig_x_star'] = draw_ig(*priors['sig_x_star'], x_star[1:] - x_star[:-1] - g[:-1])
        params['sig_g'] = draw_ig(*priors['sig_g'], np.diff(g))
        params['sig_z'] = draw_ig(*priors['sig_z'], np.diff(z))
        
        # 2f. Yield measurement errors. The Inverse-Wishart prior of the paper
        # (df, scale) reduces to an Inverse-Gamma per maturity here because
        # sig_y is diagonal: shape = df/2, scale = diag(scale_matrix)/2.
        y_shape = priors['sig_y_df'] / 2.0
        y_scale = np.mean(np.diag(priors['sig_y_scale'])) / 2.0
        S_bar_upd, C_bar_upd = sc_unconditional_mean(params)
        A_upd = arbitrage_adj_many(
            lam, taus, params['sig_Lc'], params['sig_S'], params['sig_C'],
            params['sig_pi_star'], params['sig_g'], params['sig_z'],
            S_bar_upd, C_bar_upd, params['c'])
        for k in range(1, K):
            tau = taus[k]
            _, ts, tc = ns_loadings(lam, tau)
            # Recompute gamma for this tau
            adj = params['omega'] + A_upd[k]
            y_fit = adj + L_tilde + ts*S + tc*C + pi_star + 4*params['c']*g + z
            y_obs = Y_mat[:, k]
            mask = ~np.isnan(y_obs)
            if mask.sum() > 2:
                resid_y = y_obs[mask] - y_fit[mask]
                params['sig_y'][k-1] = draw_ig(y_shape, y_scale, resid_y)

        # 2g/2h build the same state space (2g only updates sig_s_pi and
        # bias_s_pi, neither of which enters the rows 2h reads), so build it
        # once and share.
        if has_survey_pi or has_survey_sr:
            F_s, mu_s, _, _, C_s, gamma_s, _, _ = build_state_space(params, lam, config)

        # 2g. Survey bias: survey_t = bias_s_pi + E_t pi_{t+4} + u_t.
        # Conjugate normal draw for the constant, given the states.
        if has_survey_pi:
            F4_s = np.linalg.matrix_power(F_s, 4)
            F_cum_s = np.eye(N) + F_s + F_s @ F_s + F_s @ F_s @ F_s
            exp_pi4 = (xi_draw @ (F4_s[3, :] + F4_s[7, :])
                       + (F_cum_s @ mu_s)[3] + (F_cum_s @ mu_s)[7])
            s_obs = Y_mat[:, K + 2]
            m_s = ~np.isnan(s_obs)
            if m_s.sum() > 2:
                resid_s = s_obs[m_s] - exp_pi4[m_s]
                params['sig_s_pi'] = draw_ig(*priors['sig_s_pi'], resid_s - resid_s.mean())
                b_mean, b_sd = priors['bias_s_pi']
                post_var = 1.0 / (1.0/b_sd**2 + m_s.sum()/params['sig_s_pi']**2)
                post_mean = post_var * (b_mean/b_sd**2
                                        + resid_s.sum()/params['sig_s_pi']**2)
                params['bias_s_pi'] = np.random.normal(post_mean, np.sqrt(post_var))

        # 2h. Short-rate survey measurement error. The row is already built in
        # build_state_space (paper eq. 17), so read the residual straight off it.
        if has_survey_sr:
            j = K + 2 + int(has_survey_pi)
            fit_sr = gamma_s[j] + xi_draw @ C_s[j]
            sr_obs = Y_mat[:, j]
            m_r = ~np.isnan(sr_obs)
            if m_r.sum() > 2:
                params['sig_s_sr'] = draw_ig(*priors['sig_s_sr'],
                                             sr_obs[m_r] - fit_sr[m_r])

        # --- Step 3: MH step for lambda ---
        # Baseline log-likelihood for (params, lam) — reused by step 3b below
        # instead of rebuilding the identical state space a second time.
        try:
            F_c, mu_c, Q_c, _, C_c, gamma_c, R_c, _ = build_state_space(params, lam, config)
            ll_curr = kalman_loglik(Y_mat, F_c, mu_c, Q_c, C_c, gamma_c, R_c, xi0, P0)
            ll_curr_valid = True
        except Exception:
            ll_curr_valid = False

        if config.estimate_lambda:
            lam_prop = lam + mh_scale['lambda'] * np.random.randn()
            lam_accepted = False
            if ll_curr_valid and 0.01 < lam_prop < 0.5:
                try:
                    F_p, mu_p, Q_p, _, C_p, gamma_p, R_p, _ = build_state_space(params, lam_prop, config)
                    ll_prop = kalman_loglik(Y_mat, F_p, mu_p, Q_p, C_p, gamma_p, R_p, xi0, P0)

                    # Prior ratio
                    lp_prop = _norm_logpdf(lam_prop, config.lambda_prior_mean, config.lambda_prior_sd)
                    lp_curr = _norm_logpdf(lam, config.lambda_prior_mean, config.lambda_prior_sd)

                    log_alpha = (ll_prop + lp_prop) - (ll_curr + lp_curr)
                    if np.log(np.random.rand()) < log_alpha:
                        lam = lam_prop
                        n_accept_lam += 1
                        lam_accepted = True
                        ll_curr = ll_prop  # carry forward as the new current likelihood
                except Exception:
                    pass
            mh_recent['lambda'].append(lam_accepted)

        # --- Steps 3b/3c: MH steps for c and omega ---
        # Both enter the measurement equation nonlinearly (c through the 4c
        # loading on g and through Omega_uu; omega through the yield intercepts
        # and the IS intercept), so neither has a conjugate conditional. Both
        # are drawn against the MARGINAL (states-integrated) Kalman likelihood,
        # which is valid inside the Gibbs loop because the states are redrawn
        # from the updated parameters at the top of the next iteration.
        # ll_curr already reflects (params, lam) after the lambda step above.
        def _mh_scalar(key, prop, prior, lower, upper):
            """Random-walk MH on one scalar parameter. Returns (accepted, ll)."""
            nonlocal ll_curr
            if not ll_curr_valid or not (lower < prop < upper):
                return False
            try:
                params_prop = copy.deepcopy(params)
                params_prop[key] = prop
                F_p, mu_p, Q_p, _, C_p, gamma_p, R_p, _ = build_state_space(params_prop, lam, config)
                ll_prop = kalman_loglik(Y_mat, F_p, mu_p, Q_p, C_p, gamma_p, R_p, xi0, P0)

                log_alpha = ((ll_prop + _norm_logpdf(prop, *prior))
                             - (ll_curr + _norm_logpdf(params[key], *prior)))
                if np.log(np.random.rand()) < log_alpha:
                    params[key] = prop
                    ll_curr = ll_prop
                    return True
            except Exception:
                pass
            return False

        acc_c = _mh_scalar('c', params['c'] + mh_scale['c'] * np.random.randn(),
                           priors['c'], 0.0, 1.0)
        n_accept_c += acc_c
        mh_recent['c'].append(acc_c)

        if config.estimate_omega:
            acc_w = _mh_scalar('omega', params['omega'] + mh_scale['omega'] * np.random.randn(),
                               priors['omega'], -3.0, 0.5)
            n_accept_omega += acc_w
            mh_recent['omega'].append(acc_w)

        # Adapt proposal scales during burn-in only, then freeze and reset the
        # counters so the reported acceptance rates describe the frozen chain.
        if it < config.n_burnin:
            _tune_scales()
            if it == config.n_burnin - 1:
                n_accept_lam = n_accept_c = n_accept_omega = 0

        # --- Store ---
        if it >= config.n_burnin and (it - config.n_burnin) % config.thin == 0:
            if save_idx < n_keep:
                xi_store[save_idx] = xi_draw
                rstar_store[save_idx] = 4 * c_used * g + z
                pi_star_store[save_idx] = pi_star
                p_store = params.copy()
                p_store['c_state'] = c_used   # c under which this row's states were drawn
                param_store.append(p_store)
                lam_store[save_idx] = lam
                save_idx += 1
        
        if verbose and (it + 1) % 1000 == 0:
            r_star_med = np.median(4*c_used*g + z)
            # Acceptance counters are reset when the proposal scales freeze at
            # the end of burn-in, so divide by draws since whichever came last.
            n_since = (it + 1) if it < config.n_burnin else (it + 1 - config.n_burnin)
            print(f"Iter {it+1}/{config.n_draws} | r*_median={r_star_med:5.2f} | "
                  f"lam={lam:.4f}(acc {n_accept_lam/n_since:.2f}) | "
                  f"c={params['c']:.3f}(acc {n_accept_c/n_since:.2f}) | "
                  f"omega={params['omega']:5.2f}(acc {n_accept_omega/n_since:.2f})")

    actual_saved = save_idx
    results = {
        'rstar': rstar_store[:actual_saved],        # (n_keep, T)
        'pi_star': pi_star_store[:actual_saved],
        'states': xi_store[:actual_saved],
        'params': param_store[:actual_saved],
        'lambda': lam_store[:actual_saved],
        'config': config,
        'dates': Y.index if hasattr(Y, 'index') else np.arange(T),
        # Post-burn-in acceptance rates, at the frozen proposal scales.
        'accept_rates': {
            'lambda': n_accept_lam / max(config.n_draws - config.n_burnin, 1),
            'c': n_accept_c / max(config.n_draws - config.n_burnin, 1),
            'omega': n_accept_omega / max(config.n_draws - config.n_burnin, 1),
        },
        'mh_scale': dict(mh_scale),
    }

    return results


# ============================================================
# 6b. IDENTIFICATION PROFILES & CONVERGENCE DIAGNOSTICS
# ============================================================

# Optional observables, in the order build_state_space appends their rows.
OPTIONAL_OBS = [('survey_pi_star', 'sig_s_pi'),
                ('survey_i_4q', 'sig_s_sr'),
                ('pi_star_anchor', 'sig_anchor')]


def _observation_matrix(Y, config):
    """Pack the observable DataFrame into the (T, n_obs) array the filter wants."""
    taus = config.maturities
    K = len(taus)
    present = [col for col, _ in OPTIONAL_OBS if col in Y.columns]
    Y_mat = np.full((len(Y), K + 2 + len(present)), np.nan)
    for k, tau in enumerate(taus):
        if f'y_{tau}' in Y.columns:
            Y_mat[:, k] = Y[f'y_{tau}'].values
    Y_mat[:, K] = Y['gdp_real'].values
    Y_mat[:, K+1] = Y['cpi_yoy'].values
    for j, col in enumerate(present):
        Y_mat[:, K + 2 + j] = Y[col].values
    return Y_mat


def profile_likelihood(Y, results, key, grid, config=None):
    """
    Profile the marginal (states-integrated) log-likelihood over one scalar
    parameter, holding all others at their posterior medians.

    This is the honest way to show what the data do and do not say about the
    weakly-identified parameters. For India:

      * `c`     -- a shallow interior peak near 0.4-0.5. Large c is clearly
                   rejected (about -3.5 log points at c = 1.5), but the
                   Laubach-Williams value c = 1 costs only ~1 log point, so it
                   is disfavoured rather than rejected; below 0.35 the profile
                   is nearly flat and the prior does the identifying work.
      * `omega` -- essentially flat, because a constant wedge in the yield
                   intercepts is collinear with the level of the random-walk
                   z over a finite sample. Its posterior reproduces the prior;
                   report the grid in `omega_grid()` rather than claiming it
                   is estimated.

    Returns a DataFrame with the grid, the log-likelihood and the difference
    from the best grid point.
    """
    if config is None:
        config = results['config']
    params = posterior_median_params(results)
    lam = float(np.median(results['lambda']))
    Y_mat = _observation_matrix(Y, config)
    xi0, P0 = initial_state(Y, params, lam, config)

    rows = []
    for v in grid:
        p = copy.deepcopy(params)
        if key == 'lambda':
            lam_use = v
        else:
            p[key] = v
            lam_use = lam
        try:
            F, mu_v, Q, _, C_m, gam, R, _ = build_state_space(p, lam_use, config)
            ll = kalman_loglik(Y_mat, F, mu_v, Q, C_m, gam, R, xi0, P0)
        except Exception:
            ll = np.nan
        rows.append({key: v, 'loglik': ll})
    out = pd.DataFrame(rows)
    out['delta'] = out['loglik'] - out['loglik'].max()
    return out


def posterior_median_params(results):
    """Element-wise posterior median of the stored parameter draws."""
    plist = results['params']
    p = copy.deepcopy(plist[-1])
    for k in p:
        vals = [d[k] for d in plist]
        p[k] = (np.median(np.array(vals), axis=0) if np.ndim(vals[0])
                else float(np.median(vals)))
    return p


def geweke(chain, first=0.1, last=0.5):
    """
    Geweke (1991) convergence z-score: compares the mean of the first `first`
    fraction of the chain with the mean of the last `last` fraction, scaled by
    their spectral-density-corrected standard errors. |z| < 2 is consistent
    with convergence.
    """
    x = np.asarray(chain, dtype=float)
    n = len(x)
    if n < 20:
        return np.nan
    a, b = x[:int(first*n)], x[int((1-last)*n):]

    def _spec_var(v):
        v = v - v.mean()
        m = max(1, int(len(v) ** (1/3)))  # Bartlett bandwidth
        s = np.dot(v, v) / len(v)
        for k in range(1, m + 1):
            w = 1.0 - k / (m + 1)
            s += 2 * w * np.dot(v[k:], v[:-k]) / len(v)
        return max(s, 1e-12) / len(v)

    return (a.mean() - b.mean()) / np.sqrt(_spec_var(a) + _spec_var(b))


def _geweke_scores(results, keys):
    rows = {}
    for k in keys:
        try:
            rows[k] = geweke([p[k] for p in results['params']])
        except (KeyError, TypeError):
            continue
    rows['lambda'] = geweke(results['lambda'])
    rows['r*_end'] = geweke(results['rstar'][:, -1])
    return rows


def convergence_report(results, keys=('c', 'omega', 'a1', 'a2', 'a3', 'b1', 'b2',
                                      'aL', 'sig_z', 'sig_g', 'bias_s_pi')):
    """Print Geweke z-scores for the scalar parameters plus r* at end of sample."""
    # On a pooled multi-chain result the concatenated draws are not a single
    # time-ordered chain, so a Geweke score over them would compare chain 1's
    # start with the last chains' ends. Report the worst |z| per chain instead.
    if 'chains' in results and len(results['chains']) > 1:
        per_chain = [_geweke_scores(r, keys) for r in results['chains']]
        rows = {k: max((d[k] for d in per_chain if k in d), key=abs)
                for k in per_chain[0]}
        # This is a maximum over len(per_chain) x len(rows) z-scores, so a few
        # exceedances of |z| > 2 are expected by chance even under perfect
        # convergence. Gelman-Rubin R-hat from run_chains() is the statistic to
        # judge convergence on; treat this panel as a per-chain sanity check.
        title = (f'Worst Geweke z across {len(per_chain)} chains '
                 f'(max of {len(per_chain) * len(rows)}; judge on R-hat)')
    else:
        rows = _geweke_scores(results, keys)
        title = 'Geweke convergence diagnostics (|z| < 2 is fine)'

    print(f"\n{title:^58}")
    print("-" * 58)
    for k, z in rows.items():
        flag = '' if abs(z) < 2 else '   <-- not converged'
        print(f"  {k:12s} z = {z:7.2f}{flag}")
    return rows


def gelman_rubin(chains):
    """
    Gelman-Rubin potential scale reduction factor R-hat for a list of
    equal-length scalar chains. R-hat < 1.1 is the usual convergence threshold.
    """
    x = np.asarray(chains, dtype=float)      # (m chains, n draws)
    m, n = x.shape
    if m < 2 or n < 4:
        return np.nan
    chain_means = x.mean(axis=1)
    W = x.var(axis=1, ddof=1).mean()          # within-chain variance
    B = n * chain_means.var(ddof=1)           # between-chain variance
    if W <= 0:
        return np.nan
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def split_rhat(chains):
    """Split-R-hat: each chain is cut in half and the halves are treated as
    separate chains, so within-chain non-stationarity also inflates the value."""
    x = np.asarray(chains, dtype=float)
    h = x.shape[1] // 2
    if h < 4:
        return np.nan
    return gelman_rubin(np.vstack([x[:, :h], x[:, h:2 * h]]))


def effective_sample_size(x):
    """ESS of one scalar chain: FFT autocorrelation, Geyer initial-positive-
    sequence truncation (sum autocorrelations in adjacent pairs until a pair
    sums below zero)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return np.nan
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return np.nan
    ac = ac / ac[0]
    s = 0.0
    for k in range(1, n - 1):
        if ac[k] + ac[k + 1] < 0:
            break
        s += ac[k]
    return float(n / max(1.0 + 2.0 * s, 1e-12))


def _run_one_chain(args):
    """One seeded chain. Module-level so it can be sent to a worker process."""
    seed, Y, config, priors, verbose = args
    np.random.seed(seed)
    return seed, gibbs_sampler(Y, config=config, priors=priors, verbose=verbose)


def run_chains(Y, n_chains=4, config=None, priors=None, seeds=None, verbose=False,
               n_jobs=None):
    """
    Run several independent chains from different random starts and pool them.

    The paper confirms convergence this way ("multiple chains starting from
    different initial values"), and it is the right check here because the two
    weakest-identified parameters -- c and a3 -- can show borderline Geweke
    scores within a single chain simply because their conditional posteriors
    are nearly flat.

    The chains are independent given their seeds, so they run in parallel
    processes by default (n_jobs, capped at the core count). Each worker seeds
    numpy with its own seed, so the pooled result does not depend on n_jobs;
    pass n_jobs=1 to run them sequentially in this process.

    Returns the pooled results dict, with an extra 'rhat' entry and a
    'chains' list of the individual runs.
    """
    if config is None:
        config = BGLConfig()
    if seeds is None:
        seeds = list(range(n_chains))
    if n_jobs is None:
        n_jobs = min(len(seeds), os.cpu_count() or 1)
    n_jobs = max(1, min(int(n_jobs), len(seeds)))

    jobs = [(s, Y, config, priors, verbose) for s in seeds]
    if n_jobs == 1:
        runs = []
        for i, job in enumerate(jobs):
            print(f"  chain {i+1}/{len(seeds)} (seed {job[0]})...", flush=True)
            runs.append(_run_one_chain(job)[1])
    else:
        print(f"  running {len(seeds)} chains on {n_jobs} processes...", flush=True)
        by_seed = {}
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_run_one_chain, job) for job in jobs]
            for done, fut in enumerate(as_completed(futures), start=1):
                seed, res = fut.result()
                by_seed[seed] = res
                print(f"  chain finished (seed {seed}) — {done}/{len(seeds)}",
                      flush=True)
        runs = [by_seed[s] for s in seeds]   # keep the requested seed order

    pooled = copy.deepcopy(runs[0])
    for key in ('rstar', 'pi_star', 'states'):
        pooled[key] = np.concatenate([r[key] for r in runs], axis=0)
    pooled['lambda'] = np.concatenate([r['lambda'] for r in runs])
    pooled['params'] = [p for r in runs for p in r['params']]

    n = min(len(r['params']) for r in runs)
    rhat = {}
    for key in ('c', 'omega', 'a1', 'a2', 'a3', 'b1', 'b2', 'aL',
                'sig_z', 'sig_g', 'bias_s_pi'):
        try:
            rhat[key] = gelman_rubin([[p[key] for p in r['params'][:n]] for r in runs])
        except (KeyError, TypeError):
            continue
    rhat['lambda'] = gelman_rubin([r['lambda'][:n] for r in runs])
    rhat['r*_end'] = gelman_rubin([r['rstar'][:n, -1] for r in runs])
    pooled['rhat'] = rhat
    pooled['chains'] = runs

    # Split-R-hat and ESS (summed over chains) on the same scalar series.
    series = {}
    for key in ('c', 'omega', 'a1', 'a2', 'a3', 'b1', 'b2', 'aL',
                'sig_z', 'sig_g', 'bias_s_pi'):
        try:
            series[key] = [[p[key] for p in r['params'][:n]] for r in runs]
        except (KeyError, TypeError):
            continue
    series['lambda'] = [r['lambda'][:n] for r in runs]
    series['r*_end'] = [r['rstar'][:n, -1] for r in runs]
    split, ess = {}, {}
    for key, ch in series.items():
        split[key] = split_rhat(ch)
        ess[key] = float(np.nansum([effective_sample_size(c) for c in ch]))
    pooled['split_rhat'] = split
    pooled['ess'] = ess

    print(f"\n{'Gelman-Rubin R-hat (< 1.1 is converged)':^54}")
    print("-" * 54)
    for k, v in rhat.items():
        flag = '' if v < 1.1 else '   <-- not converged'
        print(f"  {k:12s} R-hat = {v:6.3f}   split = {split.get(k, np.nan):6.3f}"
              f"   ESS = {ess.get(k, np.nan):7.0f}{flag}")
    return pooled


def _run_one_omega(args):
    """One fixed-omega re-estimation. Module-level so it can be sent to a worker."""
    w, Y, cfg, pri, seed, verbose = args
    np.random.seed(seed)
    res = gibbs_sampler(Y, config=cfg, priors=pri, verbose=verbose)
    return w, np.median(res['rstar'], axis=0)


def omega_grid(Y, config=None, priors=None, grid=(0.0, -0.5, -1.0, -1.5),
               n_draws=6000, n_burnin=4000, thin=5, seed=0, verbose=False,
               n_jobs=None):
    """
    Re-estimate the model with the real-rate-gap wedge omega FIXED at each
    value in `grid`, and return a DataFrame of the resulting r* paths.

    This is the headline robustness exercise for the India adaptation. omega=0
    recovers the paper's model exactly, in which mean(r*) is identically equal
    to the sample-mean ex-ante real short rate (about -0.01% for India
    2000-2025). Non-zero omega buys a higher r* level, but omega is only weakly
    identified, so the grid -- not the posterior for omega -- is what should be
    reported.
    """
    if config is None:
        config = BGLConfig()

    jobs = []
    for w in grid:
        cfg = copy.deepcopy(config)
        cfg.estimate_omega = False
        cfg.n_draws, cfg.n_burnin, cfg.thin = n_draws, n_burnin, thin
        pri = copy.deepcopy(priors) if priors is not None else default_priors()
        pri['omega'] = (w, 1e-6)   # effectively fixed
        jobs.append((w, Y, cfg, pri, seed, verbose))

    # Independent re-estimations with their own fixed seed, so like run_chains
    # these parallelize without changing the result.
    if n_jobs is None:
        n_jobs = min(len(jobs), os.cpu_count() or 1)
    n_jobs = max(1, min(int(n_jobs), len(jobs)))

    if n_jobs == 1:
        paths = [_run_one_omega(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            done = dict(pool.map(_run_one_omega, jobs))
        paths = [(w, done[w]) for w in grid]

    out = {}
    for w, path in paths:
        out[f'omega={w:+.1f}'] = path
        print(f"  omega={w:+.1f}: r* mean={np.mean(path):+.2f}%, "
              f"end={path[-1]:+.2f}%")
    return pd.DataFrame(out, index=Y.index)


# ============================================================
# 7. TERM PREMIUM COMPUTATION
# ============================================================

def compute_term_premia(results, tau=40):
    """
    Compute term premium for maturity tau (quarters).
    TP(tau) = fitted_yield(tau) - (1/tau) * sum_{h=0}^{tau-1} E_t[i_{t+h}]
    """
    states = results['states']  # (n_draws, T, N)
    params_list = results['params']
    lam_arr = results['lambda']
    config = results['config']
    
    n_draws, T, N = states.shape
    tp = np.zeros((n_draws, T))
    
    for d in range(n_draws):
        p = params_list[d]
        lam = lam_arr[d]
        F, mu_vec, _, _, C_mat, gamma_vec, _, _ = build_state_space(p, lam, config)
        
        _, ts_tau, tc_tau = ns_loadings(lam, tau)
        S_bar, C_bar = sc_unconditional_mean(p)
        adj = p['omega'] + arbitrage_adj(lam, tau, p['sig_Lc'], p['sig_S'], p['sig_C'],
                            p['sig_pi_star'], p['sig_g'], p['sig_z'],
                            S_bar, C_bar, p['c'])

        C1 = C_mat[0]  # short rate loadings

        for t in range(T):
            xi_t = states[d, t]

            # Fitted yield
            y_fit = adj + xi_t[0] + ts_tau*xi_t[1] + tc_tau*xi_t[2] + xi_t[3] + 4*p['c']*xi_t[5] + xi_t[6]
            
            # Expected short rates
            exp_sum = 0.0
            xi_h = xi_t.copy()
            for h in range(tau):
                i_h = gamma_vec[0] + C1 @ xi_h
                exp_sum += i_h
                xi_h = mu_vec + F @ xi_h
            
            tp[d, t] = y_fit - exp_sum / tau
    
    return tp


# ============================================================
# 8. CONVENIENCE: LOAD & PREP DATA
# ============================================================

def prepare_data(filepath):
    """
    Load CSV and validate columns.
    Expected: date column + yield columns + gdp_real + cpi_yoy + optional surveys.
    """
    df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
    df = df.sort_index()
    
    required = ['gdp_real', 'cpi_yoy', 'y_1']
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"
    
    print(f"Data loaded: {len(df)} quarters, {df.index[0]} to {df.index[-1]}")
    print(f"Available yields: {[c for c in df.columns if c.startswith('y_')]}")
    print(f"Surveys: {[c for c in df.columns if c.startswith('survey')]}")

    return df


def _fy_quarter_to_date(label):
    """Convert an Indian-fiscal-year quarter label (e.g. '1996-97 Q1') to its
    calendar quarter-end timestamp. FY Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec,
    Q4=Jan-Mar (of the following calendar year)."""
    m = re.match(r'(\d{4})-\d{2}\s*Q(\d)', label.strip())
    start_year, q = int(m.group(1)), int(m.group(2))
    month_day = {1: (6, 30), 2: (9, 30), 3: (12, 31), 4: (3, 31)}[q]
    year = start_year if q != 4 else start_year + 1
    return pd.Timestamp(year=year, month=month_day[0], day=month_day[1])


def _survey_pi_bias(macro_df):
    """
    Report the raw forecast-error bias of the household inflation survey:
    mean(survey_t - cpi_yoy_{t+4}) over the quarters where both the survey and
    the realized 4-quarter-ahead CPI are observed.

    Diagnostic only -- the survey is NOT debiased here. The bias is estimated
    inside the model as `bias_s_pi` (see build_state_space), because a ~5pp
    pre-subtracted constant does a great deal of work on pi* and therefore on
    the real rate and on omega; making it a parameter lets that uncertainty
    propagate into the r* credible set instead of being assumed away.
    """
    realized_4q_ahead = macro_df['cpi_yoy'].shift(-4)
    overlap = macro_df['survey_pi_star'].notna() & realized_4q_ahead.notna()
    if overlap.sum() == 0:
        return np.nan
    bias = (macro_df.loc[overlap, 'survey_pi_star'] - realized_4q_ahead[overlap]).mean()
    print(f"survey_pi_star raw bias={bias:.2f}pp (n={int(overlap.sum())} overlapping "
          f"quarters) -- estimated in-model as bias_s_pi, not subtracted here")
    return bias


def load_india_data(path='Data.xlsx', config=None, start_date='2000-01-01'):
    """
    Load and merge the India Macro + Yield Data sheets into the quarterly
    DataFrame shape expected by gibbs_sampler (index=date, columns
    gdp_real, cpi_yoy, y_1...y_40, survey_pi_star).

    Missing observations (inflation/survey gaps) are left as NaN — the
    Kalman filter / Durbin-Koopman smoother already handle missing
    observations natively.

    The sample starts at `start_date` (default 2000Q1, where the yield curve
    data begin). The 1996-2000 stretch has no yield data at all and, for much
    of it, no CPI either, so L_tilde/S/C/pi*/z would be identified only by the
    GDP series for 15+ quarters. Pass start_date=None to keep the full history.

    Note on the level of r*: this model makes the real rate gap stationary, so
    with omega = 0 the sample mean of r* is identically the sample mean
    ex-ante real short rate — about -0.01% over 2000-2025 (3m T-bill 6.38%,
    CPI 6.19%). It is +0.45% on 2011-2025 and +0.80% on 2015-2025, so the
    start date materially moves the level of r*. See omega_grid().
    """
    if config is None:
        config = BGLConfig()

    # --- Macro sheet ---
    macro = pd.read_excel(path, sheet_name='Macro')
    macro['date'] = macro['Quarter'].apply(_fy_quarter_to_date)
    macro = macro.set_index('date').sort_index()
    macro = macro.rename(columns={
        'log.GDP.SA': 'gdp_real',
        'Inflation': 'cpi_yoy',
        'Expectation.Inflation.IESH': 'survey_pi_star',
    })[['gdp_real', 'cpi_yoy', 'survey_pi_star']]

    # 'log.GDP.SA' is log10 of real GDP in Rs crore per quarter, NOT 100*ln.
    # Check: 10**6.672 = 4,699,071 crore/quarter -> Rs 188 lakh crore/yr, which
    # is India's FY25 real GDP, and it implies 6.3%/yr average growth. Read as
    # a natural log it would imply 2.7%/yr, which is wrong for India.
    # The model needs the output gap in percent, i.e. 100*ln(GDP), so convert
    # with 100*ln(10) = 230.2585. Without this g is ~0.007/quarter and
    # 4*c*g ~ 0.01%, and the IS/Phillips coefficients (a3, b2) are calibrated
    # for percent units.
    macro['gdp_real'] = LOG10_TO_PCT_LOG * macro['gdp_real']
    _survey_pi_bias(macro)

    # --- Yield Data sheet ---
    yields = pd.read_excel(path, sheet_name='Yield Data')
    yields = yields.set_index('Date').sort_index()
    yields = yields.resample('QE').last()

    yield_df = pd.DataFrame(index=yields.index)
    for tau in config.maturities:
        months = tau * 3
        col = f'{months}m'
        if col in yields.columns:
            yield_df[f'y_{tau}'] = yields[col]
        else:
            # No direct column (e.g. 9m) — linearly interpolate from
            # neighboring maturities on the same daily grid, then resample.
            lower = max((c for c in yields.columns if int(c[:-1]) < months),
                        key=lambda c: int(c[:-1]))
            upper = min((c for c in yields.columns if int(c[:-1]) > months),
                        key=lambda c: int(c[:-1]))
            lo_m, up_m = int(lower[:-1]), int(upper[:-1])
            w = (months - lo_m) / (up_m - lo_m)
            yield_df[f'y_{tau}'] = (1 - w) * yields[lower] + w * yields[upper]

    df = macro.join(yield_df, how='outer').sort_index()

    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]

    # The yield sheet runs past the macro sheet, leaving trailing quarters with
    # no GDP or CPI at all. Those add nothing and let x*/g drift unconstrained.
    df = df.loc[:df['gdp_real'].last_valid_index()]

    required = ['gdp_real', 'cpi_yoy', 'y_1']
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"

    # Guard the log-base conversion: if 'log.GDP.SA' were ever supplied as a
    # natural log or already rescaled, these bounds catch it immediately
    # rather than silently producing a nonsensical r*.
    gdp = df['gdp_real'].dropna()
    assert 1200 < gdp.min() and gdp.max() < 1700, (
        f"gdp_real out of expected range for 100*ln(India real GDP): "
        f"[{gdp.min():.1f}, {gdp.max():.1f}], expected within (1200, 1700). "
        f"Check the log base of 'log.GDP.SA'.")
    ann_growth = (gdp.iloc[-1] - gdp.iloc[0]) / (len(gdp) - 1) * 4
    assert 5.0 < ann_growth < 7.5, (
        f"implied average annual real growth {ann_growth:.2f}% outside (5.0, 7.5)")
    print(f"gdp_real rescaled to 100*ln: [{gdp.min():.1f}, {gdp.max():.1f}], "
          f"avg annual growth {ann_growth:.2f}%")

    print(f"Data loaded: {len(df)} quarters, {df.index[0]} to {df.index[-1]}")
    print(f"Available yields: {[c for c in df.columns if c.startswith('y_')]}")
    print(f"Surveys: {[c for c in df.columns if c.startswith('survey')]}")
    print(f"Missing counts:\n{df.isna().sum()}")

    return df


# ============================================================
# 9. RESULTS SUMMARY
# ============================================================

def summarize_results(results):
    """Print summary statistics of posterior estimates."""
    rstar = results['rstar']
    pi_star = results['pi_star']
    dates = results['dates']
    
    n_draws = rstar.shape[0]
    print(f"\n{'='*60}")
    print(f"BGL Model — Posterior Summary ({n_draws} draws)")
    print(f"{'='*60}")
    
    # r* at end of sample
    rstar_end = rstar[:, -1]
    print(f"\nr* (end of sample):")
    print(f"  Median: {np.median(rstar_end):.2f}%")
    print(f"  5th:    {np.percentile(rstar_end, 5):.2f}%")
    print(f"  95th:   {np.percentile(rstar_end, 95):.2f}%")
    
    pi_end = pi_star[:, -1]
    print(f"\nTrend inflation (end of sample):")
    print(f"  Median: {np.median(pi_end):.2f}%")
    print(f"  5th:    {np.percentile(pi_end, 5):.2f}%")
    print(f"  95th:   {np.percentile(pi_end, 95):.2f}%")
    
    # i* = r* + pi*
    istar = rstar + pi_star
    istar_end = istar[:, -1]
    print(f"\ni* = r* + pi* (end of sample):")
    print(f"  Median: {np.median(istar_end):.2f}%")
    
    # Lambda
    print(f"\nDNS lambda: median={np.median(results['lambda']):.4f}")

    def _q(key, label, fmt="{:.3f}"):
        v = np.array([p[key] for p in results['params']])
        print(f"\n{label}:")
        print(f"  Median: {fmt.format(np.median(v))}")
        print(f"  5th:    {fmt.format(np.percentile(v, 5))}")
        print(f"  95th:   {fmt.format(np.percentile(v, 95))}")
        return v

    # c: growth-to-r* scaling (r* = 4*c*g + z). The data bound c from above
    # (they reject the LW value c=1) but not from below -- see
    # profile_likelihood(Y, results, 'c', ...).
    _q('c', "Growth-to-r* scaling c (r*=4*c*g+z)")
    # omega: constant real-rate-gap wedge, mean(r*) = mean(r) - omega.
    # Weakly identified; its posterior largely reproduces the prior.
    _q('omega', "Real-rate-gap wedge omega (mean(r*) = mean(r) - omega)")
    if 'bias_s_pi' in results['params'][0]:
        _q('bias_s_pi', "Household survey inflation bias (pp)")

    if 'accept_rates' in results:
        print("\nMH acceptance (post-burn-in, target 0.2-0.4): "
              + ", ".join(f"{k}={v:.2f}" for k, v in results['accept_rates'].items()))

    return {
        'rstar_median': np.median(rstar, axis=0),
        'rstar_5': np.percentile(rstar, 5, axis=0),
        'rstar_95': np.percentile(rstar, 95, axis=0),
        'pi_star_median': np.median(pi_star, axis=0),
        'istar_median': np.median(istar, axis=0),
        'dates': dates,
    }


# ============================================================
# 9b. PLOTS
# ============================================================

# One hue per series, taken in fixed slot order from the chart palette (never
# cycled). Validated for colour-vision deficiency against the chart surface:
# worst pair dE 24.7 (protan), 33.6 (normal vision), both well clear of the
# floors. The remaining entries are chrome, deliberately recessive so the data
# carries the ink.
PLOT_COLOR = {'rstar': '#2a78d6', 'pi_star': '#eb6834'}
PLOT_INK = {'primary': '#0b0b0b', 'secondary': '#52514e', 'muted': '#898781',
            'grid': '#e1e0d9', 'axis': '#c3c2b7', 'surface': '#fcfcfb'}


def plot_trends(results, out_path='rstar_pistar.png', band=(5, 95), show=False):
    """
    Plot the time evolution of r* and trend inflation pi*.

    One panel each, sharing the date axis. Both series are in annualized
    percent, but they answer different questions and each carries its own
    credible band; drawn on one set of axes the two bands would overplot in
    exactly the region — the 2000s, where both are wide — that the reader most
    needs to see.

    results: the dict from gibbs_sampler() or run_chains().
    band:    percentile pair drawn as the shaded credible interval.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    x = results['dates']
    is_time = isinstance(x, pd.DatetimeIndex)
    if not is_time:
        x = np.asarray(x)

    panels = [('rstar', 'Natural rate of interest  $r^*$'),
              ('pi_star', 'Trend inflation  $\\pi^*$')]

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 7), sharex=True)
    fig.patch.set_facecolor(PLOT_INK['surface'])

    for ax, (key, title) in zip(axes, panels):
        draws = results[key]                       # (n_draws, T)
        med = np.median(draws, axis=0)
        lo, hi = np.percentile(draws, band, axis=0)
        color = PLOT_COLOR[key]

        ax.set_facecolor(PLOT_INK['surface'])
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, med, color=color, linewidth=2.0, solid_capstyle='round')
        # Zero is a reference the reader needs where the band actually reaches
        # it (r*, whose lower tail is negative for India). On a series that
        # never goes near it — pi* sits at 4-8% — the line would only stretch
        # the axis down to an uninformative zero.
        if lo.min() < 0.0 < hi.max():
            ax.axhline(0.0, color=PLOT_INK['axis'], linewidth=1.0,
                       linestyle=(0, (2, 3)), zorder=0)

        # Direct-label the end-of-sample value only — a number on every point
        # would bury the path it is meant to annotate.
        ax.annotate(f'{med[-1]:.2f}%', xy=(x[-1], med[-1]),
                    xytext=(7, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=10,
                    fontweight='bold', color=PLOT_INK['primary'])

        # The title names the single series, so no legend box is needed; the
        # band is described here instead.
        ax.set_title(f'{title}   —   posterior median, '
                     f'{band[1] - band[0]}% credible interval',
                     loc='left', fontsize=11.5, color=PLOT_INK['primary'], pad=10)
        ax.set_ylabel('Percent', fontsize=10, color=PLOT_INK['secondary'])
        ax.grid(axis='y', color=PLOT_INK['grid'], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(PLOT_INK['axis'])
        ax.tick_params(colors=PLOT_INK['muted'], labelsize=9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(PLOT_INK['secondary'])
        # Leave room on the right for the end-of-sample labels.
        ax.set_xlim(x[0], x[-1] + (x[-1] - x[0]) * 0.05)

    if is_time:
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    n_draws = results['rstar'].shape[0]
    fig.suptitle(f'India: estimated trends, {n_draws:,} posterior draws',
                 x=0.005, ha='left', fontsize=9, color=PLOT_INK['muted'], y=0.995)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved → {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================
# 10. MAIN
# ============================================================

N_CHAINS = 4   # independent chains from different seeds, pooled for reporting


if __name__ == '__main__':
    import os
    import sys

    if len(sys.argv) < 2:
        default_path = 'Data.xlsx'
        if os.path.exists(default_path):
            print(f"No data file argument given — defaulting to '{default_path}'.\n")
            data_path = default_path
        else:
            print(__doc__)
            print("\nUsage: python run_model.py <data.csv|Data.xlsx>")
            print("\nRequired CSV columns:")
            print("  date, gdp_real, cpi_yoy, y_1, y_2, ..., y_40")
            print("Optional: survey_pi_star, survey_i_4q")
            print("\nFor Excel input (Data.xlsx), expects 'Macro' and 'Yield Data' sheets.")
            sys.exit(0)
    else:
        data_path = sys.argv[1]

    df = load_india_data(data_path) if data_path.endswith('.xlsx') else prepare_data(data_path)

    config = BGLConfig()
    print(f"\nRunning {N_CHAINS} chains of {config.n_draws} draws "
          f"({config.n_burnin} burn-in, thin {config.thin})...")
    results = run_chains(df, n_chains=N_CHAINS, config=config)
    summary = summarize_results(results)
    plot_trends(results)
    convergence_report(results)

    # Identification profile for c: the evidence that the Laubach-Williams
    # unit loading on trend growth (c = 1) is rejected for India.
    print("\nMarginal log-likelihood profile in c (all else at posterior median):")
    prof = profile_likelihood(df, results, 'c',
                              [0.05, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0])
    print(prof.to_string(index=False, float_format=lambda v: f"{v:10.2f}"))

    print("\nr* under a fixed real-rate-gap wedge (omega=0 recovers the paper):")
    omega_paths = omega_grid(df, config=config)

    print("\nDone. Access results['rstar'], results['states'], etc.")