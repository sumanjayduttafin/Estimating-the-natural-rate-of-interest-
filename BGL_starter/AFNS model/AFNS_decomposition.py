"""
AFNS (Arbitrage-Free Nelson-Siegel) Model — Yield Decomposition
================================================================
Indian Yield Curve Data

Reference: Christensen, Diebold & Rudebusch (2011), Journal of Econometrics.

Decomposition:
  Fitted yield  = Expectations component + Term premium
  Expectations  = average expected short rate under physical (P) measure
  Term premium  = Fitted yield − Expectations

Short rate in AFNS: r_t = L_t + S_t
  (both Level and Slope load on the instantaneous rate since B_L(0)=B_S(0)=1)

Outputs
-------
  afns_decomposition.csv    — full decomposition time series
  afns_decomposition.png    — 4-panel plot (3M · 2Y · 5Y · 10Y)
  afns_model_fit.png        — observed vs fitted yields
  afns_factors.png          — smoothed Level, Slope, Curvature factors
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.linalg
import scipy.optimize
import scipy.integrate
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# 0.  DATA LOADING
# ---------------------------------------------------------------------------

# Paths are relative to this file, so the script runs from any working
# directory: the data sit in BGL_starter/Data, outputs next to this script.
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
DATA_PATH = os.path.join(OUT_DIR, os.pardir, "Data", "Indian Yield Curve Data.xlsx")

# Target maturities for the 4-panel decomposition plot
# Values are maturity in YEARS; labels are shown on the plot panels.
TARGET_MATURITIES_YEARS = np.array([0.5, 2.0, 5.0, 10.0])
TARGET_LABELS = {0.5: "3-Month", 2.0: "2-Year", 5.0: "5-Year", 10.0: "10-Year"}


def load_data(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load and preprocess the yield curve data.

    Returns
    -------
    yields  : (T × N) DataFrame, yields in decimal, columns = maturities in YEARS
    grid    : (N,) array of maturities in years
    """
    df = pd.read_excel(path, sheet_name="Long_data")
    df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)

    # Percent → decimal
    df = df / 100.0

    # Resample to month-end (one observation per month)
    df = df.resample("ME").last()
    #df = df.loc["2010-08":"2026-04"]

    # Column labels: "3m" → maturity in years (float)
    months = df.columns.str.replace("m", "").astype(int)
    df.columns = months / 12.0          # e.g. 0.25, 0.5, 1.0, ..., 20.0
    grid = df.columns.values            # (N,) maturities in years

    print(f"Loaded {len(df)} monthly observations "
          f"({df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')})")
    print(f"Maturities (years): {list(np.round(grid, 4))}")
    return df, grid


# ---------------------------------------------------------------------------
# 1.  NELSON-SIEGEL LOADINGS & YIELD-ADJUSTMENT
# ---------------------------------------------------------------------------

def ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """
    Nelson-Siegel loading matrix (N × 3).
    Columns: [Level=1, Slope=(1-e^{-λτ})/(λτ), Curvature=(…-e^{-λτ})]
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    L = np.ones((len(tau), 3))
    for i, t in enumerate(tau):
        if t < 1e-8:
            L[i, 1] = 1.0   # lim_{τ→0} = 1
            L[i, 2] = 0.0   # lim_{τ→0} = 0
        else:
            e = np.exp(-lam * t)
            L[i, 1] = (1.0 - e) / (lam * t)
            L[i, 2] = (1.0 - e) / (lam * t) - e
    return L


def yield_adj(tau: np.ndarray, lam: float, sigma: np.ndarray) -> np.ndarray:
    """
    Yield-adjustment (convexity) term  −A(τ)/τ  for the independent AFNS.

    Under no-arbitrage, the bond-pricing ODE gives:
        A(τ) = ½ ∫₀^τ B_raw(s)′ Σ² B_raw(s) ds   (> 0)

    Yield adjustment: −A(τ)/τ < 0  (convexity lowers yields).

    Raw bond-pricing loadings (sign convention from the ODE):
        B_L_raw = −s,  B_S_raw = −(1−e^{−λs})/λ,
        B_C_raw = −(1−e^{−λs})/λ + s·e^{−λs}
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    s1, s2, s3 = sigma
    adj = np.zeros(len(tau))

    for i, t in enumerate(tau):
        if t < 1e-8:
            continue

        def integrand(s: float) -> float:
            if s < 1e-8:
                return s2 ** 2 / lam ** 2   # only BS_raw survives at s=0
            e = np.exp(-lam * s)
            bl2 = s ** 2
            bs2 = ((1.0 - e) / lam) ** 2
            bc2 = ((1.0 - e) / lam - s * e) ** 2
            return s1 ** 2 * bl2 + s2 ** 2 * bs2 + s3 ** 2 * bc2

        integral, _ = scipy.integrate.quad(integrand, 0.0, t, limit=200)
        adj[i] = -integral / (2.0 * t)     # negative

    return adj


# ---------------------------------------------------------------------------
# 2.  STATE-SPACE MATRICES
# ---------------------------------------------------------------------------

def build_matrices(params: np.ndarray, grid: np.ndarray, dt: float = 1.0 / 12):
    """
    Construct Kalman-filter system matrices.

    Parameter vector (11 elements, all log-transformed where > 0):
      params[0]    log λ         — NS decay parameter (per year)
      params[1:4]  log κ         — P-measure mean-reversion speeds [κ_L, κ_S, κ_C]
      params[4:7]  θ             — P-measure long-run means [θ_L, θ_S, θ_C]
      params[7:10] log σ         — state volatilities [σ_L, σ_S, σ_C]
      params[10]   log σ_meas    — scalar measurement-error std

    State equation  (monthly, dt = 1/12 yr):
      X_t = c + Φ X_{t-1} + η_t,   η_t ~ N(0, Q)
        Φ = diag(exp(−κ·dt))        diagonal OU discretisation
        c = (I − Φ) θ
        Q = diag(σ_i² (1−φ_i²)/(2κ_i))   exact OU conditional variance

    Observation equation:
      y_t = Λ X_t + d + ε_t,   ε_t ~ N(0, H)
        Λ — (N×3) NS loadings
        d — (N,)  yield-adjustment offsets  (negative)
        H — σ_meas² I_N
    """
    lam   = np.exp(params[0])
    kappa = np.exp(params[1:4])
    theta = params[4:7]
    sigma = np.exp(params[7:10])
    smeas = np.exp(params[10])

    Lambda = ns_loadings(grid, lam)         # (N, 3)
    d      = yield_adj(grid, lam, sigma)    # (N,)

    phi = np.exp(-kappa * dt)               # (3,)
    Phi = np.diag(phi)
    c   = (1.0 - phi) * theta              # intercept = (I−Φ)θ

    var = sigma ** 2 * (1.0 - phi ** 2) / (2.0 * kappa)   # exact OU variance
    Q   = np.diag(var)
    H   = smeas ** 2 * np.eye(len(grid))

    return Lambda, d, Phi, c, Q, H


# ---------------------------------------------------------------------------
# 3.  KALMAN FILTER  +  RTS SMOOTHER
# ---------------------------------------------------------------------------

def kalman_filter(y_obs: np.ndarray, Lambda, d, Phi, c, Q, H):
    """
    Standard Kalman filter (no Numba).

    Initialised at the unconditional (stationary) distribution of the
    O-U process:  X_0 = θ^P,  P_0 = solve discrete Lyapunov equation.

    Returns log-likelihood and full filtered/predicted sequences.
    """
    T, _ = y_obs.shape
    n = 3

    # Stationary initial state
    X0 = np.linalg.solve(np.eye(n) - Phi, c)     # = θ^P
    try:
        P0 = scipy.linalg.solve_discrete_lyapunov(Phi, Q)
    except Exception:
        P0 = np.eye(n) * 0.1

    Xf = np.zeros((T, n));  Pf = np.zeros((T, n, n))
    Xp = np.zeros((T, n));  Pp = np.zeros((T, n, n))

    Xpred = Phi @ X0 + c
    Ppred = Phi @ P0 @ Phi.T + Q
    ll = 0.0

    for t in range(T):
        Xp[t] = Xpred;  Pp[t] = Ppred

        obs   = y_obs[t]
        valid = ~np.isnan(obs)

        if valid.any():
            Lv = Lambda[valid];  dv = d[valid]
            Hv = H[np.ix_(valid, valid)]
            yv = obs[valid]

            nu = yv - (Lv @ Xpred + dv)                 # innovation
            S  = Lv @ Ppred @ Lv.T + Hv                 # innovation cov

            try:
                logdet = np.linalg.slogdet(S)[1]
                K  = Ppred @ Lv.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                return -1e10, Xf, Pf, Xp, Pp

            ll += -0.5 * (valid.sum() * np.log(2 * np.pi) + logdet
                          + nu @ np.linalg.solve(S, nu))

            Xf[t] = Xpred + K @ nu
            Pf[t] = (np.eye(n) - K @ Lv) @ Ppred
        else:
            Xf[t] = Xpred;  Pf[t] = Ppred

        Xpred = Phi @ Xf[t] + c
        Ppred = Phi @ Pf[t] @ Phi.T + Q

    return ll, Xf, Pf, Xp, Pp


def rts_smoother(Xf, Pf, Xp, Pp, Phi):
    """Rauch–Tung–Striebel backward smoother."""
    T = Xf.shape[0]
    Xs = Xf.copy();  Ps = Pf.copy()

    for t in range(T - 2, -1, -1):
        G  = Pf[t] @ Phi.T @ np.linalg.inv(Pp[t + 1])
        Xs[t] = Xf[t] + G @ (Xs[t + 1] - Xp[t + 1])
        Ps[t] = Pf[t] + G @ (Ps[t + 1] - Pp[t + 1]) @ G.T

    return Xs, Ps


def neg_loglik(params, y_obs, grid):
    """Negative log-likelihood for scipy.optimize.minimize."""
    try:
        Lambda, d, Phi, c, Q, H = build_matrices(params, grid)
        ll, *_ = kalman_filter(y_obs, Lambda, d, Phi, c, Q, H)
        return -ll if np.isfinite(ll) else 1e10
    except Exception:
        return 1e10


# ---------------------------------------------------------------------------
# 4.  MAXIMUM LIKELIHOOD ESTIMATION
# ---------------------------------------------------------------------------

def estimate_model(y_obs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Estimate AFNS parameters by MLE using L-BFGS-B.

    Returns the optimal parameter vector (11 elements).
    """
    # Initial guess
    p0 = np.array([
        np.log(0.3),    # log λ  → λ ≈ 0.5/yr, max hump at ~2 yr
        np.log(0.3),    # log κ_L  (slow mean-reversion, Level)
        np.log(1.0),    # log κ_S
        np.log(2.0),    # log κ_C  (fast, Curvature)
        0.07,           # θ_L  ~7% (Indian long-run rate)
       -0.01,           # θ_S
        0.01,           # θ_C
        np.log(0.008),  # log σ_L
        np.log(0.010),  # log σ_S
        np.log(0.012),  # log σ_C
        np.log(0.001),  # log σ_meas ≈ 10 bps
    ])

    ll0 = -neg_loglik(p0, y_obs, grid)
    print(f"Initial log-likelihood: {ll0:.2f}")
    print("Optimising parameters (L-BFGS-B) …")

    res = scipy.optimize.minimize(
        neg_loglik, p0, args=(y_obs, grid),
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7},
    )

    phat = res.x
    print(f"Optimisation {'CONVERGED' if res.success else 'did NOT converge'}")
    print(f"Final  log-likelihood: {-res.fun:.2f}  |  {res.message}")
    _print_params(phat)
    return phat


def _print_params(params: np.ndarray):
    lam   = np.exp(params[0])
    kappa = np.exp(params[1:4])
    theta = params[4:7]
    sigma = np.exp(params[7:10])
    smeas = np.exp(params[10])
    print(f"\n{'Parameter':<24} {'Value':>10}")
    print("-" * 36)
    print(f"{'λ':<24} {lam:>10.4f}  (max hump at {1/lam:.2f} yr)")
    print(f"{'κ_L  (yr⁻¹)':<24} {kappa[0]:>10.4f}")
    print(f"{'κ_S  (yr⁻¹)':<24} {kappa[1]:>10.4f}")
    print(f"{'κ_C  (yr⁻¹)':<24} {kappa[2]:>10.4f}")
    print(f"{'θ_L  (%)':<24} {theta[0]*100:>10.3f}")
    print(f"{'θ_S  (%)':<24} {theta[1]*100:>10.3f}")
    print(f"{'θ_C  (%)':<24} {theta[2]*100:>10.3f}")
    print(f"{'σ_L':<24} {sigma[0]:>10.5f}")
    print(f"{'σ_S':<24} {sigma[1]:>10.5f}")
    print(f"{'σ_C':<24} {sigma[2]:>10.5f}")
    print(f"{'σ_meas  (bps)':<24} {smeas*10000:>10.2f}")


# ---------------------------------------------------------------------------
# 5.  YIELD DECOMPOSITION
# ---------------------------------------------------------------------------

def decompose_yields(
    Xs: np.ndarray,
    params: np.ndarray,
    grid: np.ndarray,
    target_maturities: np.ndarray,
    dates: pd.DatetimeIndex,
    yields_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decompose AFNS yields into expectations component and term premium.

    Fitted yield at maturity τ:
        ŷ_t(τ) = Λ(τ)ᵀ X_t + adj(τ)

    Short rate in AFNS (instantaneous): r_t = L_t + S_t
    (B_L(0) = B_S(0) = 1,  B_C(0) = 0)

    Expectations component (average expected short rate under ℙ):
        EH_t(n) = (θ_L + θ_S)
                  + φ̄_{L,n}·(L_t − θ_L)
                  + φ̄_{S,n}·(S_t − θ_S)

    where  φ̄_{i,n} = φ_i (1 − φ_i^n) / [n (1 − φ_i)]
           n = maturity in months, φ_i = monthly AR coefficient

    Term premium: TP_t(τ) = ŷ_t(τ) − EH_t(n)

    Parameters
    ----------
    Xs               : (T, 3) smoothed states [Level, Slope, Curvature]
    params           : (11,) estimated parameter vector
    grid             : (N,) maturities in years (must match yields_df columns)
    target_maturities: (M,) maturities in years to decompose
    dates            : DatetimeIndex, length T
    yields_df        : (T, N) DataFrame of observed yields (columns = grid)

    Returns
    -------
    decomp_df : (T, M×4) DataFrame with columns
                {τ}Y_actual, {τ}Y_fitted, {τ}Y_expectations, {τ}Y_term_premium
    """
    lam   = np.exp(params[0])
    kappa = np.exp(params[1:4])          # per-year
    theta = params[4:7]
    sigma = np.exp(params[7:10])

    phi_monthly = np.exp(-kappa / 12.0)  # monthly AR coefficients  (3,)
    Lambda_all  = ns_loadings(grid, lam)  # (N, 3)  — for fitted yields
    d_all       = yield_adj(grid, lam, sigma)  # (N,) yield adjustments

    # Short-rate loading: r = L + S  →  e_r = [1, 1, 0]
    e_r      = np.array([1.0, 1.0, 0.0])
    er_theta = float(e_r @ theta)         # θ_L + θ_S

    rows = {}

    for tau_y in target_maturities:
        n_months = int(round(tau_y * 12))   # maturity in months

        # ── verify this maturity exists in data grid ──────────────────
        if tau_y not in grid:
            # find nearest
            idx_near = int(np.argmin(np.abs(grid - tau_y)))
            tau_y_used = grid[idx_near]
            print(f"  [warn] {tau_y} yr not in grid → using {tau_y_used:.3f} yr")
        else:
            tau_y_used = tau_y

        col_idx = int(np.where(grid == tau_y_used)[0][0])

        lam_row = ns_loadings(np.array([tau_y_used]), lam)[0]    # (3,)
        d_val   = yield_adj(np.array([tau_y_used]), lam, sigma)[0]

        # ── fitted yield ──────────────────────────────────────────────
        fitted = Xs @ lam_row + d_val          # (T,)

        # ── EH component: average expected short rate over n months ──
        # φ̄_{i,n} = φ_i (1 − φ_i^n) / [n (1 − φ_i)]
        phi_n   = phi_monthly ** n_months      # (3,)
        denom   = np.where(np.abs(phi_monthly - 1.0) < 1e-10,
                           1e-10, 1.0 - phi_monthly)
        phi_bar = np.where(
            np.abs(phi_monthly - 1.0) < 1e-10,
            1.0,
            phi_monthly * (1.0 - phi_n) / (n_months * denom),
        )   # (3,)

        # EH_t = er_theta  +  (e_r * phi_bar) · (X_t − theta)
        # Curvature does not enter because e_r[2] = 0
        X_dm = Xs - theta[np.newaxis, :]       # (T, 3)
        eh   = er_theta + X_dm @ (e_r * phi_bar)  # (T,)

        tp = fitted - eh                       # (T,)

        # ── actual observed yield ─────────────────────────────────────
        actual = yields_df[tau_y_used].values  # (T,)

        label = TARGET_LABELS.get(tau_y, f"{tau_y}Y")
        rows[f"{label}_actual"]       = actual * 100
        rows[f"{label}_fitted"]       = fitted * 100
        rows[f"{label}_expectations"] = eh     * 100
        rows[f"{label}_term_premium"] = tp     * 100

    decomp_df = pd.DataFrame(rows, index=dates)
    return decomp_df


# ---------------------------------------------------------------------------
# 6.  PLOTS
# ---------------------------------------------------------------------------

def plot_factors(factor_df: pd.DataFrame, out_path: str):
    """Plot smoothed Level, Slope, Curvature factors."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for ax, col, c in zip(axes, ["Level", "Slope", "Curvature"], colors):
        ax.plot(factor_df.index, factor_df[col] * 100, color=c, linewidth=1.3, label=col)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_ylabel("Percent (%)")
        ax.set_title(col)
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    fig.suptitle("Smoothed AFNS Latent Factors", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def plot_model_fit(decomp_df: pd.DataFrame, target_labels: list[str], out_path: str):
    """Observed vs fitted yields for the 4 target maturities."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, label in zip(axes.flatten(), target_labels):
        actual = decomp_df[f"{label}_actual"]
        fitted = decomp_df[f"{label}_fitted"]
        resid  = actual - fitted   # bps
        rmse   = np.sqrt(np.mean(resid ** 2))

        ax.plot(decomp_df.index, actual.values, color="#888888", linewidth=0.9,
                linestyle="--", label="Observed")
        ax.plot(decomp_df.index, fitted.values, color="#1f77b4", linewidth=1.5,
                label="Fitted")
        ax.set_title(f"{label}  (RMSE = {rmse:.1f} bps)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Percent (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("AFNS Model Fit: Observed vs Fitted Yields",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def analyze_decomposition(
    decomp_df: pd.DataFrame,
    target_labels: list[str],
    out_path: str,
):
    """
    4-panel figure (2 × 2).  Each panel shows one target maturity with:
      - Observed yield     (dashed grey)
      - Fitted yield       (blue solid)
      - Expectations component  (green solid)
      - Term premium       (red solid)

    All four series are on the SAME axes for direct comparison.
    """
    if len(target_labels) != 4:
        raise ValueError("Exactly 4 target maturities are required.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    colors = {
        "actual":       ("#888888", "--", 0.7, 1.0),
        "fitted":       ("#1f77b4", "-",  1.0, 1.6),
        "expectations": ("#2ca02c", "-",  1.0, 1.6),
        "term_premium": ("#d62728", "-",  1.0, 1.6),
    }
    display_names = {
        "actual":       "Observed yield",
        "fitted":       "Fitted yield",
        "expectations": "Expectations component",
        "term_premium": "Term premium",
    }

    for ax, label in zip(axes.flatten(), target_labels):
        dates = decomp_df.index

        for key in ("actual", "fitted", "expectations", "term_premium"):
            col_name = f"{label}_{key}"
            if col_name not in decomp_df.columns:
                print(f"  [warn] column '{col_name}' not found — skipping.")
                continue
            c, ls, alpha, lw = colors[key]
            ax.plot(dates, decomp_df[col_name].values,
                    color=c, linestyle=ls, linewidth=lw, alpha=alpha,
                    label=display_names[key])

        ax.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Percent (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    fig.suptitle(
        "AFNS Yield Decomposition\n"
        "Fitted Yield = Expectations Component + Term Premium",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# 7.  MAIN
# ---------------------------------------------------------------------------

def main():
    # ── Load data ────────────────────────────────────────────────────
    yields_df, grid = load_data(DATA_PATH)
    y_obs = yields_df.values   # (T, N)

    # Verify each target maturity is in the grid
    for tau in TARGET_MATURITIES_YEARS:
        if tau not in grid:
            nearest = grid[np.argmin(np.abs(grid - tau))]
            print(f"[warn] {tau} yr not in grid (nearest: {nearest:.3f} yr)")

    # ── Estimate model ───────────────────────────────────────────────
    params_hat = estimate_model(y_obs, grid)

    # ── Extract smoothed factors ─────────────────────────────────────
    Lambda_hat, d_hat, Phi_hat, c_hat, Q_hat, H_hat = build_matrices(params_hat, grid)
    _, Xf, Pf, Xp, Pp = kalman_filter(y_obs, Lambda_hat, d_hat, Phi_hat, c_hat, Q_hat, H_hat)
    Xs, _ = rts_smoother(Xf, Pf, Xp, Pp, Phi_hat)

    factor_df = pd.DataFrame(Xs, index=yields_df.index,
                             columns=["Level", "Slope", "Curvature"])
    plot_factors(factor_df, OUT_DIR + "afns_factors.png")

    # ── Decompose yields ─────────────────────────────────────────────
    decomp_df = decompose_yields(
        Xs, params_hat, grid,
        TARGET_MATURITIES_YEARS,
        yields_df.index,
        yields_df,
    )

    target_labels = [TARGET_LABELS[tau] for tau in TARGET_MATURITIES_YEARS]

    # ── Sanity-check print ───────────────────────────────────────────
    print("\nDecomposition at last observation:")
    last = decomp_df.iloc[-1]
    fmt  = " {:<26} {:>8.2f}%"
    for label in target_labels:
        print(f"\n  {label}:")
        for key in ("actual", "fitted", "expectations", "term_premium"):
            col = f"{label}_{key}"
            print(fmt.format(key, last[col]))

    # ── Plots ────────────────────────────────────────────────────────
    analyze_decomposition(decomp_df, target_labels, OUT_DIR + "afns_decomposition.png")
    plot_model_fit(decomp_df, target_labels, OUT_DIR + "afns_model_fit.png")

    # ── CSV output ───────────────────────────────────────────────────
    excel_path = OUT_DIR + "afns_decomposition.xlsx"
    decomp_df.to_excel(excel_path)
    print(f"Saved → {excel_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
