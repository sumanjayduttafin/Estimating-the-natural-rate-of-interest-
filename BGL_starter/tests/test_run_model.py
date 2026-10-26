"""
Regression tests for the BGL India model.

The three unit/algebra bugs these pin down (the RATE_SCALE factor in
arbitrage_adj, the log10 GDP conversion, and the factor of 2 in the IS-curve
intercept) each individually produced nonsensical r* estimates and were not
visible from the sampler output -- the chain simply returned garbage. They are
tested directly so they cannot silently regress.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_model as rm

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'Data.xlsx')

# Paper posterior medians (Table 3.1), used as a unit-independent reference.
US_SIGMAS = dict(sigma_L=0.68, sigma_S=1.09, sigma_C=1.76,
                 sigma_pi=0.20, sigma_g=0.04, sigma_z=0.32)
LAM_DL = 1.7932823074676663 / 6  # ~0.29888


# ---------------------------------------------------------------- arbitrage_adj

def _A(tau, lam=LAM_DL, S_bar=-1.5, C_bar=-0.3, c=1.0, **kw):
    s = dict(US_SIGMAS, **kw)
    return rm.arbitrage_adj(lam, tau, s['sigma_L'], s['sigma_S'], s['sigma_C'],
                            s['sigma_pi'], s['sigma_g'], s['sigma_z'],
                            S_bar, C_bar, c)


def test_arbitrage_adj_a1_equals_paper_normalization():
    """A(1) = -theta_s(1)*S_bar - theta_c(1)*C_bar, the delta_0 choice that
    makes the long-run level factor equal i* (paper eq. 10 / Appendix B)."""
    _, ts1, tc1 = rm.ns_loadings(LAM_DL, 1)
    S_bar, C_bar = -1.5, -0.3
    assert _A(1, S_bar=S_bar, C_bar=C_bar) == pytest.approx(
        -ts1 * S_bar - tc1 * C_bar, rel=1e-12)


def test_arbitrage_adj_scale():
    """The convexity term must be in per-period decimals, not annualized
    percent. Without the RATE_SCALE division A(40) is about -179pp instead of
    +0.88pp, which is what destroyed the Kalman filter."""
    assert _A(40) == pytest.approx(0.8816, abs=5e-3)
    # Convexity at 10y relative to the short end: tens of bp, not tens of pp.
    convexity = _A(40) - _A(1)
    assert -1.0 < convexity < 0.0


def test_arbitrage_adj_monotone_decreasing():
    """Convexity accumulates with maturity, so A(tau) declines monotonically."""
    vals = [_A(t) for t in [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]]
    assert all(b < a for a, b in zip(vals, vals[1:]))


def test_arbitrage_adj_grows_with_level_variance():
    """A(tau) is driven by Omega_uu[0,0]; more level-factor variance means a
    larger (more negative) convexity adjustment at long maturities."""
    lo = _A(40, sigma_z=0.16)
    hi = _A(40, sigma_z=1.00)
    assert hi < lo


def test_arbitrage_adj_c_enters_as_4c_squared():
    """r* = 4*c*g + z, so the level factor loads on g with weight 4c and
    Omega_uu[0,0] picks up (4c)^2 * sigma_g^2."""
    a_lo = _A(40, c=0.35)
    a_hi = _A(40, c=0.35, sigma_g=0.40)   # 10x sigma_g
    assert a_hi < a_lo


# ------------------------------------------------------------------ data scale

@pytest.fixture(scope='module')
def india():
    if not os.path.exists(DATA):
        pytest.skip('Data.xlsx not present')
    return rm.load_india_data(DATA)


def test_gdp_is_converted_from_log10(india):
    """'log.GDP.SA' is log10(real GDP in Rs crore). The model needs 100*ln so
    the output gap is in percentage points; read as a natural log it implies
    2.7%/yr growth for India instead of 6.3%."""
    gdp = india['gdp_real'].dropna()
    assert 1200 < gdp.min() < gdp.max() < 1700
    ann = (gdp.iloc[-1] - gdp.iloc[0]) / (len(gdp) - 1) * 4
    assert 5.5 < ann < 7.0


def test_sample_has_no_trailing_macro_gap(india):
    """The yield sheet runs past the macro sheet; those quarters are dropped."""
    assert india['gdp_real'].notna().iloc[-1]
    assert len(india) > 90


def test_survey_is_not_pre_debiased(india):
    """The ~5pp household survey bias is estimated in-model as bias_s_pi, so
    the raw survey level must still be in the data."""
    assert india['survey_pi_star'].dropna().mean() > 8.0


# ------------------------------------------------------------- state space

def _params(config, lam):
    np.random.seed(0)
    pr = rm.default_priors()
    p = rm.initialize_params(pr)
    while not rm.check_stationarity(p, lam):
        p = rm.initialize_params(pr)
    p['sig_s_sr'] = 0.0
    return p


def test_is_intercept_implies_zero_mean_output_gap():
    """The real rate gap enters the IS curve at both t-1 and t-2, so its
    constant is picked up twice. With the factor of 2 the model implies a zero
    unconditional mean output gap; without it, about +9pp for India."""
    cfg = rm.BGLConfig()
    lam = 0.095
    p = _params(cfg, lam)
    p['omega'] = 0.0
    F, mu, *_ = rm.build_state_space(p, lam, cfg)

    # Unconditional mean of the coupled cyclical block: (I - F_c)^-1 mu_c over
    # the stationary states [L~, S, C, pi~, x~] and their lags.
    keep = [0, 1, 2, 7, 8, 9, 10, 11, 12, 13]
    Fc, muc = F[np.ix_(keep, keep)], mu[keep]
    mean_states = np.linalg.solve(np.eye(len(keep)) - Fc, muc)
    mean_gap = mean_states[4]   # x_tilde
    assert abs(mean_gap) < 1e-8


def test_omega_shifts_mean_real_rate_gap():
    """omega is the wedge mean(r - r*): it must move the mean output gap off
    zero via the IS intercept, otherwise it is a no-op relabelling."""
    cfg = rm.BGLConfig()
    lam = 0.095
    keep = [0, 1, 2, 7, 8, 9, 10, 11, 12, 13]

    def mean_gap(w):
        p = _params(cfg, lam)
        p['omega'] = w
        F, mu, *_ = rm.build_state_space(p, lam, cfg)
        Fc, muc = F[np.ix_(keep, keep)], mu[keep]
        return np.linalg.solve(np.eye(len(keep)) - Fc, muc)[4]

    assert abs(mean_gap(0.0)) < 1e-8
    assert abs(mean_gap(-1.0)) > 1e-3


def test_omega_shifts_all_yield_intercepts_equally():
    """omega enters L* = pi* + 4cg + z + omega, a parallel shift of the curve."""
    cfg = rm.BGLConfig()
    lam = 0.095
    p0 = _params(cfg, lam); p0['omega'] = 0.0
    p1 = rm.copy.deepcopy(p0); p1['omega'] = -1.0
    K = len(cfg.maturities)
    g0 = rm.build_state_space(p0, lam, cfg)[5][:K]
    g1 = rm.build_state_space(p1, lam, cfg)[5][:K]
    assert np.allclose(g1 - g0, -1.0)


def test_survey_row_targets_four_quarter_ahead_expectation():
    """The IESH survey is a 1-year-ahead forecast, so it loads on
    E_t pi_{t+4} = F^4 rows for pi* and pi~, not on pi* alone."""
    cfg = rm.BGLConfig()
    lam = 0.095
    p = _params(cfg, lam)
    p['sig_s_pi'] = 0.5
    F, mu, Q, G, C, gam, R, n_obs = rm.build_state_space(p, lam, cfg)
    K = len(cfg.maturities)
    row = C[K + 2]
    F4 = np.linalg.matrix_power(F, 4)
    assert np.allclose(row, F4[3, :] + F4[7, :])
    # pi* is a random walk, so the pi* part of the loading is still exactly 1.
    assert row[3] == pytest.approx(1.0)
    # ...but the row is not the naive "pi* only" mapping.
    assert not np.allclose(row, np.eye(cfg.n_states)[3])


def test_survey_bias_enters_intercept():
    cfg = rm.BGLConfig()
    lam = 0.095
    p = _params(cfg, lam); p['sig_s_pi'] = 0.5
    K = len(cfg.maturities)
    p['bias_s_pi'] = 0.0
    g0 = rm.build_state_space(p, lam, cfg)[5][K + 2]
    p['bias_s_pi'] = 5.0
    g1 = rm.build_state_space(p, lam, cfg)[5][K + 2]
    assert g1 - g0 == pytest.approx(5.0)


def test_check_stationarity_bounds_c_to_unit_interval():
    cfg = rm.BGLConfig()
    lam = 0.095
    p = _params(cfg, lam)
    for bad in (-0.1, 0.0, 1.0, 1.5):
        q = rm.copy.deepcopy(p); q['c'] = bad
        assert not rm.check_stationarity(q, lam)


# ------------------------------------------------------------------- filter

def test_kalman_loglik_is_sane(india):
    """With the unit bugs present this was around -1.7e7."""
    cfg = rm.BGLConfig()
    lam = 0.095
    p = _params(cfg, lam)
    p['omega'] = -1.0
    F, mu, Q, G, C, gam, R, _ = rm.build_state_space(p, lam, cfg)
    Y_mat = rm._observation_matrix(india, cfg)
    xi0, P0 = rm.initial_state(india, p, lam, cfg)
    ll = rm.kalman_filter(Y_mat, F, mu, Q, C, gam, R, xi0, P0)[4]
    assert -5000 < ll < 0


def test_filter_covariance_stays_symmetric_psd(india):
    """Joseph form: R is near-singular on the exactly-matched rows."""
    cfg = rm.BGLConfig()
    lam = 0.095
    p = _params(cfg, lam)
    F, mu, Q, G, C, gam, R, _ = rm.build_state_space(p, lam, cfg)
    Y_mat = rm._observation_matrix(india, cfg)
    xi0, P0 = rm.initial_state(india, p, lam, cfg)
    _, P_filt, _, _, _ = rm.kalman_filter(Y_mat, F, mu, Q, C, gam, R, xi0, P0)
    for P in P_filt[::10]:
        assert np.allclose(P, P.T, atol=1e-8)
        assert np.min(np.linalg.eigvalsh((P + P.T) / 2)) > -1e-8


def test_initial_state_is_centred_on_the_data(india):
    cfg = rm.BGLConfig()
    p = _params(cfg, 0.095)
    xi0, P0 = rm.initial_state(india, p, 0.095, cfg)
    # x* within a few pp of the first GDP observation
    assert abs(xi0[4] - india['gdp_real'].dropna().iloc[0]) < 5.0
    # trend growth ~1.5/quarter for India, not the 0.5 that suits the US
    assert 1.0 < xi0[5] < 2.0
    assert np.allclose(P0, P0.T)


# --------------------------------------------------------------- end-to-end

@pytest.fixture(scope='module')
def short_run(india):
    np.random.seed(0)
    cfg = rm.BGLConfig(n_draws=2500, n_burnin=1800, thin=5)
    return rm.gibbs_sampler(india, config=cfg, verbose=False)


def test_smoke_run_produces_sensible_macro_states(short_run):
    st, ps, rs = short_run['states'], short_run['pi_star'], short_run['rstar']
    assert 3.0 < np.median(ps[:, -1]) < 6.5           # trend inflation
    assert 4.0 < np.median(4 * st[:, -1, 5]) < 7.5    # annualized trend growth
    assert abs(np.median(st[:, -1, 8])) < 5.0         # output gap
    assert -3.0 < np.median(rs[:, -1]) < 5.0          # r*


def test_smoke_run_fits_the_yield_curve(short_run, india):
    """Posterior-mean yield residuals should be comparable to the paper's
    (7-24bp). Before the fixes these were tens of percentage points."""
    cfg = short_run['config']
    st, P, lams = short_run['states'], short_run['params'], short_run['lambda']
    resid = np.zeros((len(P), len(india), len(cfg.maturities)))
    for d, p in enumerate(P):
        lam = lams[d]
        Sb, Cb = rm.sc_unconditional_mean(p)
        for k, tau in enumerate(cfg.maturities):
            _, ts, tc = rm.ns_loadings(lam, tau)
            adj = p['omega'] + rm.arbitrage_adj(
                lam, tau, p['sig_Lc'], p['sig_S'], p['sig_C'],
                p['sig_pi_star'], p['sig_g'], p['sig_z'], Sb, Cb, p['c'])
            fit = (adj + st[d, :, 0] + ts*st[d, :, 1] + tc*st[d, :, 2]
                   + st[d, :, 3] + 4*p['c']*st[d, :, 5] + st[d, :, 6])
            resid[d, :, k] = india[f'y_{tau}'].values - fit
    mean_resid = resid.mean(0)
    assert np.abs(np.nanmean(mean_resid, 0)).max() < 0.15   # bias < 15bp
    assert np.nanstd(mean_resid, 0).max() < 0.35            # sd  < 35bp


def test_mh_acceptance_rates_are_tuned(short_run):
    """Adaptive burn-in tuning should land every MH block near 30%."""
    for k, rate in short_run['accept_rates'].items():
        assert 0.10 < rate < 0.60, f"{k} acceptance {rate:.2f} outside 0.10-0.60"


def test_lambda_posterior_matches_the_indian_curve(short_run):
    """DNS-OLS on the Indian G-sec curve prefers lambda around 0.095."""
    assert 0.05 < np.median(short_run['lambda']) < 0.20


def test_estimated_survey_bias_matches_the_raw_forecast_error(short_run):
    """The in-model bias should recover roughly the 5pp raw IESH bias."""
    b = np.median([p['bias_s_pi'] for p in short_run['params']])
    assert 3.0 < b < 7.0


def test_c_profile_peaks_below_the_laubach_williams_value(india, short_run):
    """The India result on c, stated at the strength the data actually support.

    The profile is shallow: it peaks around c = 0.4-0.5, so the data mildly
    favour a growth loading well below the Laubach-Williams unit value, and
    they clearly reject large c (c = 1.5 costs 3-4 log points). But c = 1
    itself costs only about 1 log point, which is NOT a decisive rejection.
    The identifying work at the low end is done by the prior, not the data.
    """
    prof = rm.profile_likelihood(india, short_run, 'c',
                                 [0.05, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5])
    prof = prof.set_index('c')['delta']
    assert 0.2 <= prof.idxmax() <= 0.8       # interior mode below 1
    assert prof.loc[1.5] < -2.0              # large c clearly rejected
    assert prof.loc[1.0] < -0.3              # c=1 worse, but only mildly
    assert prof.loc[0.05] > -3.0             # flat/uninformative at the low end


def test_omega_zero_pins_rstar_to_the_mean_real_rate(india):
    """The sharpest check that the macro and finance blocks are wired
    together: with omega = 0 the model's real rate gap is mean-zero by
    construction, so mean(r*) must equal the sample-mean ex-ante real rate."""
    np.random.seed(0)
    cfg = rm.BGLConfig(n_draws=2500, n_burnin=1800, thin=5)
    cfg.estimate_omega = False
    pri = rm.default_priors()
    pri['omega'] = (0.0, 1e-6)
    res = rm.gibbs_sampler(india, config=cfg, priors=pri, verbose=False)

    mean_rstar = np.median(res['rstar'], axis=0).mean()
    ex_ante_real = (india['y_1'] - india['cpi_yoy'].shift(-1)).mean()
    assert abs(mean_rstar - ex_ante_real) < 1.0


def test_geweke_runs_and_reports(short_run):
    z = rm.convergence_report(short_run)
    assert 'r*_end' in z and np.isfinite(z['r*_end'])


def test_split_rhat_and_ess():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal((4, 1000))
    assert rm.split_rhat(iid) < 1.02
    # a drifting chain is flagged by split-R-hat
    drift = iid + np.linspace(0, 5, 1000)
    assert rm.split_rhat(drift) > 1.1
    # iid ESS ~ n; AR(1) rho=.9 ESS ~ n(1-rho)/(1+rho) ~ n/19
    assert 600 < rm.effective_sample_size(iid[0]) < 1500
    ar = np.zeros(20000)
    e = rng.standard_normal(20000)
    for t in range(1, 20000):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    assert 600 < rm.effective_sample_size(ar) < 1800


def test_rstar_uses_c_of_state_draw(short_run):
    # stored r* must be consistent with the c recorded for the state draw
    res = short_run
    k = 5
    p = res['params'][k]
    g, z = res['states'][k][:, 5], res['states'][k][:, 6]
    assert np.allclose(res['rstar'][k], 4 * p['c_state'] * g + z)
