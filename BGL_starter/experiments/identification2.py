"""
Identification experiments, round 2.

Round 1 (experiments/identification.py) established two things:
  * Pinning z (sigma_z -> 0) does NOT sharpen c: its posterior sd went from
    0.127 to 0.142. The yield-curve level has several near-unit-root
    absorbers -- pi* is a free random walk and L_tilde has aL ~ 0.85, which
    over 100 quarters looks trend-like -- so removing one just shifts the
    absorption to another.
  * Round 1's short-rate "survey" proxy was INVALID. It used the realized
    91-day rate 4 quarters ahead as a stand-in for E_t y_{t+4}(1), which
    injects the forecast error as if it were signal. India's policy-rate
    cycles make that error large and serially correlated, so the r* band
    nearly doubled. That is an artefact of the proxy, not evidence about
    surveys.

This round fixes the proxy and tests the lever that should matter more.

  C'. Honest ex-ante forecast proxy. A recursive AR(4) on the 91-day rate,
      re-estimated each quarter on data available at t only, forecast 4
      quarters ahead. That is a genuine conditional expectation formed from
      past information, so its errors are realistic rather than clairvoyant.

  E.  Trend-inflation anchor. From 2016Q3 (adoption of flexible inflation
      targeting) pi* is tied to the RBI's 4% midpoint with a 0.5pp
      measurement sd. This speaks to pi* directly, whereas the IESH survey is
      a 1-year-ahead forecast. Since r* = i* - pi*, pinning pi* at the end of
      the sample is the most direct route to pinning the level of r*.

  F.  Both.

Run:  python experiments/identification2.py
"""

import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_model as rm

N_DRAWS, N_BURNIN, THIN = 20000, 10000, 10
AR_LAGS, HORIZON = 4, 4
FIT_START = '2016-07-01'    # RBI flexible inflation targeting
TARGET = 4.0                # CPI target midpoint


def recursive_ar_forecast(y, lags=AR_LAGS, h=HORIZON, min_obs=24):
    """
    Real-time AR(lags) forecast of y, h steps ahead, re-estimated each period
    on data through t only. Returns a series aligned so that entry t is the
    forecast of y_{t+h} formed at t -- i.e. the object a survey would report.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    for t in range(min_obs, n):
        hist = y[:t+1]
        hist = hist[~np.isnan(hist)]
        if len(hist) < lags + 10:
            continue
        X = np.column_stack([np.ones(len(hist)-lags)] +
                            [hist[lags-i-1:len(hist)-i-1] for i in range(lags)])
        b = np.linalg.lstsq(X, hist[lags:], rcond=None)[0]
        state = list(hist[-lags:][::-1])          # most recent first
        for _ in range(h):
            nxt = b[0] + float(np.dot(b[1:], state))
            state = [nxt] + state[:-1]
        out[t] = state[0]
    return out


def build_specs(df):
    base_pri = rm.default_priors()

    df_sr = df.copy()
    df_sr['survey_i_4q'] = recursive_ar_forecast(df['y_1'].values)

    df_anchor = df.copy()
    df_anchor['pi_star_anchor'] = np.where(
        df.index >= pd.Timestamp(FIT_START), TARGET, np.nan)

    df_both = df_sr.copy()
    df_both['pi_star_anchor'] = df_anchor['pi_star_anchor']

    return {
        "A.  baseline":                    (df,        base_pri),
        "C'. + real-time AR T-bill fcst":  (df_sr,     base_pri),
        "E.  + pi* anchored to 4% target": (df_anchor, base_pri),
        "F.  + both":                      (df_both,   base_pri),
    }


def summarize(name, res):
    p = res['params']
    c = np.array([d['c'] for d in p])
    w = np.array([d['omega'] for d in p])
    rs = res['rstar']
    ps = res['pi_star']
    band = np.percentile(rs, 95, axis=0) - np.percentile(rs, 5, axis=0)
    pband = np.percentile(ps, 95, axis=0) - np.percentile(ps, 5, axis=0)
    return {
        'spec': name,
        'c_med': np.median(c), 'c_sd': c.std(),
        'omega_med': np.median(w), 'omega_sd': w.std(),
        'rstar_end': np.median(rs[:, -1]),
        'rstar_mean': np.median(rs, axis=0).mean(),
        'rstar_band_avg': band.mean(), 'rstar_band_end': band[-1],
        'pistar_end': np.median(ps[:, -1]), 'pistar_band_end': pband[-1],
    }


def main():
    df = rm.load_india_data('Data.xlsx')
    proxy = recursive_ar_forecast(df['y_1'].values)
    err = proxy[:-HORIZON] - df['y_1'].values[HORIZON:]
    ok = ~np.isnan(err)
    print(f"\nreal-time AR({AR_LAGS}) {HORIZON}q-ahead forecast of the 91-day rate:")
    print(f"  n={ok.sum()}, RMSE={np.sqrt(np.nanmean(err[ok]**2)):.2f}pp, "
          f"bias={np.nanmean(err[ok]):+.2f}pp")

    rows = []
    for name, (data, pri) in build_specs(df).items():
        print(f"\n=== {name} ===", flush=True)
        cfg = rm.BGLConfig(n_draws=N_DRAWS, n_burnin=N_BURNIN, thin=THIN)
        np.random.seed(0)
        res = rm.gibbs_sampler(data, config=cfg, priors=pri, verbose=False)
        r = summarize(name, res)
        rows.append(r)
        print(f"  c={r['c_med']:.3f} (sd {r['c_sd']:.3f}) | "
              f"omega={r['omega_med']:+.2f} | r* end={r['rstar_end']:+.2f}% "
              f"band={r['rstar_band_avg']:.2f}pp | pi* end={r['pistar_end']:.2f}%")

    out = pd.DataFrame(rows).set_index('spec')
    print("\n" + "=" * 108)
    print("IDENTIFICATION COMPARISON (round 2)")
    print("=" * 108)
    print(out.round(3).to_string())

    base = out.iloc[0]
    print("\nRelative to baseline:")
    for name, r in out.iloc[1:].iterrows():
        print(f"  {name:34s} c sd {r['c_sd']/base['c_sd']:5.2f}x | "
              f"r* band {r['rstar_band_avg']/base['rstar_band_avg']:5.2f}x | "
              f"pi* end band {r['pistar_band_end']/base['pistar_band_end']:5.2f}x")
    return out


if __name__ == '__main__':
    main()
