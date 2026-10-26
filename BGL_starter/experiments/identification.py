"""
Identification experiments for the India BGL model.

Answers two questions with numbers rather than assertions:

  1. Does restricting the non-growth component z buy identification of c?
     The baseline has FOUR independent random walks (pi*, x*, g, z) competing
     to explain low-frequency movement. z is a free random walk, so it can
     absorb any trend that c*g would otherwise have to explain -- which is
     exactly why c is only weakly identified. Shrinking sigma_z towards zero
     turns r* = 4c*g + zbar into "trend growth, scaled, plus a constant",
     forcing all low-frequency movement in r* through g and so identifying c
     off the co-movement of the yield-curve level with trend growth.
     (This is the Davis et al. (2024) cointegration restriction of the paper's
     footnote 17, in its sharpest form.)

  2. Does a survey of expected future 91-day T-bill rates help?
     The paper attributes a large share of its precision to exactly this
     series: the r* interpercentile range falls from 2.9pp to 1.7pp once the
     Consensus 1-year-ahead T-bill forecast becomes available in 1990Q1.
     India has no such series in Data.xlsx, so we PROXY it with the realized
     4-quarter-ahead 91-day rate and a correspondingly loose measurement
     error. Realized = expectation + forecast error, so this is a valid but
     noisy measure of the same object; the proxy uses future information, so
     treat the resulting numbers as an upper bound on what a real survey with
     the same noise would deliver, and the direction/rough size as the signal.

Run:  python experiments/identification.py
"""

import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_model as rm

N_DRAWS, N_BURNIN, THIN = 20000, 10000, 10


def build_specs(df):
    """Return {name: (dataframe, priors, config)} for each specification."""
    base_pri = rm.default_priors()

    # z pinned to a constant: sigma_z ~ 0 makes z a fixed (but estimated,
    # via P0[6,6]) level, so r* = 4c*g + zbar.
    pinned_pri = copy.deepcopy(base_pri)
    pinned_pri['sig_z'] = (200, 1e-4)      # prior mean sd ~7e-4, effectively 0

    # Proxy short-rate survey: realized 91-day rate 4 quarters ahead.
    df_sr = df.copy()
    df_sr['survey_i_4q'] = df_sr['y_1'].shift(-4)

    return {
        'A. baseline (z random walk)':        (df,    base_pri),
        'B. z pinned (r* = 4c*g + zbar)':     (df,    pinned_pri),
        'C. baseline + T-bill survey':        (df_sr, base_pri),
        'D. z pinned + T-bill survey':        (df_sr, pinned_pri),
    }


def summarize(name, res, df):
    p = res['params']
    c = np.array([d['c'] for d in p])
    w = np.array([d['omega'] for d in p])
    rs = res['rstar']
    band = np.percentile(rs, 95, axis=0) - np.percentile(rs, 5, axis=0)
    return {
        'spec': name,
        'c_med': np.median(c),
        'c_sd': c.std(),
        'c_90pct_width': np.percentile(c, 95) - np.percentile(c, 5),
        'omega_med': np.median(w),
        'omega_sd': w.std(),
        'rstar_end': np.median(rs[:, -1]),
        'rstar_mean': np.median(rs, axis=0).mean(),
        'rstar_IPR_avg': band.mean(),
        'rstar_IPR_end': band[-1],
        'sig_z': np.median([d['sig_z'] for d in p]),
    }


def main():
    df = rm.load_india_data('Data.xlsx')
    rows = []
    for name, (data, pri) in build_specs(df).items():
        print(f"\n=== {name} ===", flush=True)
        cfg = rm.BGLConfig(n_draws=N_DRAWS, n_burnin=N_BURNIN, thin=THIN)
        np.random.seed(0)
        res = rm.gibbs_sampler(data, config=cfg, priors=pri, verbose=False)
        row = summarize(name, res, data)
        rows.append(row)
        print(f"  c = {row['c_med']:.3f} (sd {row['c_sd']:.3f}, "
              f"90% width {row['c_90pct_width']:.3f}) | "
              f"omega = {row['omega_med']:+.2f} (sd {row['omega_sd']:.2f})")
        print(f"  r* end = {row['rstar_end']:+.2f}%, mean = {row['rstar_mean']:+.2f}%, "
              f"avg 90% band = {row['rstar_IPR_avg']:.2f}pp")

    out = pd.DataFrame(rows).set_index('spec')
    print("\n" + "=" * 100)
    print("IDENTIFICATION COMPARISON")
    print("=" * 100)
    print(out.round(3).to_string())

    base = out.iloc[0]
    print("\nRelative to baseline:")
    for name, r in out.iloc[1:].iterrows():
        print(f"  {name:36s} c sd {r['c_sd']/base['c_sd']:5.2f}x | "
              f"r* band {r['rstar_IPR_avg']/base['rstar_IPR_avg']:5.2f}x")
    return out


if __name__ == '__main__':
    main()
