"""Generate a synthetic 4-wave household panel with a regional child-benefit pilot.

Design
------
* 6 countries x 20 regions; 40% of regions in each country are randomly chosen to pilot a
  child-benefit increase that starts in wave 3.
* Household incomes follow persistent (AR(1)) shocks, job-loss spells, regional shocks, and
  means-tested minimum-income top-ups with incomplete take-up.
* Attrition is mildly informative (low-income households drop out more often).
* The TRUE effect of the pilot is known by construction: +0.03 x country scale per child (annual EUR),
  which makes it possible to check that the estimators recover it.

All data are SYNTHETIC. Nothing here describes a real country or programme.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = {"DE": (32000, 0.55), "ES": (24000, 0.62), "IT": (25000, 0.60),
             "PL": (15000, 0.58), "RO": (10000, 0.70), "BG": (9500, 0.72)}
EDU = {"low": 0.0, "medium": 0.28, "high": 0.62}
N_WAVES = 4
BASE_CHILD_BEN, PILOT_BONUS = 0.03, 0.03   # share of country scale, per child


def generate(n_hh: int = 2500, n_regions: int = 20, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, hid = [], 0
    for country, (scale, sigma) in COUNTRIES.items():
        regions = [f"{country}-{r:02d}" for r in range(n_regions)]
        pilot = set(rng.choice(regions, size=int(0.4 * n_regions), replace=False))
        reg_eff = {r: rng.normal(0, 0.15) for r in regions}
        reg_shock = {(r, t): rng.normal(0, 0.03) for r in regions for t in range(1, N_WAVES + 1)}
        for _ in range(n_hh):
            hid += 1
            region = str(rng.choice(regions))
            n_adults = int(rng.choice([1, 2], p=[0.3, 0.7]))
            n_child = int(rng.choice([0, 1, 2, 3], p=[0.40, 0.28, 0.24, 0.08]))
            edu = str(rng.choice(list(EDU), p=[0.25, 0.50, 0.25]))
            eq_size = 1 + 0.5 * (n_adults - 1) + 0.3 * n_child
            a = rng.normal(0, sigma * 0.6) + EDU[edu] + reg_eff[region]
            takeup = rng.random() < 0.55
            u, jobloss = rng.normal(0, sigma * 0.35), False
            for t in range(1, N_WAVES + 1):
                if t > 1:
                    u = 0.7 * u + rng.normal(0, sigma * 0.25)
                    jobloss = (jobloss and rng.random() < 0.5) or rng.random() < 0.05
                market = (scale * (0.8 + 0.4 * (n_adults - 1))
                          * np.exp(a + u + 0.015 * t + reg_shock[(region, t)] - (0.7 if jobloss else 0.0)))
                market_eq = market / eq_size
                in_pilot = region in pilot
                rate = BASE_CHILD_BEN + (PILOT_BONUS if (in_pilot and t >= 3) else 0.0)
                child_ben = rate * scale * n_child / eq_size
                top_up = 0.9 * max(0.30 * scale - market_eq, 0) if takeup else 0.0
                eq_income = market_eq + child_ben + top_up
                rows.append(dict(hid=hid, wave=t, country=country, region=region, pilot=int(in_pilot),
                                 n_adults=n_adults, n_children=n_child, edu=edu, scale=scale,
                                 market_eq=market_eq, child_ben=child_ben, top_up=top_up,
                                 eq_income=eq_income, jobloss=int(jobloss)))
                # informative attrition after this wave
                if t < N_WAVES and rng.random() < 0.04 + (0.08 if eq_income < 0.5 * scale else 0.0):
                    break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(out / "panel_synthetic.csv", index=False)
    n = df.hid.nunique()
    print(f"{n:,} households, {len(df):,} household-wave rows "
          f"({df.groupby('hid').size().eq(N_WAVES).mean():.0%} observed in all {N_WAVES} waves)")
