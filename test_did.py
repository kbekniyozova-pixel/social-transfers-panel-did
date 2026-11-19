import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis import manual_2x2, twfe_did  # noqa: E402


def _toy_panel(effect=-0.05, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    region_pilot = {f"r{i}": int(i % 2 == 0) for i in range(20)}
    for h in range(400):
        region = f"r{h % 20}"
        alpha = rng.normal(0, 0.1)
        for w in range(1, 5):
            pilot = region_pilot[region]
            post = int(w >= 3)
            y = 0.3 + alpha + 0.01 * w + effect * pilot * post + rng.normal(0, noise)
            rows.append(dict(hid=h, wave=w, region=region, pilot=pilot, post=post,
                             treat_post=pilot * post, y=y))
    return pd.DataFrame(rows)


def test_twfe_recovers_known_effect_without_noise():
    res = twfe_did(_toy_panel(), "y")
    assert abs(res["coef"] - (-0.05)) < 1e-8


def test_twfe_close_to_effect_with_noise():
    res = twfe_did(_toy_panel(noise=0.05), "y")
    assert abs(res["coef"] - (-0.05)) < 0.01


def test_manual_2x2_matches_twfe_on_balanced_panel():
    df = _toy_panel()
    assert abs(manual_2x2(df, "y") - twfe_did(df, "y")["coef"]) < 1e-8
