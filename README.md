# Social Transfers and Poverty: Panel Data and Difference-in-Differences

A reproducible Python project that (1) measures how much social transfers reduce poverty and (2) evaluates a
regional child-benefit pilot with panel data, two-way fixed effects, an event study, and robustness checks.

> **Data note.** The repository generates a **synthetic** 4-wave household panel (6 countries x 20 regions,
> ~15,000 households). Because the pilot's effect is built in, the estimators can be checked against a known
> benchmark. Results are **not** statistics about any country. The same code structure applies to real longitudinal
> survey data such as the EU-SILC longitudinal files (household id x wave).

## Question

Did a child-benefit increase, piloted in 40% of randomly chosen regions from wave 3, reduce poverty among
households with children?

## Approach

| Component | Detail |
|---|---|
| Poverty measurement | 60% of wave-1 median equivalised income (anchored line), modified OECD scale; "before transfers" uses market income only |
| Transfers | share of poverty removed by transfers, by wave |
| Identification | difference-in-differences: pilot vs non-pilot regions, households with children, before vs after wave 3 |
| Estimator | `linearmodels.PanelOLS` with household and wave fixed effects; SEs clustered by region (120 clusters) |
| Dynamics | event study with wave 2 as reference (pre-trend check at wave 1) |
| Robustness | placebo on households without children; balanced panel (attrition is mildly informative); log income; plain 2x2 means |
| Validation | mechanical effect of the bonus computed from the simulation; unit tests recover a known effect exactly |

## Run it

```bash
pip install -r requirements.txt
python run_all.py       # generate panel -> analyse
pytest -q               # unit tests (estimator recovers a known effect)
```

Outputs in `outputs/`: `poverty_by_wave.csv`, `did_results.csv`, `event_study.csv`, `figures/`, `policy_brief.md`.

## Layout

```
src/generate_panel.py   simulation (AR(1) income shocks, job loss, take-up, attrition, regional shocks)
src/analysis.py         indicators, TWFE DiD, event study, robustness, figures, brief
tests/test_did.py       estimator tests
outputs/                tables, figures, policy brief
```

## Limitations

- Synthetic data; effect sizes depend on simulation parameters.
- Two-way fixed effects with a single adoption date (wave 3) avoids the staggered-timing problems of TWFE; with
  staggered adoption, use heterogeneity-robust estimators (e.g. Callaway and Sant'Anna).
- Anchored poverty line is a modelling choice; a relative line that moves with the median would give smaller
  measured effects if the policy shifts the median.
- Unweighted analysis; real panel surveys require longitudinal weights.

## Skills demonstrated

Panel data management, causal inference (DiD, event study, placebo tests), clustered inference, poverty
measurement, simulation-based validation, testing, and policy communication.
