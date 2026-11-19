# Policy brief: did the regional child-benefit pilot reduce poverty? (synthetic data)

> **Illustrative only.** Data are synthetic (`src/generate_panel.py`); the true effect is known by construction,
> which lets us check that the estimators work. Nothing here describes a real country or programme.

## Key findings

1. **Social transfers reduce poverty.** In wave 1, 18.5% of households were below the
   poverty line before transfers and 16.5% after, a relative reduction of 10%
   (rising to 13% in wave 4, after the pilot).
2. **The pilot reduced poverty among households with children.** Two-way fixed-effects difference-in-differences
   estimates a **-3.3 pp** change in poverty risk (95% CI -4.8 to -1.8),
   with standard errors clustered by region (120 clusters).
3. **Incomes rose.** Equivalised income in pilot regions rose by about EUR 477 relative to other regions.
4. **Checks pass.** Pre-pilot event-study coefficient (wave 1) is +1.0 pp (CI includes zero:
   yes); the placebo on households without children is
   -0.1 pp (p = 0.89); the balanced-panel estimate is -3.3 pp, close to the
   main result, so attrition does not appear to drive the finding; a plain 2x2 comparison of means gives -2.8 pp.

## Policy implications

- Child-focused cash transfers are an effective, targetable lever for reducing household poverty.
- The event study shows the effect appears as soon as the benefit starts (wave 3) and persists in wave 4, so
  tracking households over several waves matters to confirm that gains last.
- Pilot-style rollout across randomly selected regions makes credible evaluation possible; this is worth designing
  in from the start.

## Method notes

- Poverty line anchored at 60% of each country's wave-1 median equivalised income (modified OECD scale), so the line
  does not move with the policy. "Before transfers" uses market income only.
- Sample for DiD: households with children (the eligible group). Household and wave fixed effects; treatment =
  pilot region x post (waves 3-4). Clustering by region because the policy varies at region level.
- Event-study reference period is wave 2. Informative attrition is mild by design; the balanced panel is a robustness check.
- Benchmark: because the data are simulated, we can compute the mechanical effect of the bonus on treated
  households (poverty with vs. without the bonus, ignoring dynamics): **-2.7 pp**. The DiD estimate
  (-3.3 pp) should be and is close to it.
