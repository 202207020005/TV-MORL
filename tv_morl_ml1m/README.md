# TV-MORL Code

This folder contains the complete implementation used for the final `TV-MORL` experiments on `MovieLens-1M`.

## Files

- [movielens_env.py](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/tv_morl_ml1m/movielens_env.py)
  dataset loader and time-varying recommendation environment
- [tv_model.py](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/tv_morl_ml1m/tv_model.py)
  multi-objective actor-critic model, rollout buffer, PPO update, and IPO-style extension update
- [tv_morl.py](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/tv_morl_ml1m/tv_morl.py)
  two-stage Pareto discovery trainer
- [run_tv.py](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/tv_morl_ml1m/run_tv.py)
  main entry for training, same-environment comparison, and lambda ablation

## Install

From the repository root:

```powershell
pip install -r requirements.txt
```

## Reproduce the Main Same-Environment 5-Seed Experiment

Reference `full_tv` uses the controlled comparison configuration:

- `lambda_rel = 0.10`
- `lambda_div = 0.10`
- `lambda_nov = 0.10`

Single run example:

```powershell
python tv_morl_ml1m/run_tv.py `
  --data-dir C:\Users\86198\ml-1m `
  --save-dir results\example_full_tv_ref_seed42 `
  --num-steps 4000 `
  --init-steps 2000 `
  --min-user-ratings 20 `
  --seed 42 `
  --rt-rel-strength 0.10 `
  --rt-div-strength 0.10 `
  --rt-nov-strength 0.10 `
  --time-gate-min 0.15 `
  --time-gate-max 0.65 `
  --seasonal-amp 0.0 `
  --rel-static-gate 0.35 `
  --rel-pop-weight 0.10 `
  --rel-explicit-weight 0.70 `
  --run-same-env-baseline
```

The retained aggregated final result is in:

- [mean_std_summary_all_methods.txt](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/results/tv_morl_pt_rt_same_env_multiseed/mean_std_summary_all_methods.txt)

To reproduce the retained 5-seed structure, repeat the same command with:

- `--seed 41 --save-dir results\tv_morl_pt_rt_same_env_multiseed\seed_41`
- `--seed 42 --save-dir results\tv_morl_pt_rt_same_env_multiseed\seed_42`
- `--seed 43 --save-dir results\tv_morl_pt_rt_same_env_multiseed\seed_43`
- `--seed 44 --save-dir results\tv_morl_pt_rt_same_env_multiseed\seed_44`
- `--seed 45 --save-dir results\tv_morl_pt_rt_same_env_multiseed\seed_45`

## Reproduce the Final Lambda Analysis

Single-seed coarse scan:

```powershell
python tv_morl_ml1m/run_tv.py `
  --data-dir C:\Users\86198\ml-1m `
  --save-dir results\example_lambda_scan `
  --num-steps 4000 `
  --init-steps 2000 `
  --min-user-ratings 20 `
  --seed 42 `
  --rt-rel-strength 0.10 `
  --rt-div-strength 0.10 `
  --rt-nov-strength 0.10 `
  --time-gate-min 0.15 `
  --time-gate-max 0.65 `
  --seasonal-amp 0.0 `
  --rel-static-gate 0.35 `
  --rel-pop-weight 0.10 `
  --rel-explicit-weight 0.70 `
  --run-lambda-decoupled-ablation
```

Retained final ablation outputs:

- [summary.txt](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/results/tv_morl_pt_rt_lambda_ablation/summary.txt)
- [lambda_decoupled_ablation.png](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/results/tv_morl_pt_rt_lambda_ablation/lambda_decoupled_ablation.png)
- [mean_std_summary_key_points_all.txt](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/results/tv_morl_pt_rt_lambda_key_multiseed_consistent/mean_std_summary_key_points_all.txt)

The retained Pareto-front figure used in the report is:

- [pareto_front.png](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/results/tv_morl_pt_rt_pareto_final_anchor/pareto_front.png)
  anchor configuration: `lambda_rel = 0.10`, `lambda_div = 0.10`, `lambda_nov = 0.00`

## Tuned Full-TV Result

The tuned 5-seed configuration retained in this repository is:

- `lambda_rel = 0.20`
- `lambda_div = 0.00`
- `lambda_nov = 0.00`

Retained result directory:

- [tv_morl_full_tv_tuned_5seed](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/results/tv_morl_full_tv_tuned_5seed)

This tuned configuration improves over the reference `full_tv` mean on HV, EU, and SP in the current experiments.
