# TV-MORL on MovieLens-1M

This repository contains the complete code and final retained experiment results for the paper-style validation of `TV-MORL` on `MovieLens-1M`.

`TV-MORL` is a time-varying multi-objective reinforcement learning method built on the two-stage Pareto discovery structure of `C-MORL`. The final experimental setting in this repository only models two time-varying factors:

- `P_t`: time-varying user preference
- `R_t`: time-varying reward

## Repository Layout

- [tv_morl_ml1m](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/tv_morl_ml1m)
  Complete implementation used in the final experiments.
- [results](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/results)
  Final experiment outputs retained for reproducibility.
- [results/README.md](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/results/README.md)
  Mapping from each retained result folder to the corresponding table or figure usage.
- [TV-MORL_PtRt_MovieLens_论文正文_可复制版.md](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/TV-MORL_PtRt_MovieLens_论文正文_可复制版.md)
  Final paper-style experiment report synchronized with the current code and retained results.

## Dataset

The code expects the real `MovieLens-1M` dataset at:

```text
C:\Users\86198\ml-1m
```

The formal experiments use:

- all `3883` candidate movies from `movies.dat`
- `min_user_ratings = 20`
- final usable users: `6040`

## Final Retained Results

The `results/` directory keeps only the artifacts used in the final report:

- `tv_morl_pt_rt_same_env_multiseed`
  5-seed same-environment baseline comparison for `freeze_all`, `pt_only`, `rt_only`, and reference `full_tv`
- `tv_morl_pt_rt_lambda_ablation`
  single-seed lambda scan used for coarse ablation
- `tv_morl_pt_rt_lambda_key_multiseed_consistent`
  3-seed key-point lambda verification used in the final analysis
- `tv_morl_pt_rt_pareto_final_anchor`
  Pareto front figure under the final anchor configuration
- `tv_morl_full_tv_tuned_5seed`
  5-seed tuned `full_tv` experiment with better-performing lambda combination

The fair same-environment 5-seed baseline summary is:

- `freeze_all`: `HV = 2180.773930 +- 438.227498`, `EU = 13.630243 +- 0.903903`, `SP = 16.130532 +- 10.551756`
- `pt_only`: `HV = 2728.073283 +- 618.843497`, `EU = 14.198089 +- 0.958873`, `SP = 11.452986 +- 6.207802`
- `rt_only`: `HV = 3076.563424 +- 622.635564`, `EU = 15.395301 +- 1.060080`, `SP = 14.910919 +- 10.656806`
- `full_tv` reference: `HV = 3735.721102 +- 719.940432`, `EU = 15.772512 +- 0.877925`, `SP = 8.839561 +- 2.089377`

The tuned 5-seed `full_tv` summary retained separately is:

- `lambda_rel = 0.20`, `lambda_div = 0.00`, `lambda_nov = 0.00`
- `HV = 4005.166656 +- 571.364430`
- `EU = 17.011926 +- 0.622961`
- `SP = 7.812146 +- 4.673676`

`SP` denotes sparsity, so smaller values are better.

## Quick Start

See [tv_morl_ml1m/README.md](C:/Users/86198/Documents/Codex/2026-06-23/new-chat/tv_morl_ml1m/README.md) for exact commands.
