# Result Index

This directory keeps the final experiment outputs retained for the report. All reported numbers should map back to one of the folders below.

## Retained Result Sets

- `tv_morl_pt_rt_same_env_multiseed`
  Same-environment 5-seed comparison of `freeze_all`, `pt_only`, `rt_only`, and reference `full_tv`.
  Main summary: `mean_std_summary_all_methods.txt`
- `tv_morl_pt_rt_lambda_ablation`
  Formal single-seed coarse scan over `lambda_rel`, `lambda_div`, and `lambda_nov`.
  Main files: `summary.txt`, `lambda_decoupled_ablation.png`
- `tv_morl_pt_rt_lambda_key_multiseed_consistent`
  3-seed key-point verification under a consistent anchor configuration.
  Main summary: `mean_std_summary_key_points_all.txt`
- `tv_morl_pt_rt_pareto_final_anchor`
  Pareto front visualization used in the report.
  Anchor configuration: `lambda_rel=0.10`, `lambda_div=0.10`, `lambda_nov=0.00`
- `tv_morl_full_tv_tuned_5seed`
  Tuned 5-seed `full_tv` experiment.
  Tuned configuration: `lambda_rel=0.20`, `lambda_div=0.00`, `lambda_nov=0.00`

## Notes

- `SP` denotes sparsity, so smaller values are better.
- The same-environment baseline is the fair comparison used against `TV-MORL`.
- The tuned `full_tv` result is kept separately from the controlled same-environment reference `full_tv` because they serve different purposes:
  controlled comparison versus best-performing lambda setting.
