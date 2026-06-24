# TV-MORL on MovieLens-1M

This folder implements a time-varying multi-objective reinforcement learning method, `TV-MORL`, built on top of the two-stage Pareto discovery idea of `C-MORL`.

Concretely, we retain the original `C-MORL` pipeline:

- initialization with multiple weighted-sum preference vectors
- external Pareto archive maintenance
- constrained extension that improves one objective while preserving the others
- preference-aware policy allocation at evaluation time

The main change is that the static multi-objective return is replaced by a time-varying objective process driven by:

- `P_t`: time-varying user preference
- `R_t`: time-varying reward shaping over relevance, diversity, and novelty

## Dataset

We validate `TV-MORL` on the real `MovieLens-1M` dataset located at:

```text
C:\Users\86198\ml-1m
```

The dataset used in the formal experiments contains:

- `6040` users
- `1,000,209` ratings
- `3883` movies in `movies.dat`
- `3706` movies that actually appear in `ratings.dat`

The formal run uses:

- all `3883` movie entries from `movies.dat` as candidate items
- `min_user_ratings = 20`
- final usable users: `6040`

## Method

`TV-MORL` extends `C-MORL` with a time-varying recommendation environment.

### Time-varying preference

The environment maintains:

- a long-term user profile from historical ratings
- a short-term session profile from recent interaction context
- a bounded time gate controlling the drift from short-term interest back toward long-term preference
- a mild seasonal perturbation term

### Time-varying reward

The reward vector contains three objectives:

- `Relevance`
- `Diversity`
- `Novelty`

The relevance term uses the revised bounded formulation:

- long-term match
- short-term match
- an optional popularity auxiliary term
- an explicit-rating fusion weight

All key weights are bounded and configurable, so the reward definition can be ablated directly.

## Code Structure

- [movielens_env.py](C:\Users\86198\Documents\Codex\2026-06-23\new-chat\tv_morl_ml1m\movielens_env.py)
  MovieLens-1M time-varying environment and dataset loader
- [tv_model.py](C:\Users\86198\Documents\Codex\2026-06-23\new-chat\tv_morl_ml1m\tv_model.py)
  multi-objective actor-critic PPO and IPO-style constrained update
- [tv_morl.py](C:\Users\86198\Documents\Codex\2026-06-23\new-chat\tv_morl_ml1m\tv_morl.py)
  two-stage trainer and Pareto archive
- [run_tv.py](C:\Users\86198\Documents\Codex\2026-06-23\new-chat\tv_morl_ml1m\run_tv.py)
  training entry, evaluation, ablation, and plotting

## Basic Usage

To run the formal `TV-MORL` experiment:

```powershell
python C:\Users\86198\Documents\Codex\2026-06-23\new-chat\tv_morl_ml1m\run_tv.py `
  --data-dir C:\Users\86198\ml-1m `
  --num-steps 4000 `
  --init-steps 2000 `
  --min-user-ratings 20 `
  --rt-rel-strength 0.1 `
  --rt-div-strength 0.1 `
  --rt-nov-strength 0.1 `
  --time-gate-min 0.15 `
  --time-gate-max 0.65 `
  --rel-static-gate 0.45 `
  --rel-pop-weight 0.10 `
  --rel-explicit-weight 0.70 `
  --save-dir C:\Users\86198\Documents\Codex\2026-06-23\new-chat\results\tv_morl_formal_rel_recheck\gate_0p45
```

To run the relevance-formula ablation:

```powershell
python C:\Users\86198\Documents\Codex\2026-06-23\new-chat\tv_morl_ml1m\run_tv.py `
  --data-dir C:\Users\86198\ml-1m `
  --num-steps 4000 `
  --init-steps 2000 `
  --min-user-ratings 20 `
  --rt-rel-strength 0.1 `
  --rt-div-strength 0.1 `
  --rt-nov-strength 0.1 `
  --time-gate-min 0.15 `
  --time-gate-max 0.65 `
  --rel-static-gate 0.35 `
  --rel-pop-weight 0.10 `
  --rel-explicit-weight 0.70 `
  --save-dir C:\Users\86198\Documents\Codex\2026-06-23\new-chat\results\tv_morl_formal_rel_formula_scan `
  --run-rel-formula-ablation
```

## Main Results

The current formal `TV-MORL` result is stored at:

- [summary.txt](C:\Users\86198\Documents\Codex\2026-06-23\new-chat\results\tv_morl_formal_rel_recheck\gate_0p45\summary.txt)

Key metrics:

- `Final HV = 3391.594862`
- `Final EU = 16.884902`
- `Final SP = 7.619808`

For comparison, the real original-style `C-MORL` baseline is stored at:

- [summary.txt](C:\Users\86198\Documents\Codex\2026-06-23\new-chat\results\cmorl_movielens_real\summary.txt)

Key baseline metrics:

- `Final HV = 2513.631492`
- `Final EU = 16.567879`
- `Final SP = 8.566774`

Under the current formal run, `TV-MORL` improves over the real `C-MORL` baseline on both `HV` and `EU`, while `SP` decreases relative to the baseline.

The relevance-formula ablation is stored at:

- [summary.txt](C:\Users\86198\Documents\Codex\2026-06-23\new-chat\results\tv_morl_formal_rel_formula_scan\summary.txt)
- [rel_formula_ablation.png](C:\Users\86198\Documents\Codex\2026-06-23\new-chat\results\tv_morl_formal_rel_formula_scan\rel_formula_ablation.png)

Its current formal-budget result shows:

- `c_rel = 0.45` gives the best `HV` and `EU` among the scanned settings
- reducing `c_pop` to `0.00` improves `HV`, but does not beat `c_rel = 0.45` on `EU`
- `eta_rel = 0.50` to `0.70` is more stable than `eta_rel = 0.85`

## Reference

The implementation in this folder follows the two-stage Pareto discovery framework of `C-MORL`, and adapts it to a time-varying recommendation setting on `MovieLens-1M`.
