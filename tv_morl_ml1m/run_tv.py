import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tv_morl_ml1m.movielens_env import MovieLensRealTVEnv, load_movielens_bundle
from tv_morl_ml1m.tv_morl import TVMORLTrainer


@dataclass
class TVRunConfig:
    data_dir: str = r"C:\Users\86198\ml-1m"
    save_dir: str = os.path.join(ROOT_DIR, "results", "tv_morl_ml1m")

    obj_num: int = 3
    n_items: int | None = None
    min_user_ratings: int = 20
    max_episode_steps: int = 40

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_steps: int = 32
    num_processes: int = 2
    ppo_epoch: int = 3
    num_mini_batch: int = 4
    clip_param: float = 0.2

    num_time_steps: int = 4000
    num_init_steps: int = 2000
    min_weight: float = 0.0
    max_weight: float = 1.0
    delta_weight: float = 0.5
    eval_delta_weight: float = 0.1
    update_iter: int = 4
    eval_num: int = 3
    eval_gamma: float = 0.99
    num_select: int = 3
    policy_buffer: int = 30
    ref_point: tuple[float, float, float] = (0.0, 0.0, 0.0)

    beta: float = 0.85
    t: int = 12
    seed: int = 42
    rl_eval_interval: int = 3

    enable_pt: bool = True
    enable_rt: bool = True
    rt_rel_strength: float = 1.0
    rt_div_strength: float = 1.0
    rt_nov_strength: float = 1.0
    time_gate_min: float = 0.15
    time_gate_max: float = 0.65
    seasonal_amp: float = 0.0
    rel_static_gate: float = 0.35
    rel_pop_weight: float = 0.10
    rel_explicit_weight: float = 0.70
    div_scale_min: float = 0.60
    nov_scale_min: float = 0.50
    nov_repeat_penalty_max: float = 0.20
    session_keep_rate: float = 0.85
    session_update_rate: float = 0.15


def visualize_preference_drift(bundle, save_path: str) -> None:
    env = MovieLensRealTVEnv(bundle, max_steps=40, seed=42, enable_pt=True, enable_rt=True)
    env.reset(seed=42)
    prefs = [env.current_preference.copy()]
    for _ in range(40):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        prefs.append(env.current_preference.copy())
        if terminated or truncated:
            break
    env.close()

    prefs = np.array(prefs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for idx in range(min(6, prefs.shape[1])):
        axes[0].plot(prefs[:, idx], linewidth=2, label=bundle.genre_names[idx])
    axes[0].set_title("Time-varying preference $P_t$")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Genre weight")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(prefs.T, aspect="auto", cmap="YlOrRd")
    axes[1].set_title("Preference heatmap")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Genre")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_pareto_front(obj_array: np.ndarray, save_path: str) -> None:
    if len(obj_array) < 2:
        return

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(obj_array[:, 0], obj_array[:, 1], obj_array[:, 2], c=obj_array[:, 2], cmap="viridis", s=40)
    ax.set_xlabel("Relevance")
    ax.set_ylabel("Diversity")
    ax.set_zlabel("Novelty")
    ax.set_title("Pareto front")

    ax2 = fig.add_subplot(122)
    ax2.scatter(obj_array[:, 0], obj_array[:, 1], label="Rel-Div", alpha=0.7)
    ax2.scatter(obj_array[:, 0], obj_array[:, 2], label="Rel-Nov", alpha=0.7)
    ax2.scatter(obj_array[:, 1], obj_array[:, 2], label="Div-Nov", alpha=0.7)
    ax2.set_title("Objective projections")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_lambda_decoupled_ablation(rows: list[dict], save_path: str) -> None:
    groups = {"rel": [], "div": [], "nov": []}
    for row in rows:
        groups[row["param"]].append(row)
    for param in groups:
        groups[param] = sorted(groups[param], key=lambda item: item["strength"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    titles = {
        "rel": r"$\lambda_{rel}$ scan",
        "div": r"$\lambda_{div}$ scan",
        "nov": r"$\lambda_{nov}$ scan",
    }
    colors = {"hv": "#c0392b", "eu": "#2980b9", "sp": "#27ae60"}

    for ax, param in zip(axes, ["rel", "div", "nov"]):
        rows_param = groups[param]
        if not rows_param:
            continue
        x = [row["strength"] for row in rows_param]
        ax.plot(x, [row["final_hv"] for row in rows_param], marker="o", linewidth=2, color=colors["hv"], label="HV")
        ax.plot(x, [row["final_eu"] for row in rows_param], marker="s", linewidth=2, color=colors["eu"], label="EU")
        ax.plot(x, [row["final_sp"] for row in rows_param], marker="^", linewidth=2, color=colors["sp"], label="SP")
        ax.set_title(titles[param])
        ax.set_xlabel("Strength")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Metric value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_same_env_baseline(rows: list[dict], save_path: str) -> None:
    labels = [row["label"] for row in rows]
    hv = [row["final_hv"] for row in rows]
    eu = [row["final_eu"] for row in rows]
    sp = [row["final_sp"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for ax, values, title, color in zip(
        axes,
        [hv, eu, sp],
        ["Hypervolume", "Expected Utility", "Sparsity"],
        ["#7f8c8d", "#1f77b4", "#2ecc71"],
    ):
        ax.bar(range(len(labels)), values, color=color, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_config_from_cli(cli_args: argparse.Namespace) -> TVRunConfig:
    config = TVRunConfig()
    config.data_dir = cli_args.data_dir
    config.num_time_steps = cli_args.num_steps
    config.num_init_steps = cli_args.init_steps
    config.n_items = None if cli_args.n_items <= 0 else cli_args.n_items
    config.min_user_ratings = cli_args.min_user_ratings
    config.seed = cli_args.seed
    config.rt_rel_strength = cli_args.rt_rel_strength
    config.rt_div_strength = cli_args.rt_div_strength
    config.rt_nov_strength = cli_args.rt_nov_strength
    config.time_gate_min = cli_args.time_gate_min
    config.time_gate_max = cli_args.time_gate_max
    config.seasonal_amp = cli_args.seasonal_amp
    config.rel_static_gate = cli_args.rel_static_gate
    config.rel_pop_weight = cli_args.rel_pop_weight
    config.rel_explicit_weight = cli_args.rel_explicit_weight
    config.div_scale_min = cli_args.div_scale_min
    config.nov_scale_min = cli_args.nov_scale_min
    config.nov_repeat_penalty_max = cli_args.nov_repeat_penalty_max
    config.session_keep_rate = cli_args.session_keep_rate
    config.session_update_rate = cli_args.session_update_rate
    if cli_args.save_dir is not None:
        config.save_dir = cli_args.save_dir
    return config


def print_run_header(bundle, config: TVRunConfig) -> None:
    print("=" * 72)
    print("TV-MORL on MovieLens-1M")
    print("=" * 72)
    print("raw dataset users=6040 movies_file=3883 rated_movies=3706 ratings=1000209")
    print(
        f"experiment subset users={len(bundle.user_ids)} candidate_items={len(bundle.movie_ids)} "
        f"min_user_ratings={config.min_user_ratings}"
    )
    print(f"Pt/Rt only setup: seasonal_amp={config.seasonal_amp}")
    print(
        f"Rt strengths: rel={config.rt_rel_strength}, div={config.rt_div_strength}, nov={config.rt_nov_strength}"
    )
    print(
        f"rel formula: gate=[{config.time_gate_min}, {config.time_gate_max}], "
        f"static_gate={config.rel_static_gate}, pop={config.rel_pop_weight}, "
        f"explicit={config.rel_explicit_weight}"
    )
    print("=" * 72)


def run_one_setting(config: TVRunConfig, save_dir: str):
    run_config = TVRunConfig(**config.__dict__)
    run_config.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    trainer = TVMORLTrainer(run_config)
    results = trainer.run()
    obj_array = results["obj_array"]
    np.save(os.path.join(save_dir, "final_obj_array.npy"), obj_array)
    visualize_pareto_front(obj_array, os.path.join(save_dir, "pareto_front.png"))
    return trainer.bundle, results


def run_same_environment_baseline(config: TVRunConfig) -> list[dict]:
    rows = []
    baseline_settings = [
        ("freeze_all", False, False, 0.0, 0.0, 0.0),
        ("pt_only", True, False, 0.0, 0.0, 0.0),
        ("rt_only", False, True, config.rt_rel_strength, config.rt_div_strength, config.rt_nov_strength),
        ("full_tv", True, True, config.rt_rel_strength, config.rt_div_strength, config.rt_nov_strength),
    ]
    for label, enable_pt, enable_rt, rel_s, div_s, nov_s in baseline_settings:
        sub_config = TVRunConfig(**config.__dict__)
        sub_config.enable_pt = enable_pt
        sub_config.enable_rt = enable_rt
        sub_config.rt_rel_strength = rel_s
        sub_config.rt_div_strength = div_s
        sub_config.rt_nov_strength = nov_s
        run_dir = os.path.join(config.save_dir, "same_env_baseline", label)
        _, results = run_one_setting(sub_config, run_dir)
        rows.append(
            {
                "label": label,
                "enable_pt": enable_pt,
                "enable_rt": enable_rt,
                "final_hv": results["final_hv"],
                "final_eu": results["final_eu"],
                "final_sp": results["final_sp"],
            }
        )
    visualize_same_env_baseline(rows, os.path.join(config.save_dir, "same_env_baseline.png"))
    return rows


def run_lambda_decoupled_ablation(config: TVRunConfig) -> list[dict]:
    rows = []
    lambda_grid = [0.0, 0.05, 0.10, 0.15, 0.20]
    for param_name in ["rel", "div", "nov"]:
        for strength in lambda_grid:
            sub_config = TVRunConfig(**config.__dict__)
            if param_name == "rel":
                sub_config.rt_rel_strength = strength
            elif param_name == "div":
                sub_config.rt_div_strength = strength
            else:
                sub_config.rt_nov_strength = strength
            label = f"{param_name}_{strength:.2f}"
            run_dir = os.path.join(config.save_dir, "lambda_decoupled_ablation", label)
            _, results = run_one_setting(sub_config, run_dir)
            rows.append(
                {
                    "label": label,
                    "param": param_name,
                    "strength": strength,
                    "final_hv": results["final_hv"],
                    "final_eu": results["final_eu"],
                    "final_sp": results["final_sp"],
                }
            )
    visualize_lambda_decoupled_ablation(rows, os.path.join(config.save_dir, "lambda_decoupled_ablation.png"))
    return rows


def build_summary_lines(config: TVRunConfig, main_results: dict, best_tv: np.ndarray, mean_tv: np.ndarray) -> list[str]:
    return [
        "TV-MORL MovieLens-1M Validation",
        "=" * 56,
        f"Rt strengths: rel={config.rt_rel_strength}, div={config.rt_div_strength}, nov={config.rt_nov_strength}",
        (
            f"Rel formula: gate=[{config.time_gate_min}, {config.time_gate_max}], "
            f"static_gate={config.rel_static_gate}, pop={config.rel_pop_weight}, "
            f"explicit={config.rel_explicit_weight}"
        ),
        f"TV-MORL best: {best_tv}",
        f"TV-MORL mean: {mean_tv}",
        f"Init HV: {main_results['init_hv']:.6f} -> Final HV: {main_results['final_hv']:.6f}",
        f"Init EU: {main_results['init_eu']:.6f} -> Final EU: {main_results['final_eu']:.6f}",
        f"Init SP: {main_results['init_sp']:.6f} -> Final SP: {main_results['final_sp']:.6f}",
        f"Elapsed: {main_results['elapsed']:.2f}s",
    ]


def append_ablation_summary(summary_lines: list[str], title: str, rows: list[dict]) -> None:
    if not rows:
        return
    summary_lines.extend(["", title])
    for row in rows:
        summary_lines.append(
            f"  {row['label']}: HV={row['final_hv']:.6f}, "
            f"EU={row['final_eu']:.6f}, SP={row['final_sp']:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TV-MORL on MovieLens-1M")
    parser.add_argument("--data-dir", type=str, default=r"C:\Users\86198\ml-1m")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--init-steps", type=int, default=2000)
    parser.add_argument("--n-items", type=int, default=0)
    parser.add_argument("--min-user-ratings", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rt-rel-strength", type=float, default=1.0)
    parser.add_argument("--rt-div-strength", type=float, default=1.0)
    parser.add_argument("--rt-nov-strength", type=float, default=1.0)
    parser.add_argument("--time-gate-min", type=float, default=0.15)
    parser.add_argument("--time-gate-max", type=float, default=0.65)
    parser.add_argument("--seasonal-amp", type=float, default=0.0)
    parser.add_argument("--rel-static-gate", type=float, default=0.35)
    parser.add_argument("--rel-pop-weight", type=float, default=0.10)
    parser.add_argument("--rel-explicit-weight", type=float, default=0.70)
    parser.add_argument("--div-scale-min", type=float, default=0.60)
    parser.add_argument("--nov-scale-min", type=float, default=0.50)
    parser.add_argument("--nov-repeat-penalty-max", type=float, default=0.20)
    parser.add_argument("--session-keep-rate", type=float, default=0.85)
    parser.add_argument("--session-update-rate", type=float, default=0.15)
    parser.add_argument("--run-lambda-decoupled-ablation", action="store_true")
    parser.add_argument("--run-same-env-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    config = build_config_from_cli(cli_args)

    os.makedirs(config.save_dir, exist_ok=True)
    bundle = load_movielens_bundle(config.data_dir, max_items=config.n_items, min_user_ratings=config.min_user_ratings)
    visualize_preference_drift(bundle, os.path.join(config.save_dir, "preference_drift.png"))
    print_run_header(bundle, config)

    _, main_results = run_one_setting(config, config.save_dir)
    obj_array = main_results["obj_array"]
    best_tv = obj_array.max(axis=0)
    mean_tv = obj_array.mean(axis=0)

    lambda_rows = run_lambda_decoupled_ablation(config) if cli_args.run_lambda_decoupled_ablation else []
    same_env_rows = run_same_environment_baseline(config) if cli_args.run_same_env_baseline else []

    summary_lines = build_summary_lines(config, main_results, best_tv, mean_tv)
    append_ablation_summary(summary_lines, "Lambda decoupled ablation:", lambda_rows)
    append_ablation_summary(summary_lines, "Same-environment baseline comparison:", same_env_rows)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(os.path.join(config.save_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary_text + "\n")


if __name__ == "__main__":
    main()
