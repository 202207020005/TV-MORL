import argparse
import os
import sys

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


class Args:
    def __init__(self):
        self.data_dir = r"C:\Users\86198\ml-1m"
        self.obj_num = 3
        self.n_items = None
        self.min_user_ratings = 20
        self.max_episode_steps = 40

        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.num_steps = 32
        self.num_processes = 2
        self.ppo_epoch = 3
        self.num_mini_batch = 4
        self.clip_param = 0.2

        self.num_time_steps = 4000
        self.num_init_steps = 2000
        self.min_weight = 0.0
        self.max_weight = 1.0
        self.delta_weight = 0.5
        self.eval_delta_weight = 0.1
        self.update_iter = 4
        self.eval_num = 3
        self.eval_gamma = 0.99
        self.num_select = 3
        self.policy_buffer = 30
        self.ref_point = [0.0, 0.0, 0.0]

        self.beta = 0.85
        self.t = 12
        self.seed = 42
        self.rl_eval_interval = 3

        self.enable_pt = True
        self.enable_rt = True
        self.rt_rel_strength = 1.0
        self.rt_div_strength = 1.0
        self.rt_nov_strength = 1.0
        self.time_gate_min = 0.15
        self.time_gate_max = 0.65
        self.seasonal_amp = 0.04
        self.rel_static_gate = 0.35
        self.rel_pop_weight = 0.10
        self.rel_explicit_weight = 0.70
        self.div_scale_min = 0.60
        self.nov_scale_min = 0.50
        self.nov_repeat_penalty_max = 0.20
        self.session_keep_rate = 0.85
        self.session_update_rate = 0.15

        self.save_dir = os.path.join(ROOT_DIR, "results", "tv_morl_ml1m")


def visualize_preference_drift(bundle, save_path):
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


def visualize_pareto_front(obj_array, save_path):
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


def visualize_rt_param_ablation(rows, save_path):
    labels = [row["label"] for row in rows]
    hv = [row["final_hv"] for row in rows]
    eu = [row["final_eu"] for row in rows]
    sp = [row["final_sp"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, values, title, color in zip(
        axes,
        [hv, eu, sp],
        ["Hypervolume", "Expected Utility", "Sparsity"],
        ["#d35400", "#2980b9", "#27ae60"],
    ):
        ax.bar(range(len(labels)), values, color=color, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_rel_formula_ablation(rows, save_path):
    labels = [row["label"] for row in rows]
    hv = [row["final_hv"] for row in rows]
    eu = [row["final_eu"] for row in rows]
    sp = [row["final_sp"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, values, title, color in zip(
        axes,
        [hv, eu, sp],
        ["Hypervolume", "Expected Utility", "Sparsity"],
        ["#8e44ad", "#16a085", "#c0392b"],
    ):
        ax.bar(range(len(labels)), values, color=color, alpha=0.82)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_lambda_decoupled_ablation(rows, save_path):
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


def run_one_setting(base_args, save_dir):
    args = Args()
    args.__dict__.update(base_args.__dict__)
    args.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    trainer = TVMORLTrainer(args)
    results = trainer.run()
    obj_array = results["obj_array"]
    np.save(os.path.join(save_dir, "final_obj_array.npy"), obj_array)
    visualize_pareto_front(obj_array, os.path.join(save_dir, "pareto_front.png"))
    return trainer.bundle, results


def main():
    parser = argparse.ArgumentParser(description="TV-MORL on MovieLens-1M")
    parser.add_argument("--data-dir", type=str, default=r"C:\Users\86198\ml-1m")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--init-steps", type=int, default=2000)
    parser.add_argument("--n-items", type=int, default=0)
    parser.add_argument("--min-user-ratings", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rt-rel-strength", type=float, default=1.0)
    parser.add_argument("--rt-click-strength", type=float, default=None)
    parser.add_argument("--rt-div-strength", type=float, default=1.0)
    parser.add_argument("--rt-nov-strength", type=float, default=1.0)
    parser.add_argument("--time-gate-min", type=float, default=0.15)
    parser.add_argument("--time-gate-max", type=float, default=0.65)
    parser.add_argument("--seasonal-amp", type=float, default=0.04)
    parser.add_argument("--rel-static-gate", type=float, default=0.35)
    parser.add_argument("--rel-pop-weight", type=float, default=0.10)
    parser.add_argument("--rel-explicit-weight", type=float, default=0.70)
    parser.add_argument("--div-scale-min", type=float, default=0.60)
    parser.add_argument("--nov-scale-min", type=float, default=0.50)
    parser.add_argument("--nov-repeat-penalty-max", type=float, default=0.20)
    parser.add_argument("--session-keep-rate", type=float, default=0.85)
    parser.add_argument("--session-update-rate", type=float, default=0.15)
    parser.add_argument("--run-rt-param-ablation", action="store_true")
    parser.add_argument("--run-rel-formula-ablation", action="store_true")
    parser.add_argument("--run-lambda-decoupled-ablation", action="store_true")
    cli_args = parser.parse_args()

    args = Args()
    args.data_dir = cli_args.data_dir
    args.num_time_steps = cli_args.num_steps
    args.num_init_steps = cli_args.init_steps
    args.n_items = None if cli_args.n_items <= 0 else cli_args.n_items
    args.min_user_ratings = cli_args.min_user_ratings
    args.seed = cli_args.seed
    args.rt_rel_strength = cli_args.rt_rel_strength if cli_args.rt_click_strength is None else cli_args.rt_click_strength
    args.rt_div_strength = cli_args.rt_div_strength
    args.rt_nov_strength = cli_args.rt_nov_strength
    args.time_gate_min = cli_args.time_gate_min
    args.time_gate_max = cli_args.time_gate_max
    args.seasonal_amp = cli_args.seasonal_amp
    args.rel_static_gate = cli_args.rel_static_gate
    args.rel_pop_weight = cli_args.rel_pop_weight
    args.rel_explicit_weight = cli_args.rel_explicit_weight
    args.div_scale_min = cli_args.div_scale_min
    args.nov_scale_min = cli_args.nov_scale_min
    args.nov_repeat_penalty_max = cli_args.nov_repeat_penalty_max
    args.session_keep_rate = cli_args.session_keep_rate
    args.session_update_rate = cli_args.session_update_rate
    if cli_args.save_dir is not None:
        args.save_dir = cli_args.save_dir

    os.makedirs(args.save_dir, exist_ok=True)
    bundle = load_movielens_bundle(args.data_dir, max_items=args.n_items, min_user_ratings=args.min_user_ratings)
    visualize_preference_drift(bundle, os.path.join(args.save_dir, "preference_drift.png"))

    print("=" * 72)
    print("TV-MORL on MovieLens-1M")
    print("=" * 72)
    print(f"raw dataset users=6040 movies_file=3883 rated_movies=3706 ratings=1000209")
    print(f"experiment subset users={len(bundle.user_ids)} candidate_items={len(bundle.movie_ids)} min_user_ratings={args.min_user_ratings}")
    print(
        f"Rt strengths: rel={args.rt_rel_strength}, div={args.rt_div_strength}, nov={args.rt_nov_strength}"
    )
    print(
        f"rel formula: gate=[{args.time_gate_min}, {args.time_gate_max}], static_gate={args.rel_static_gate}, "
        f"pop={args.rel_pop_weight}, explicit={args.rel_explicit_weight}"
    )
    print("=" * 72)

    _, main_results = run_one_setting(args, args.save_dir)
    obj_array = main_results["obj_array"]
    best_tv = obj_array.max(axis=0)
    mean_tv = obj_array.mean(axis=0)

    ablation_rows = []
    if cli_args.run_rt_param_ablation:
        param_grid = [
            (0.0, 0.0, 0.0, "Rt=0"),
            (0.25, 0.25, 0.25, "Rt=0.25"),
            (0.5, 0.5, 0.5, "Rt=0.5"),
            (0.75, 0.75, 0.75, "Rt=0.75"),
            (1.0, 1.0, 1.0, "Rt=1.0"),
        ]
        for c, d, n, label in param_grid:
            sub_args = Args()
            sub_args.__dict__.update(args.__dict__)
            sub_args.rt_rel_strength = c
            sub_args.rt_div_strength = d
            sub_args.rt_nov_strength = n
            run_dir = os.path.join(args.save_dir, "rt_param_ablation", label.replace("=", "_"))
            _, results = run_one_setting(sub_args, run_dir)
            ablation_rows.append(
                {
                    "label": label,
                    "rel": c,
                    "div": d,
                    "nov": n,
                    "final_hv": results["final_hv"],
                    "final_eu": results["final_eu"],
                    "final_sp": results["final_sp"],
                }
            )
        visualize_rt_param_ablation(ablation_rows, os.path.join(args.save_dir, "rt_param_ablation.png"))

    rel_formula_rows = []
    if cli_args.run_rel_formula_ablation:
        rel_grid = [
            ("pop_0.00", 0.00, args.rel_explicit_weight, args.rel_static_gate),
            ("pop_0.05", 0.05, args.rel_explicit_weight, args.rel_static_gate),
            ("pop_0.10", 0.10, args.rel_explicit_weight, args.rel_static_gate),
            ("pop_0.20", 0.20, args.rel_explicit_weight, args.rel_static_gate),
            ("exp_0.50", args.rel_pop_weight, 0.50, args.rel_static_gate),
            ("exp_0.70", args.rel_pop_weight, 0.70, args.rel_static_gate),
            ("exp_0.85", args.rel_pop_weight, 0.85, args.rel_static_gate),
            ("gate_0.25", args.rel_pop_weight, args.rel_explicit_weight, 0.25),
            ("gate_0.35", args.rel_pop_weight, args.rel_explicit_weight, 0.35),
            ("gate_0.45", args.rel_pop_weight, args.rel_explicit_weight, 0.45),
        ]
        for label, pop_w, explicit_w, static_gate in rel_grid:
            sub_args = Args()
            sub_args.__dict__.update(args.__dict__)
            sub_args.rel_pop_weight = pop_w
            sub_args.rel_explicit_weight = explicit_w
            sub_args.rel_static_gate = static_gate
            run_dir = os.path.join(args.save_dir, "rel_formula_ablation", label)
            _, results = run_one_setting(sub_args, run_dir)
            rel_formula_rows.append(
                {
                    "label": label,
                    "pop": pop_w,
                    "explicit": explicit_w,
                    "static_gate": static_gate,
                    "final_hv": results["final_hv"],
                    "final_eu": results["final_eu"],
                    "final_sp": results["final_sp"],
                }
            )
        visualize_rel_formula_ablation(rel_formula_rows, os.path.join(args.save_dir, "rel_formula_ablation.png"))

    lambda_rows = []
    if cli_args.run_lambda_decoupled_ablation:
        lambda_grid = [0.0, 0.05, 0.10, 0.15, 0.20]
        param_specs = [
            ("rel", "rt_rel_strength", args.rt_div_strength, args.rt_nov_strength),
            ("div", "rt_div_strength", args.rt_rel_strength, args.rt_nov_strength),
            ("nov", "rt_nov_strength", args.rt_rel_strength, args.rt_div_strength),
        ]
        for param_name, attr_name, fixed_a, fixed_b in param_specs:
            for strength in lambda_grid:
                sub_args = Args()
                sub_args.__dict__.update(args.__dict__)
                if param_name == "rel":
                    sub_args.rt_rel_strength = strength
                    sub_args.rt_div_strength = fixed_a
                    sub_args.rt_nov_strength = fixed_b
                elif param_name == "div":
                    sub_args.rt_rel_strength = fixed_a
                    sub_args.rt_div_strength = strength
                    sub_args.rt_nov_strength = fixed_b
                else:
                    sub_args.rt_rel_strength = fixed_a
                    sub_args.rt_div_strength = fixed_b
                    sub_args.rt_nov_strength = strength
                label = f"{param_name}_{strength:.2f}"
                run_dir = os.path.join(args.save_dir, "lambda_decoupled_ablation", label)
                _, results = run_one_setting(sub_args, run_dir)
                lambda_rows.append(
                    {
                        "label": label,
                        "param": param_name,
                        "strength": strength,
                        "final_hv": results["final_hv"],
                        "final_eu": results["final_eu"],
                        "final_sp": results["final_sp"],
                    }
                )
        visualize_lambda_decoupled_ablation(lambda_rows, os.path.join(args.save_dir, "lambda_decoupled_ablation.png"))

    summary_lines = [
        "TV-MORL MovieLens-1M Validation",
        "=" * 56,
        f"Rt strengths: rel={args.rt_rel_strength}, div={args.rt_div_strength}, nov={args.rt_nov_strength}",
        (
            f"Rel formula: gate=[{args.time_gate_min}, {args.time_gate_max}], static_gate={args.rel_static_gate}, "
            f"pop={args.rel_pop_weight}, explicit={args.rel_explicit_weight}"
        ),
        f"TV-MORL best: {best_tv}",
        f"TV-MORL mean: {mean_tv}",
        f"Init HV: {main_results['init_hv']:.6f} -> Final HV: {main_results['final_hv']:.6f}",
        f"Init EU: {main_results['init_eu']:.6f} -> Final EU: {main_results['final_eu']:.6f}",
        f"Init SP: {main_results['init_sp']:.6f} -> Final SP: {main_results['final_sp']:.6f}",
        f"Elapsed: {main_results['elapsed']:.2f}s",
    ]
    if ablation_rows:
        summary_lines.extend(["", "Rt parameter ablation:"])
        for row in ablation_rows:
            summary_lines.append(
                f"  {row['label']}: HV={row['final_hv']:.6f}, EU={row['final_eu']:.6f}, SP={row['final_sp']:.6f}"
            )
    if rel_formula_rows:
        summary_lines.extend(["", "Relevance formula ablation:"])
        for row in rel_formula_rows:
            summary_lines.append(
                f"  {row['label']}: HV={row['final_hv']:.6f}, EU={row['final_eu']:.6f}, SP={row['final_sp']:.6f}"
            )
    if lambda_rows:
        summary_lines.extend(["", "Lambda decoupled ablation:"])
        for row in lambda_rows:
            summary_lines.append(
                f"  {row['label']}: HV={row['final_hv']:.6f}, EU={row['final_eu']:.6f}, SP={row['final_sp']:.6f}"
            )

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(os.path.join(args.save_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary_text + "\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
