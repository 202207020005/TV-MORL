import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, WORKSPACE_ROOT)
sys.path.insert(0, os.path.join(ROOT, "morl"))
sys.path.insert(0, os.path.join(ROOT, "pytorch-a2c-ppo-acktr-gail"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))

import numpy as np
import torch

from movielens_static_env import load_movielens_static_bundle, MovieLensStaticEnv
from scalarization_methods import WeightedSumScalarization
from utils import generate_weights_batch_dfs, compute_eu, compute_sparsity, generate_w_batch_test
from pymoo.indicators.hv import Hypervolume

from tv_morl_ml1m.tv_model import MOPolicy, RolloutBuffer, TVPPO


class Args:
    def __init__(self):
        self.data_dir = r"C:\Users\86198\ml-1m"
        self.obj_num = 3
        self.min_user_ratings = 20
        self.max_episode_steps = 40
        self.seed = 42

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
        self.rl_eval_interval = 3
        self.save_dir = os.path.join(os.path.dirname(ROOT), "results", "cmorl_movielens_real")


class SimpleVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.num_envs = len(self.envs)

    def reset(self):
        obs = []
        for env in self.envs:
            ob, _ = env.reset()
            obs.append(ob)
        return np.stack(obs)

    def step(self, actions):
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, terminated, truncated, info = env.step(int(actions[i]))
            done = terminated or truncated
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)
        return np.stack(obs_list), np.array(rew_list), np.array(done_list), info_list

    def close(self):
        for env in self.envs:
            env.close()


def compute_metrics(obj_batch, ref, obj_num, eval_delta):
    if len(obj_batch) < 2:
        return 0.0, 0.0, 0.0
    hv = Hypervolume(ref_point=-np.array(ref)).do(-obj_batch)
    prefs = generate_w_batch_test(obj_num, eval_delta)
    eu = compute_eu(obj_batch, prefs)
    sp = compute_sparsity(obj_batch)
    return float(hv), float(eu), float(sp)


def check_dominated(obj_batch, obj):
    return (np.logical_and((obj_batch >= obj).all(axis=1), (obj_batch > obj).any(axis=1))).any()


def get_pareto_indices(obj_batch, ref_point):
    sorted_idx = np.argsort(obj_batch[:, 0])
    ep_idx = []
    for idx in sorted_idx:
        if (obj_batch[idx] >= ref_point).all() and not check_dominated(obj_batch, obj_batch[idx]):
            ep_idx.append(idx)
    return np.array(ep_idx, dtype=np.int64)


class EP:
    def __init__(self, ref_point, num_select, buffer_size):
        self.ref_point = np.array(ref_point, dtype=np.float32)
        self.num_select = num_select
        self.buffer_size = buffer_size
        self.obj_batch = np.array([])
        self.policies = []
        self.selected = []

    def _crowding_distance(self, obj_batch):
        n = obj_batch.shape[0]
        if n <= 2:
            return np.full(n, np.inf)
        distances = np.zeros(n)
        for dim in range(obj_batch.shape[1]):
            vals = obj_batch[:, dim]
            order = np.argsort(vals)
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            denom = vals[order[-1]] - vals[order[0]]
            if denom > 1e-10:
                for j in range(1, n - 1):
                    distances[order[j]] += (vals[order[j + 1]] - vals[order[j - 1]]) / denom
        return distances

    def update(self, policies, objs):
        if len(self.obj_batch) == 0:
            self.obj_batch = np.array(objs)
            self.policies = list(policies)
        else:
            self.obj_batch = np.vstack([self.obj_batch, objs])
            self.policies.extend(policies)
        keep = get_pareto_indices(self.obj_batch, self.ref_point)
        self.obj_batch = self.obj_batch[keep]
        self.policies = [self.policies[i] for i in keep]
        if len(self.obj_batch) > self.buffer_size:
            keep2 = np.argsort(-self._crowding_distance(self.obj_batch))[: self.buffer_size]
            self.obj_batch = self.obj_batch[keep2]
            self.policies = [self.policies[i] for i in keep2]
        order = np.argsort(-self._crowding_distance(self.obj_batch))[: self.num_select]
        self.selected = [self.policies[i] for i in order]


def main():
    args = Args()
    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bundle = load_movielens_static_bundle(args.data_dir, min_user_ratings=args.min_user_ratings)
    sample_env = MovieLensStaticEnv(bundle, max_steps=args.max_episode_steps, seed=args.seed)
    obs_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.n
    sample_env.close()

    def make_env(seed):
        return MovieLensStaticEnv(bundle, max_steps=args.max_episode_steps, seed=seed)

    def evaluate(policy):
        env = make_env(args.seed + 999)
        objs = np.zeros(args.obj_num, dtype=np.float32)
        with torch.no_grad():
            for eval_id in range(args.eval_num):
                obs, _ = env.reset(seed=args.seed + eval_id)
                gamma = 1.0
                done = False
                step = 0
                while not done and step < args.max_episode_steps:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    _, action, _, _ = policy.act(obs_t, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action.item())
                    done = terminated or truncated
                    objs += gamma * info["obj"]
                    gamma *= args.eval_gamma
                    step += 1
        env.close()
        return objs / args.eval_num

    weights_batch = []
    generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
    weights_batch = [w for w in weights_batch if np.isclose(np.sum(w), 1.0)]

    ep = EP(args.ref_point, args.num_select, args.policy_buffer)
    init_policies = []
    init_objs = []
    num_update = max(1, int(args.num_init_steps // max(1, len(weights_batch)) // args.num_steps))

    for task_id, weights in enumerate(weights_batch):
        policy = MOPolicy(obs_dim, action_dim, args.obj_num, is_discrete=True, hidden_size=64)
        agent = TVPPO(policy, clip_param=args.clip_param, ppo_epoch=args.ppo_epoch, num_mini_batch=args.num_mini_batch, value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm, beta=args.beta, t=args.t)
        envs = SimpleVecEnv([lambda s=args.seed + task_id * 100 + i: make_env(s) for i in range(args.num_processes)])
        rollouts = RolloutBuffer(args.num_steps, args.num_processes, (obs_dim,), args.obj_num)
        obs = torch.from_numpy(envs.reset()).float()
        rollouts.obs[0].copy_(obs)
        scalar = WeightedSumScalarization(args.obj_num, weights)

        for update in range(num_update):
            for step in range(args.num_steps):
                with torch.no_grad():
                    values, action, action_log_prob, _ = policy.act(rollouts.obs[step])
                obs_np, _, dones, infos = envs.step(action.cpu().numpy())
                obs_t = torch.from_numpy(obs_np).float()
                obj_tensor = torch.zeros(args.num_processes, args.obj_num)
                for idx, info in enumerate(infos):
                    obj_tensor[idx] = torch.from_numpy(info["obj"])
                masks = torch.FloatTensor([[0.0] if d else [1.0] for d in dones])
                bad_masks = torch.ones_like(masks)
                rollouts.insert(obs_t, action, action_log_prob.squeeze(-1) if action_log_prob.dim() > 1 else action_log_prob, values, obj_tensor, masks, bad_masks)

            with torch.no_grad():
                next_value = policy.get_value(rollouts.obs[-1])
            rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)
            agent.update(rollouts, scalar)
            rollouts.after_update()

            if (update + 1) % args.rl_eval_interval == 0 or update == num_update - 1:
                init_policies.append(policy)
                init_objs.append(evaluate(policy))
        envs.close()

    ep.update(init_policies, np.array(init_objs))
    init_hv, init_eu, init_sp = compute_metrics(ep.obj_batch, args.ref_point, args.obj_num, args.eval_delta_weight)

    total_updates = int((args.num_time_steps - args.num_init_steps) // max(1, len(ep.selected)) // args.num_steps // args.obj_num)
    ext_policies = []
    ext_objs = []
    for base_policy in ep.selected:
        for obj_idx in range(args.obj_num):
            policy = base_policy
            agent = TVPPO(policy, clip_param=args.clip_param, ppo_epoch=args.ppo_epoch, num_mini_batch=args.num_mini_batch, value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm, beta=args.beta, t=args.t)
            envs = SimpleVecEnv([lambda s=args.seed + 5000 + i: make_env(s) for i in range(args.num_processes)])
            rollouts = RolloutBuffer(args.num_steps, args.num_processes, (obs_dim,), args.obj_num)
            obs = torch.from_numpy(envs.reset()).float()
            rollouts.obs[0].copy_(obs)
            for _ in range(max(1, total_updates)):
                for step in range(args.num_steps):
                    with torch.no_grad():
                        values, action, action_log_prob, _ = policy.act(rollouts.obs[step])
                    obs_np, _, dones, infos = envs.step(action.cpu().numpy())
                    obs_t = torch.from_numpy(obs_np).float()
                    obj_tensor = torch.zeros(args.num_processes, args.obj_num)
                    for idx, info in enumerate(infos):
                        obj_tensor[idx] = torch.from_numpy(info["obj"])
                    masks = torch.FloatTensor([[0.0] if d else [1.0] for d in dones])
                    bad_masks = torch.ones_like(masks)
                    rollouts.insert(obs_t, action, action_log_prob.squeeze(-1) if action_log_prob.dim() > 1 else action_log_prob, values, obj_tensor, masks, bad_masks)
                with torch.no_grad():
                    next_value = policy.get_value(rollouts.obs[-1])
                rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)
                agent.ipo_update(rollouts, obj_idx, args.obj_num)
                rollouts.after_update()
            ext_policies.append(policy)
            ext_objs.append(evaluate(policy))
            envs.close()

    ep.update(ext_policies, np.array(ext_objs))
    final_hv, final_eu, final_sp = compute_metrics(ep.obj_batch, args.ref_point, args.obj_num, args.eval_delta_weight)

    summary = [
        "Original-style C-MORL on MovieLens static environment",
        "=" * 56,
        f"Users: {len(bundle.user_ids)}",
        f"Candidate items: {len(bundle.movie_ids)}",
        f"Init HV: {init_hv:.6f} -> Final HV: {final_hv:.6f}",
        f"Init EU: {init_eu:.6f} -> Final EU: {final_eu:.6f}",
        f"Init SP: {init_sp:.6f} -> Final SP: {final_sp:.6f}",
        f"Pareto points: {len(ep.obj_batch)}",
    ]
    with open(os.path.join(args.save_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    np.save(os.path.join(args.save_dir, "final_obj_array.npy"), ep.obj_batch)
    print("\n".join(summary))


if __name__ == "__main__":
    main()
