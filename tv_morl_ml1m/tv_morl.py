import itertools
import time
from copy import deepcopy

import numpy as np
import torch
from pymoo.indicators.hv import Hypervolume

from tv_morl_ml1m.movielens_env import MovieLensRealTVEnv, load_movielens_bundle
from tv_morl_ml1m.tv_model import MOPolicy, RolloutBuffer, TVPPO


class SimpleVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.stack(obs_list)

    def step(self, actions):
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for idx, env in enumerate(self.envs):
            obs, rew, terminated, truncated, info = env.step(int(actions[idx]))
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(terminated or truncated)
            info_list.append(info)
            if terminated or truncated:
                obs, _ = env.reset()
                obs_list[-1] = obs
        return np.stack(obs_list), np.array(rew_list), np.array(done_list), info_list

    def close(self):
        for env in self.envs:
            env.close()


def print_info(*message):
    print("\033[96m", *message, "\033[0m")


def generate_weights_batch_dfs(i, obj_num, min_w, max_w, delta_w, weight, batch):
    if i == obj_num - 1:
        weight.append(1.0 - np.sum(weight[:i]))
        if min_w - 1e-6 <= weight[-1] <= max_w + 1e-6:
            batch.append(deepcopy(weight))
        return
    w = min_w
    while w < max_w + 0.5 * delta_w and np.sum(weight[:i]) + w < 1.0 + 0.5 * delta_w:
        weight.append(w)
        generate_weights_batch_dfs(i + 1, obj_num, min_w, max_w, delta_w, weight, batch)
        weight = weight[:i]
        w += delta_w


def generate_w_batch_test(obj_num, step):
    mesh = [np.arange(0, 1 + step, step) for _ in range(obj_num)]
    w = np.array(list(itertools.product(*mesh)))
    w = w[np.isclose(w.sum(axis=1), 1.0)]
    return np.unique(w, axis=0)


def compute_eu(obj_batch, prefs):
    return float(np.mean([max(np.dot(p, o) for o in obj_batch) for p in prefs]))


def compute_sparsity(obj_batch):
    if len(obj_batch) < 2:
        return 0.0
    sp = 0.0
    for dim in range(obj_batch.shape[1]):
        objs = np.sort(obj_batch[:, dim])
        for idx in range(1, len(objs)):
            sp += np.square(objs[idx] - objs[idx - 1])
    return float(sp / max(1, len(obj_batch) - 1))


def compute_metrics(obj_batch, ref, obj_num, eval_delta):
    if len(obj_batch) < 2:
        return 0.0, 0.0, 0.0
    hv = Hypervolume(ref_point=-np.array(ref)).do(-obj_batch)
    prefs = generate_w_batch_test(obj_num, eval_delta)
    return float(hv), compute_eu(obj_batch, prefs), compute_sparsity(obj_batch)


def check_dominated(obj_batch, obj):
    return (np.logical_and((obj_batch >= obj).all(axis=1), (obj_batch > obj).any(axis=1))).any()


def get_pareto_indices(obj_batch, ref_point):
    if len(obj_batch) == 0:
        return np.array([], dtype=np.int64)
    sorted_idx = np.argsort(obj_batch[:, 0])
    ep_idx = []
    for idx in sorted_idx:
        if (obj_batch[idx] >= ref_point).all() and not check_dominated(obj_batch, obj_batch[idx]):
            ep_idx.append(idx)
    return np.array(ep_idx, dtype=np.int64)


class TVEP:
    def __init__(self, ref_point, num_select, buffer_size):
        self.ref_point = np.array(ref_point, dtype=np.float32)
        self.num_select = num_select
        self.buffer_size = buffer_size
        self.obj_batch = np.array([])
        self.policy_batch = []
        self.selected_policies = []
        self.selected_objs = np.array([])

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
                for idx in range(1, n - 1):
                    distances[order[idx]] += (vals[order[idx + 1]] - vals[order[idx - 1]]) / denom
        return distances

    def update(self, new_policies, new_objs):
        if len(new_objs) == 0:
            return
        if len(self.obj_batch) == 0:
            self.obj_batch = np.array(new_objs)
            self.policy_batch = list(new_policies)
        else:
            self.obj_batch = np.vstack([self.obj_batch, new_objs])
            self.policy_batch.extend(new_policies)

        pareto_idx = get_pareto_indices(self.obj_batch, self.ref_point)
        if len(pareto_idx) > 0:
            self.obj_batch = self.obj_batch[pareto_idx]
            self.policy_batch = [self.policy_batch[idx] for idx in pareto_idx]

        if len(self.obj_batch) > self.buffer_size:
            keep = np.argsort(-self._crowding_distance(self.obj_batch))[: self.buffer_size]
            self.obj_batch = self.obj_batch[keep]
            self.policy_batch = [self.policy_batch[idx] for idx in keep]

        if len(self.obj_batch) <= self.num_select:
            self.selected_policies = list(self.policy_batch)
            self.selected_objs = self.obj_batch.copy()
            return

        order = np.argsort(-self._crowding_distance(self.obj_batch))
        order = order[: self.num_select]
        self.selected_policies = [self.policy_batch[idx] for idx in order]
        self.selected_objs = self.obj_batch[order]


class TVMORLTrainer:
    def __init__(self, args):
        self.args = args
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.bundle = load_movielens_bundle(
            args.data_dir,
            max_items=args.n_items,
            min_user_ratings=args.min_user_ratings,
        )
        self.ref = np.array(args.ref_point, dtype=np.float32)
        self.ep = TVEP(self.ref, args.num_select, args.policy_buffer)

        self.weights_batch = []
        generate_weights_batch_dfs(
            0,
            args.obj_num,
            args.min_weight,
            args.max_weight,
            args.delta_weight,
            [],
            self.weights_batch,
        )
        self.weights_batch = [w for w in self.weights_batch if np.isclose(np.sum(w), 1.0)]

        sample_env = MovieLensRealTVEnv(
            self.bundle,
            max_steps=args.max_episode_steps,
            seed=args.seed,
            enable_pt=args.enable_pt,
            enable_rt=args.enable_rt,
            rt_rel_strength=args.rt_rel_strength,
            rt_div_strength=args.rt_div_strength,
            rt_nov_strength=args.rt_nov_strength,
            time_gate_min=args.time_gate_min,
            time_gate_max=args.time_gate_max,
            seasonal_amp=args.seasonal_amp,
            rel_static_gate=args.rel_static_gate,
            rel_pop_weight=args.rel_pop_weight,
            rel_explicit_weight=args.rel_explicit_weight,
            div_scale_min=args.div_scale_min,
            nov_scale_min=args.nov_scale_min,
            nov_repeat_penalty_max=args.nov_repeat_penalty_max,
            session_keep_rate=args.session_keep_rate,
            session_update_rate=args.session_update_rate,
        )
        self.obs_dim = sample_env.observation_space.shape[0]
        self.action_dim = sample_env.action_space.n
        sample_env.close()

    def _create_policy(self):
        policy = MOPolicy(self.obs_dim, self.action_dim, self.args.obj_num, is_discrete=True, hidden_size=128)
        agent = TVPPO(
            policy,
            clip_param=self.args.clip_param,
            ppo_epoch=self.args.ppo_epoch,
            num_mini_batch=self.args.num_mini_batch,
            value_loss_coef=self.args.value_loss_coef,
            entropy_coef=self.args.entropy_coef,
            lr=self.args.lr,
            max_grad_norm=self.args.max_grad_norm,
            beta=self.args.beta,
            t=self.args.t,
        )
        return policy, agent

    def _create_envs(self, seed_offset=0):
        env_fns = []
        for idx in range(self.args.num_processes):
            seed = self.args.seed + seed_offset + idx * 97
            env_fns.append(
                lambda s=seed: MovieLensRealTVEnv(
                    self.bundle,
                    max_steps=self.args.max_episode_steps,
                    seed=s,
                    enable_pt=self.args.enable_pt,
                    enable_rt=self.args.enable_rt,
                    rt_rel_strength=self.args.rt_rel_strength,
                    rt_div_strength=self.args.rt_div_strength,
                    rt_nov_strength=self.args.rt_nov_strength,
                    time_gate_min=self.args.time_gate_min,
                    time_gate_max=self.args.time_gate_max,
                    seasonal_amp=self.args.seasonal_amp,
                    rel_static_gate=self.args.rel_static_gate,
                    rel_pop_weight=self.args.rel_pop_weight,
                    rel_explicit_weight=self.args.rel_explicit_weight,
                    div_scale_min=self.args.div_scale_min,
                    nov_scale_min=self.args.nov_scale_min,
                    nov_repeat_penalty_max=self.args.nov_repeat_penalty_max,
                    session_keep_rate=self.args.session_keep_rate,
                    session_update_rate=self.args.session_update_rate,
                )
            )
        return SimpleVecEnv(env_fns)

    def _evaluate(self, policy):
        env = MovieLensRealTVEnv(
            self.bundle,
            max_steps=self.args.max_episode_steps,
            seed=self.args.seed + 999,
            enable_pt=self.args.enable_pt,
            enable_rt=self.args.enable_rt,
            rt_rel_strength=self.args.rt_rel_strength,
            rt_div_strength=self.args.rt_div_strength,
            rt_nov_strength=self.args.rt_nov_strength,
            time_gate_min=self.args.time_gate_min,
            time_gate_max=self.args.time_gate_max,
            seasonal_amp=self.args.seasonal_amp,
            rel_static_gate=self.args.rel_static_gate,
            rel_pop_weight=self.args.rel_pop_weight,
            rel_explicit_weight=self.args.rel_explicit_weight,
            div_scale_min=self.args.div_scale_min,
            nov_scale_min=self.args.nov_scale_min,
            nov_repeat_penalty_max=self.args.nov_repeat_penalty_max,
            session_keep_rate=self.args.session_keep_rate,
            session_update_rate=self.args.session_update_rate,
        )
        objs = np.zeros(self.args.obj_num, dtype=np.float32)
        with torch.no_grad():
            for eval_id in range(self.args.eval_num):
                obs, _ = env.reset(seed=self.args.seed + eval_id + 5000)
                done = False
                gamma = 1.0
                step = 0
                while not done and step < self.args.max_episode_steps:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    _, action, _, _ = policy.act(obs_tensor, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action.item())
                    done = terminated or truncated
                    objs += gamma * info["obj"]
                    gamma *= self.args.eval_gamma
                    step += 1
        env.close()
        return objs / self.args.eval_num

    def train_init_phase(self):
        print_info("\n" + "=" * 60)
        print_info("TV-MORL Phase 1: Initialization")
        print_info("=" * 60)

        num_update = max(1, int(self.args.num_init_steps // max(1, len(self.weights_batch)) // self.args.num_steps))
        all_policies = []
        all_objs = []

        for task_id, weights in enumerate(self.weights_batch):
            policy, agent = self._create_policy()
            envs = self._create_envs(seed_offset=task_id * 1000)
            rollouts = RolloutBuffer(self.args.num_steps, self.args.num_processes, (self.obs_dim,), self.args.obj_num)
            obs = torch.from_numpy(envs.reset()).float()
            rollouts.obs[0].copy_(obs)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)

            for update in range(num_update):
                for step in range(self.args.num_steps):
                    with torch.no_grad():
                        values, action, action_log_prob, _ = policy.act(rollouts.obs[step])
                    obs_np, _, dones, infos = envs.step(action.cpu().numpy())
                    obs_t = torch.from_numpy(obs_np).float()
                    obj_tensor = torch.zeros(self.args.num_processes, self.args.obj_num, dtype=torch.float32)
                    for idx, info in enumerate(infos):
                        obj_tensor[idx] = torch.from_numpy(info["obj"])
                    masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones])
                    bad_masks = torch.ones_like(masks)
                    rollouts.insert(obs_t, action, action_log_prob, values, obj_tensor, masks, bad_masks)

                with torch.no_grad():
                    next_value = policy.get_value(rollouts.obs[-1])
                rollouts.compute_returns(next_value, self.args.gamma, self.args.gae_lambda)

                class Scalarization:
                    def __init__(self, w):
                        self.weights = w

                agent.update(rollouts, Scalarization(weights_tensor))
                rollouts.after_update()

                if (update + 1) % self.args.rl_eval_interval == 0 or update == num_update - 1:
                    objs = self._evaluate(policy)
                    all_policies.append(deepcopy(policy))
                    all_objs.append(objs)

            envs.close()

        self.ep.update(all_policies, np.array(all_objs))
        hv, eu, sp = compute_metrics(self.ep.obj_batch, self.ref, self.args.obj_num, self.args.eval_delta_weight)
        print_info(f"Init complete |EP|={len(self.ep.obj_batch)} HV={hv:.4f} EU={eu:.4f} SP={sp:.4f}")
        return hv, eu, sp

    def train_extension_phase(self):
        print_info("\n" + "=" * 60)
        print_info("TV-MORL Phase 2: IPO Extension")
        print_info("=" * 60)

        selected = self.ep.selected_policies
        total_updates = int(
            (self.args.num_time_steps - self.args.num_init_steps)
            // max(1, len(selected))
            // self.args.num_steps
            // self.args.obj_num
        )
        n_iters = max(1, total_updates // max(1, self.args.update_iter))

        for iter_idx in range(n_iters):
            num_update = min(self.args.update_iter, total_updates - iter_idx * self.args.update_iter)
            if num_update <= 0:
                break
            all_new_policies = []
            all_new_objs = []

            for sample_id, base_policy in enumerate(selected):
                for obj_dim in range(self.args.obj_num):
                    policy = deepcopy(base_policy)
                    agent = TVPPO(
                        policy,
                        clip_param=self.args.clip_param,
                        ppo_epoch=self.args.ppo_epoch,
                        num_mini_batch=self.args.num_mini_batch,
                        value_loss_coef=self.args.value_loss_coef,
                        entropy_coef=self.args.entropy_coef,
                        lr=self.args.lr,
                        max_grad_norm=self.args.max_grad_norm,
                        beta=self.args.beta,
                        t=self.args.t,
                    )
                    envs = self._create_envs(seed_offset=50000 + sample_id * 1000 + obj_dim * 100)
                    rollouts = RolloutBuffer(self.args.num_steps, self.args.num_processes, (self.obs_dim,), self.args.obj_num)
                    obs = torch.from_numpy(envs.reset()).float()
                    rollouts.obs[0].copy_(obs)

                    for _ in range(num_update):
                        for step in range(self.args.num_steps):
                            with torch.no_grad():
                                values, action, action_log_prob, _ = policy.act(rollouts.obs[step])
                            obs_np, _, dones, infos = envs.step(action.cpu().numpy())
                            obs_t = torch.from_numpy(obs_np).float()
                            obj_tensor = torch.zeros(self.args.num_processes, self.args.obj_num, dtype=torch.float32)
                            for idx, info in enumerate(infos):
                                obj_tensor[idx] = torch.from_numpy(info["obj"])
                            masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones])
                            bad_masks = torch.ones_like(masks)
                            rollouts.insert(obs_t, action, action_log_prob, values, obj_tensor, masks, bad_masks)

                        with torch.no_grad():
                            next_value = policy.get_value(rollouts.obs[-1])
                        rollouts.compute_returns(next_value, self.args.gamma, self.args.gae_lambda)
                        agent.ipo_update(rollouts, obj_dim, self.args.obj_num)
                        rollouts.after_update()

                    all_new_policies.append(deepcopy(policy))
                    all_new_objs.append(self._evaluate(policy))
                    envs.close()

            self.ep.update(all_new_policies, np.array(all_new_objs))
            selected = self.ep.selected_policies
            hv, eu, sp = compute_metrics(self.ep.obj_batch, self.ref, self.args.obj_num, self.args.eval_delta_weight)
            print_info(f"Iter {iter_idx + 1}/{n_iters} |EP|={len(self.ep.obj_batch)} HV={hv:.4f} EU={eu:.4f} SP={sp:.4f}")
        return self.ep

    def run(self):
        start = time.time()
        init_hv, init_eu, init_sp = self.train_init_phase()
        self.train_extension_phase()
        final_hv, final_eu, final_sp = compute_metrics(
            self.ep.obj_batch,
            self.ref,
            self.args.obj_num,
            self.args.eval_delta_weight,
        )
        return {
            "ep": self.ep,
            "obj_array": self.ep.obj_batch,
            "init_hv": init_hv,
            "final_hv": final_hv,
            "init_eu": init_eu,
            "final_eu": final_eu,
            "init_sp": init_sp,
            "final_sp": final_sp,
            "elapsed": time.time() - start,
        }
