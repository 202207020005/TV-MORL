import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal


class MOPolicy(nn.Module):
    """Minimal multi-objective actor-critic used by TV-MORL."""

    def __init__(self, obs_dim, action_dim, obj_num, is_discrete=False, hidden_size=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obj_num = obj_num
        self.is_discrete = is_discrete

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, obj_num),
        )

        if is_discrete:
            self.actor_head = nn.Linear(hidden_size, action_dim)
        else:
            self.actor_mean = nn.Linear(hidden_size, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs):
        values = self.critic(obs)
        actor_features = self.actor(obs)
        if self.is_discrete:
            logits = self.actor_head(actor_features)
            return values, logits, None
        action_mean = self.actor_mean(actor_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return values, action_mean, action_logstd

    def act(self, obs, deterministic=False):
        values, action_param, action_logstd = self.forward(obs)
        if self.is_discrete:
            dist = Categorical(logits=action_param)
            action = torch.argmax(action_param, dim=-1) if deterministic else dist.sample()
            action_log_probs = dist.log_prob(action)
        else:
            dist = Normal(action_param, torch.exp(action_logstd))
            action = action_param if deterministic else dist.sample()
            action_log_probs = dist.log_prob(action).sum(dim=-1)
        return values, action, action_log_probs, None

    def get_value(self, obs):
        values, _, _ = self.forward(obs)
        return values

    def evaluate_actions(self, obs, action):
        values, action_param, action_logstd = self.forward(obs)
        if self.is_discrete:
            dist = Categorical(logits=action_param)
            action = action.squeeze(-1)
            action_log_probs = dist.log_prob(action)
        else:
            dist = Normal(action_param, torch.exp(action_logstd))
            action_log_probs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().mean()
        return values, action_log_probs, dist_entropy, None


class RolloutBuffer:
    """Simple rollout buffer for PPO updates."""

    def __init__(self, num_steps, num_processes, obs_shape, obj_num):
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.obj_num = obj_num

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape, dtype=torch.float32)
        self.actions = torch.zeros(num_steps, num_processes, 1, dtype=torch.long)
        self.action_log_probs = torch.zeros(num_steps, num_processes, dtype=torch.float32)
        self.values = torch.zeros(num_steps + 1, num_processes, obj_num, dtype=torch.float32)
        self.rewards = torch.zeros(num_steps, num_processes, obj_num, dtype=torch.float32)
        self.masks = torch.ones(num_steps + 1, num_processes, 1, dtype=torch.float32)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1, dtype=torch.float32)
        self.returns = torch.zeros(num_steps + 1, num_processes, obj_num, dtype=torch.float32)
        self.step = 0

    def insert(self, obs, actions, action_log_probs, values, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        self.actions[self.step].copy_(actions.long())
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.values[self.step].copy_(values)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step += 1

    def compute_returns(self, next_value, gamma, gae_lambda):
        self.values[-1] = next_value
        gae = torch.zeros(self.num_processes, self.obj_num, dtype=torch.float32)
        for step in reversed(range(self.num_steps)):
            delta = self.rewards[step] + gamma * self.values[step + 1] * self.masks[step + 1] - self.values[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.values[step]

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.step = 0

    def get_batch(self, advantages):
        obs = self.obs[:-1].reshape(-1, *self.obs_shape)
        actions = self.actions.reshape(-1, 1)
        old_log_probs = self.action_log_probs.reshape(-1)
        returns = self.returns[:-1].reshape(-1, self.obj_num)
        values = self.values[:-1].reshape(-1, self.obj_num)
        advantages = advantages.reshape(-1)
        return obs, actions, old_log_probs, returns, values, advantages


class TVPPO:
    """PPO with weighted-sum update and IPO-style constrained extension."""

    def __init__(
        self,
        policy,
        clip_param=0.2,
        ppo_epoch=4,
        num_mini_batch=4,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=3e-4,
        max_grad_norm=0.5,
        beta=0.85,
        t=10,
    ):
        self.policy = policy
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.beta = beta
        self.t = t
        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    def update(self, rollouts, scalarization):
        returns = rollouts.returns[:-1]
        values = rollouts.values[:-1]
        weights = scalarization.weights.float()

        scalarized_returns = (returns * weights).sum(dim=-1)
        scalarized_values = (values * weights).sum(dim=-1)
        advantages = scalarized_returns - scalarized_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        return self._ppo_update(rollouts, advantages, weights=weights)

    def ipo_update(self, rollouts, obj_idx, obj_num):
        del obj_num
        returns = rollouts.returns[:-1]
        values = rollouts.values[:-1]
        advantages = returns[:, :, obj_idx] - values[:, :, obj_idx]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return self._ppo_update(rollouts, advantages, objective_idx=obj_idx)

    def _ppo_update(self, rollouts, advantages, weights=None, objective_idx=None):
        obs_batch, actions_batch, old_log_probs_batch, return_batch, _, adv_batch = rollouts.get_batch(advantages)
        batch_size = obs_batch.size(0)
        mini_batch_size = max(1, batch_size // self.num_mini_batch)

        total_value_loss = 0.0
        total_action_loss = 0.0
        total_entropy = 0.0
        total_updates = 0

        for _ in range(self.ppo_epoch):
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                obs = obs_batch[idx]
                actions = actions_batch[idx]
                old_log_probs = old_log_probs_batch[idx]
                returns = return_batch[idx]
                advs = adv_batch[idx]

                values, action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(obs, actions)
                ratio = torch.exp(action_log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advs
                action_loss = -torch.min(surr1, surr2).mean()

                if objective_idx is None:
                    scalarized_values = (values * weights).sum(dim=-1)
                    scalarized_returns = (returns * weights).sum(dim=-1)
                    value_loss = 0.5 * (scalarized_returns - scalarized_values).pow(2).mean()
                else:
                    target_return = returns[:, objective_idx]
                    target_value = values[:, objective_idx]
                    value_loss = 0.5 * (target_return - target_value).pow(2).mean()

                    base_returns = returns.mean(dim=0)
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    cost = clipped_ratio.unsqueeze(-1) * base_returns.unsqueeze(0)
                    epsilon = self.beta * base_returns
                    hat_cost = epsilon.unsqueeze(0) - cost
                    keep_dims = [i for i in range(self.policy.obj_num) if i != objective_idx]
                    hat_cost = hat_cost[:, keep_dims]
                    hat_cost = torch.clamp(hat_cost, max=-1e-3)
                    ipo_loss = -torch.log(-hat_cost).mean() / self.t
                    action_loss = action_loss + ipo_loss

                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_value_loss += float(value_loss.item())
                total_action_loss += float(action_loss.item())
                total_entropy += float(dist_entropy.item())
                total_updates += 1

        total_updates = max(1, total_updates)
        return (
            total_value_loss / total_updates,
            total_action_loss / total_updates,
            total_entropy / total_updates,
        )
