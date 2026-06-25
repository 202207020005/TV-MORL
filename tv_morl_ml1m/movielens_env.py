from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    total = float(vec.sum())
    if total <= 1e-8:
        return np.ones_like(vec, dtype=np.float32) / max(1, len(vec))
    return vec / total


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _bounded_interp(static_value, dynamic_value, strength):
    strength = _clip01(strength)
    return (1.0 - strength) * static_value + strength * dynamic_value


@dataclass
class MovieLensBundle:
    movie_ids: np.ndarray
    genre_names: list[str]
    item_genres: np.ndarray
    popularity: np.ndarray
    novelty: np.ndarray
    user_ids: np.ndarray
    user_profiles: dict
    user_contexts: dict
    user_meta: dict
    user_ratings: dict
    timestamp_min: int
    timestamp_max: int


def load_movielens_bundle(data_dir, max_items=None, min_user_ratings=20):
    movies = pd.read_csv(
        f"{data_dir}/movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    users = pd.read_csv(
        f"{data_dir}/users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
    )

    if max_items is not None and max_items > 0:
        top_movies = ratings["movie_id"].value_counts().head(max_items).index.to_numpy()
        movies = movies[movies["movie_id"].isin(top_movies)].copy()
        ratings = ratings[ratings["movie_id"].isin(top_movies)].copy()

    all_genres = sorted({genre for row in movies["genres"] for genre in row.split("|")})
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies["movie_id"].to_numpy())}

    item_genres = np.zeros((len(movies), len(all_genres)), dtype=np.float32)
    for _, row in movies.iterrows():
        item_idx = movie_to_idx[row["movie_id"]]
        genres = row["genres"].split("|")
        for genre in genres:
            item_genres[item_idx, genre_to_idx[genre]] = 1.0
        item_genres[item_idx] = _normalize(item_genres[item_idx])

    popularity_counts = ratings["movie_id"].value_counts().reindex(movies["movie_id"]).fillna(1.0).to_numpy(dtype=np.float32)
    popularity = popularity_counts / popularity_counts.max()
    novelty = 1.0 - np.log1p(popularity_counts) / np.log1p(popularity_counts.max())
    novelty = novelty.astype(np.float32)

    merged = ratings.merge(movies[["movie_id"]], on="movie_id", how="inner")
    eligible_users = merged["user_id"].value_counts()
    eligible_users = eligible_users[eligible_users >= min_user_ratings].index.to_numpy()
    merged = merged[merged["user_id"].isin(eligible_users)].copy()
    users = users[users["user_id"].isin(eligible_users)].copy()

    age_values = users["age"].to_numpy(dtype=np.float32)
    age_min = float(age_values.min())
    age_max = float(age_values.max())

    user_profiles = {}
    user_contexts = {}
    user_meta = {}
    user_ratings = {}

    for user_id, group in merged.sort_values("timestamp").groupby("user_id"):
        item_indices = np.array([movie_to_idx[mid] for mid in group["movie_id"].to_numpy()], dtype=np.int64)
        rating_values = group["rating"].to_numpy(dtype=np.float32)
        positive_weight = np.clip(rating_values - 2.5, 0.0, None) + 0.05
        profile = (item_genres[item_indices] * positive_weight[:, None]).sum(axis=0)
        profile = _normalize(profile)

        session_items = item_indices[-8:]
        session_weights = np.clip(rating_values[-8:] - 2.5, 0.0, None) + 0.05
        context = _normalize((item_genres[session_items] * session_weights[:, None]).sum(axis=0))

        user_profiles[int(user_id)] = profile
        user_contexts[int(user_id)] = context
        user_ratings[int(user_id)] = {
            int(idx): float((rating - 1.0) / 4.0) for idx, rating in zip(item_indices, rating_values)
        }

        meta_row = users[users["user_id"] == user_id].iloc[0]
        user_meta[int(user_id)] = np.array(
            [
                1.0 if meta_row["gender"] == "M" else 0.0,
                (float(meta_row["age"]) - age_min) / max(1.0, age_max - age_min),
                float(meta_row["occupation"]) / 20.0,
            ],
            dtype=np.float32,
        )

    return MovieLensBundle(
        movie_ids=movies["movie_id"].to_numpy(dtype=np.int64),
        genre_names=all_genres,
        item_genres=item_genres,
        popularity=popularity,
        novelty=novelty,
        user_ids=np.array(sorted(user_profiles.keys()), dtype=np.int64),
        user_profiles=user_profiles,
        user_contexts=user_contexts,
        user_meta=user_meta,
        user_ratings=user_ratings,
        timestamp_min=int(merged["timestamp"].min()),
        timestamp_max=int(merged["timestamp"].max()),
    )


class MovieLensRealTVEnv(gym.Env):
    """
    Time-varying MovieLens-1M environment.

    Pt: current preference drifts between long-term taste and short-term session context.
    Rt: satisfaction reward gradually emphasizes short-term session match.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        bundle,
        max_steps=40,
        seed=None,
        enable_pt=True,
        enable_rt=True,
        rt_rel_strength=1.0,
        rt_div_strength=1.0,
        rt_nov_strength=1.0,
        time_gate_min=0.15,
        time_gate_max=0.65,
        seasonal_amp=0.04,
        rel_static_gate=0.35,
        rel_pop_weight=0.10,
        rel_explicit_weight=0.70,
        div_scale_min=0.60,
        nov_scale_min=0.50,
        nov_repeat_penalty_max=0.20,
        session_keep_rate=0.85,
        session_update_rate=0.15,
        rt_click_strength=None,
    ):
        super().__init__()
        self.bundle = bundle
        self.max_steps = max_steps
        self.seed_val = 0 if seed is None else seed
        self.rng = np.random.RandomState(self.seed_val)
        self.enable_pt = enable_pt
        self.enable_rt = enable_rt
        if rt_click_strength is not None:
            rt_rel_strength = rt_click_strength
        self.rt_rel_strength = _clip01(rt_rel_strength)
        self.rt_div_strength = _clip01(rt_div_strength)
        self.rt_nov_strength = _clip01(rt_nov_strength)
        self.time_gate_min = _clip01(min(time_gate_min, time_gate_max))
        self.time_gate_max = _clip01(max(time_gate_min, time_gate_max))
        self.seasonal_amp = max(0.0, float(seasonal_amp))
        self.rel_static_gate = _clip01(rel_static_gate)
        self.rel_pop_weight = _clip01(rel_pop_weight)
        self.rel_explicit_weight = _clip01(rel_explicit_weight)
        self.div_scale_min = _clip01(div_scale_min)
        self.nov_scale_min = _clip01(nov_scale_min)
        self.nov_repeat_penalty_max = max(0.0, float(nov_repeat_penalty_max))
        self.session_keep_rate = _clip01(session_keep_rate)
        self.session_update_rate = _clip01(session_update_rate)

        self.n_items = len(bundle.movie_ids)
        self.n_genres = len(bundle.genre_names)
        self._max_episode_steps = max_steps
        self._elapsed_steps = 0

        obs_dim = self.n_genres * 2 + 3 + 3 + 2
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_items)
        self.reward_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.obj_dim = 3

        self.reset()

    def set_params(self, env_params):
        del env_params

    def _time_features(self):
        time_ratio = self.global_time / max(1.0, float(self.bundle.timestamp_max - self.bundle.timestamp_min))
        return np.array(
            [
                np.sin(2.0 * np.pi * time_ratio),
                np.cos(2.0 * np.pi * time_ratio),
            ],
            dtype=np.float32,
        )

    def _seasonal_shift(self):
        time_ratio = self.global_time / max(1.0, float(self.bundle.timestamp_max - self.bundle.timestamp_min))
        phases = np.linspace(0.0, 2.0 * np.pi, self.n_genres, endpoint=False)
        shift = self.seasonal_amp * np.sin(2.0 * np.pi * 3.0 * time_ratio + phases)
        return shift.astype(np.float32)

    def _progress(self):
        return self.current_step / max(1, self.max_steps - 1)

    def _time_gate(self):
        progress = self._progress()
        return self.time_gate_max - (self.time_gate_max - self.time_gate_min) * progress

    def _update_preference(self):
        if self.enable_pt:
            drift_scale = self._time_gate()
            blended = (1.0 - drift_scale) * self.base_preference + drift_scale * self.session_preference
            self.current_preference = _normalize(blended + self._seasonal_shift() + 1e-3)
        else:
            self.current_preference = self.base_preference.copy()

    def _get_obs(self):
        return np.concatenate(
            [
                self.current_preference,
                self.session_preference,
                np.array(
                    [self.history_satisfaction, self.history_diversity, self.history_novelty],
                    dtype=np.float32,
                ),
                self.user_meta,
                self._time_features(),
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.user_id = int(self.rng.choice(self.bundle.user_ids))
        self.base_preference = self.bundle.user_profiles[self.user_id].copy()
        self.session_preference = self.bundle.user_contexts[self.user_id].copy()
        self.user_meta = self.bundle.user_meta[self.user_id].copy()
        self.user_rating_map = self.bundle.user_ratings[self.user_id]

        self.current_step = 0
        self._elapsed_steps = 0
        self.recent_items = []
        self.recent_satisfaction = []
        self.history_satisfaction = 0.5
        self.history_diversity = 0.5
        self.history_novelty = 0.5
        self.anchor = self.rng.randint(self.bundle.timestamp_min, self.bundle.timestamp_max + 1)
        self.global_time = float(self.anchor - self.bundle.timestamp_min)
        self._update_preference()
        return self._get_obs(), {}

    def _compute_relevance(self, item_idx):
        item_profile = self.bundle.item_genres[item_idx]
        long_term = float(np.dot(self.base_preference, item_profile))
        short_term = float(np.dot(self.session_preference, item_profile))
        explicit = self.user_rating_map.get(int(item_idx))
        popularity = float(self.bundle.popularity[item_idx])
        if self.enable_rt:
            focus = _bounded_interp(self.rel_static_gate, self._time_gate(), self.rt_rel_strength)
        else:
            focus = self.rel_static_gate
        match_score = (1.0 - focus) * long_term + focus * short_term
        model_score = (1.0 - self.rel_pop_weight) * match_score + self.rel_pop_weight * popularity
        if explicit is not None:
            return float(
                np.clip(
                    self.rel_explicit_weight * explicit + (1.0 - self.rel_explicit_weight) * model_score,
                    0.0,
                    1.0,
                )
            )
        return float(np.clip(model_score, 0.0, 1.0))

    def _compute_diversity(self, item_idx):
        if not self.recent_items:
            base = 0.6
            if self.enable_rt:
                base_scale = self.div_scale_min + (1.0 - self.div_scale_min) * self._progress()
                scale = _bounded_interp(1.0, base_scale, self.rt_div_strength)
                return float(np.clip(base * scale, 0.0, 1.0))
            return base
        item_profile = self.bundle.item_genres[item_idx]
        hist = self.bundle.item_genres[np.array(self.recent_items, dtype=np.int64)]
        similarity = hist @ item_profile
        raw_diversity = float(np.clip(1.0 - similarity.mean(), 0.0, 1.0))
        if self.enable_rt:
            base_scale = self.div_scale_min + (1.0 - self.div_scale_min) * self._progress()
            scale = _bounded_interp(1.0, base_scale, self.rt_div_strength)
            return float(np.clip(raw_diversity * scale, 0.0, 1.0))
        return raw_diversity

    def _compute_novelty(self, item_idx):
        time_ratio = self._progress()
        dynamic_scale = self.nov_scale_min + (1.0 - self.nov_scale_min) * time_ratio
        novelty_scale = _bounded_interp(1.0, dynamic_scale, self.rt_nov_strength) if self.enable_rt else 1.0
        repeat_penalty = (
            self.rt_nov_strength * self.nov_repeat_penalty_max * time_ratio
            if (self.enable_rt and item_idx in self.recent_items)
            else (0.2 if item_idx in self.recent_items else 0.0)
        )
        novelty = float(self.bundle.novelty[item_idx]) * novelty_scale - repeat_penalty
        return float(np.clip(novelty, 0.0, 1.0))

    def step(self, action):
        item_idx = int(action)
        # Compute reward from the current state first. The next-state preference
        # will be updated only after the short-term session state is refreshed.
        preference_focus = self._time_gate()
        relevance_gate = (
            _bounded_interp(self.rel_static_gate, preference_focus, self.rt_rel_strength)
            if self.enable_rt
            else self.rel_static_gate
        )
        satisfaction = self._compute_relevance(item_idx)
        diversity = self._compute_diversity(item_idx)
        novelty = self._compute_novelty(item_idx)

        item_profile = self.bundle.item_genres[item_idx]
        if self.enable_pt:
            self.session_preference = _normalize(
                self.session_keep_rate * self.session_preference
                + self.session_update_rate * satisfaction * item_profile
                + 1e-4
            )
        self.recent_items.append(item_idx)
        self.recent_satisfaction.append(satisfaction)
        self.recent_items = self.recent_items[-6:]
        self.recent_satisfaction = self.recent_satisfaction[-6:]

        self.history_satisfaction = float(np.mean(self.recent_satisfaction))
        recent_profiles = self.bundle.item_genres[np.array(self.recent_items, dtype=np.int64)]
        pairwise = recent_profiles @ recent_profiles.T
        self.history_diversity = float(np.clip(1.0 - pairwise.mean(), 0.0, 1.0))
        self.history_novelty = float(np.mean([self.bundle.novelty[i] for i in self.recent_items]))

        self.current_step += 1
        self._elapsed_steps += 1
        self.global_time += 6.0 * 3600.0
        self._update_preference()

        terminated = False
        truncated = self.current_step >= self.max_steps
        obj = np.array([satisfaction, diversity, novelty], dtype=np.float32)
        info = {
            "obj": obj,
            "obj_raw": obj.copy(),
            "user_id": self.user_id,
            "movie_id": int(self.bundle.movie_ids[item_idx]),
            "preference_focus": preference_focus,
            "relevance_gate": relevance_gate,
        }
        return self._get_obs(), 0.0, terminated, truncated, info
