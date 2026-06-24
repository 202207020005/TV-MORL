import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    s = float(vec.sum())
    if s <= 1e-8:
        return np.ones_like(vec) / max(1, len(vec))
    return vec / s


class MovieLensStaticBundle:
    def __init__(self, movie_ids, genre_names, item_genres, popularity, novelty, user_ids, user_profiles, user_meta, user_ratings):
        self.movie_ids = movie_ids
        self.genre_names = genre_names
        self.item_genres = item_genres
        self.popularity = popularity
        self.novelty = novelty
        self.user_ids = user_ids
        self.user_profiles = user_profiles
        self.user_meta = user_meta
        self.user_ratings = user_ratings


def load_movielens_static_bundle(data_dir, min_user_ratings=20):
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

    all_genres = sorted({genre for row in movies["genres"] for genre in row.split("|")})
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
    movie_ids = movies["movie_id"].to_numpy(dtype=np.int64)
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    item_genres = np.zeros((len(movie_ids), len(all_genres)), dtype=np.float32)
    for _, row in movies.iterrows():
        idx = movie_to_idx[row["movie_id"]]
        for genre in row["genres"].split("|"):
            item_genres[idx, genre_to_idx[genre]] = 1.0
        item_genres[idx] = _normalize(item_genres[idx])

    popularity_counts = ratings["movie_id"].value_counts().reindex(movie_ids).fillna(0.0).to_numpy(dtype=np.float32)
    popularity = popularity_counts / max(1.0, popularity_counts.max())
    novelty = 1.0 - np.log1p(popularity_counts) / np.log1p(max(1.0, popularity_counts.max()))

    eligible_users = ratings["user_id"].value_counts()
    eligible_users = eligible_users[eligible_users >= min_user_ratings].index.to_numpy()
    ratings = ratings[ratings["user_id"].isin(eligible_users)].copy()
    users = users[users["user_id"].isin(eligible_users)].copy()

    age_values = users["age"].to_numpy(dtype=np.float32)
    age_min = float(age_values.min())
    age_max = float(age_values.max())

    user_profiles = {}
    user_meta = {}
    user_ratings = {}
    for user_id, group in ratings.groupby("user_id"):
        mids = [movie_to_idx[mid] for mid in group["movie_id"].to_numpy() if mid in movie_to_idx]
        if not mids:
            continue
        rv = group["rating"].to_numpy(dtype=np.float32)[: len(mids)]
        w = np.clip(rv - 2.5, 0.0, None) + 0.05
        prof = _normalize((item_genres[np.array(mids)] * w[:, None]).sum(axis=0))
        user_profiles[int(user_id)] = prof
        user_ratings[int(user_id)] = {int(idx): float((rating - 1.0) / 4.0) for idx, rating in zip(mids, rv)}
        meta = users[users["user_id"] == user_id].iloc[0]
        user_meta[int(user_id)] = np.array(
            [
                1.0 if meta["gender"] == "M" else 0.0,
                (float(meta["age"]) - age_min) / max(1.0, age_max - age_min),
                float(meta["occupation"]) / 20.0,
            ],
            dtype=np.float32,
        )

    return MovieLensStaticBundle(
        movie_ids=movie_ids,
        genre_names=all_genres,
        item_genres=item_genres,
        popularity=popularity.astype(np.float32),
        novelty=novelty.astype(np.float32),
        user_ids=np.array(sorted(user_profiles.keys()), dtype=np.int64),
        user_profiles=user_profiles,
        user_meta=user_meta,
        user_ratings=user_ratings,
    )


class MovieLensStaticEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, bundle, max_steps=40, seed=None):
        super().__init__()
        self.bundle = bundle
        self.max_steps = max_steps
        self.seed_val = 0 if seed is None else seed
        self.rng = np.random.RandomState(self.seed_val)
        self.n_items = len(bundle.movie_ids)
        self.n_genres = len(bundle.genre_names)

        obs_dim = self.n_genres + 3 + 3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_items)
        self.reward_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.obj_dim = 3
        self.reset()

    def set_params(self, env_params):
        del env_params

    def _get_obs(self):
        return np.concatenate(
            [
                self.user_preference,
                np.array([self.history_relevance, self.history_diversity, self.history_novelty], dtype=np.float32),
                self.user_meta,
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.current_step = 0
        self.user_id = int(self.rng.choice(self.bundle.user_ids))
        self.user_preference = self.bundle.user_profiles[self.user_id].copy()
        self.user_meta = self.bundle.user_meta[self.user_id].copy()
        self.user_rating_map = self.bundle.user_ratings[self.user_id]
        self.recent_items = []
        self.relevance_history = []
        self.history_relevance = 0.5
        self.history_diversity = 0.5
        self.history_novelty = 0.5
        return self._get_obs(), {}

    def step(self, action):
        item_idx = int(action)
        item_profile = self.bundle.item_genres[item_idx]
        match = float(np.dot(self.user_preference, item_profile))
        explicit = self.user_rating_map.get(item_idx)
        popularity = float(self.bundle.popularity[item_idx])
        relevance = float(np.clip(0.65 * explicit + 0.35 * (0.75 * match + 0.25 * popularity), 0.0, 1.0)) if explicit is not None else float(np.clip(0.75 * match + 0.25 * popularity, 0.0, 1.0))

        if not self.recent_items:
            diversity = 0.6
        else:
            hist = self.bundle.item_genres[np.array(self.recent_items, dtype=np.int64)]
            diversity = float(np.clip(1.0 - (hist @ item_profile).mean(), 0.0, 1.0))
        novelty = float(np.clip(self.bundle.novelty[item_idx] - (0.2 if item_idx in self.recent_items else 0.0), 0.0, 1.0))

        self.recent_items.append(item_idx)
        self.recent_items = self.recent_items[-6:]
        self.relevance_history.append(relevance)
        self.relevance_history = self.relevance_history[-6:]
        self.history_relevance = float(np.mean(self.relevance_history))
        recent_profiles = self.bundle.item_genres[np.array(self.recent_items, dtype=np.int64)]
        self.history_diversity = float(np.clip(1.0 - (recent_profiles @ recent_profiles.T).mean(), 0.0, 1.0))
        self.history_novelty = float(np.mean([self.bundle.novelty[i] for i in self.recent_items]))

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        obj = np.array([relevance, diversity, novelty], dtype=np.float32)
        info = {"obj": obj, "obj_raw": obj.copy()}
        return self._get_obs(), 0.0, terminated, truncated, info
