from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from src.rl.environment import PortfolioEnv, DiscretePortfolioEnv
from src.features.graph import CorrelationGraph
from src.features.tda import TDAFeatureExtractor
from src.features.sentiment import (
    build_sentiment_features,
    GLOBAL_SENTIMENT_DIM,
    PER_ASSET_SENTIMENT_DIM,
    PER_ASSET_SENTIMENT_DIM_FALLBACK,
)
from src.data.sentiment import SentimentPipeline

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _build_price_features(returns: np.ndarray, lookback: int = 20) -> np.ndarray:

    T, N = returns.shape
    features = np.zeros((T, N, 5), dtype=np.float32)

    features[:, :, 0] = returns

    for t in range(T):

        if t >= 5:
            features[t, :, 1] = returns[t - 5:t].sum(axis=0)

        if t >= 20:
            features[t, :, 2] = returns[t - 20:t].sum(axis=0)

        if t >= 20:
            features[t, :, 3] = returns[t - 20:t].std(axis=0)

        if t >= 20:
            mean = returns[t - 20:t].mean(axis=0)
            std = returns[t - 20:t].std(axis=0)
            std[std < 1e-8] = 1.0
            features[t, :, 4] = (returns[t] - mean) / std

    return features

def _build_graph_tda_features(
    returns: np.ndarray,
    corr_window: int = 60,
    graph_k: int = 10,
    diffusion_times: list[float] | None = None,
    n_clusters: int = 4,
    recompute_freq: int = 5,
) -> tuple[np.ndarray, np.ndarray]:

    if diffusion_times is None:
        diffusion_times = [0.5, 1.0, 5.0]

    T, N = returns.shape
    n_diff = len(diffusion_times)
    n_pa = n_diff + 4
    n_global = 13

    per_asset = np.zeros((T, N, n_pa), dtype=np.float32)
    global_feats = np.zeros((T, n_global), dtype=np.float32)

    graph = CorrelationGraph(
        method="knn", k=graph_k,
        diffusion_times=diffusion_times,
    )
    tda = TDAFeatureExtractor(max_homology_dim=1, n_persistence_stats=5)

    prev_lambda2 = 0.0
    prev_betti1 = 0.0
    last_recompute = -recompute_freq

    for t in range(corr_window, T):
        need_recompute = (t - last_recompute) >= recompute_freq

        if need_recompute:
            window = returns[t - corr_window:t]
            try:
                graph.fit(window)
                tda_feats = tda.extract_features(window)
                last_recompute = t
            except Exception:
                continue

        try:
            day_ret = np.nan_to_num(returns[t], nan=0.0)
            gf = graph.extract_features(day_ret, n_clusters=n_clusters)

            per_asset[t, :, :n_diff] = gf["diffusion_residuals"]
            per_asset[t, :, n_diff] = gf["cluster_labels"]
            per_asset[t, :, n_diff + 1] = gf["cluster_distances"]
            per_asset[t, :, n_diff + 2] = gf["fiedler"]
            per_asset[t, :, n_diff + 3] = gf["eigenvector_centrality"]

            lambda2 = float(gf["algebraic_connectivity"][0])
            sgap = float(gf["spectral_gap"][0])
            betti0 = float(tda_feats["betti_0"][0])
            betti1 = float(tda_feats["betti_1"][0])

            global_feats[t] = [
                lambda2, sgap, betti0, betti1,
                float(tda_feats["h0_mean_lifetime"][0]),
                float(tda_feats["h0_max_lifetime"][0]),
                float(tda_feats["h1_mean_lifetime"][0]),
                float(tda_feats["h1_max_lifetime"][0]),
                float(tda_feats["h1_total_persistence"][0]),
                float(tda_feats["persistence_entropy_h0"][0]),
                float(tda_feats["persistence_entropy_h1"][0]),
                lambda2 - prev_lambda2,
                betti1 - prev_betti1,
            ]
            prev_lambda2 = lambda2
            prev_betti1 = betti1

        except Exception:
            continue

    return per_asset, global_feats

def _build_sentiment_features(
    config: dict,
    returns: np.ndarray,
    train_data: dict,
    daily_sentiment: pd.DataFrame | None,
    per_ticker_sentiment: pd.DataFrame | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:

    if not config["use_sentiment"]:
        return None, None
    if daily_sentiment is None or len(daily_sentiment) == 0:
        return None, None

    dates = train_data.get("dates", None)
    if dates is None:
        ret_df = train_data["returns"]
        if isinstance(ret_df, pd.DataFrame):
            dates = ret_df.index
        else:
            T = returns.shape[0]
            dates = pd.bdate_range(end="2024-12-31", periods=T)

    permno_list = train_data.get("permno_list", None)
    if permno_list is None:
        ret_df = train_data["returns"]
        if isinstance(ret_df, pd.DataFrame):
            try:
                permno_list = [int(c) for c in ret_df.columns]
            except (ValueError, TypeError):
                permno_list = None

    try:
        sent_feats = build_sentiment_features(
            daily_sentiment, returns, dates,
            per_ticker_sentiment=per_ticker_sentiment,
            permno_list=permno_list,
        )
        return sent_feats.per_asset_features, sent_feats.global_features
    except Exception as e:
        logger.warning("Failed to build sentiment features: %s", e)
        return None, None

class _BaseRLModel:

    def __init__(
        self,
        algorithm: str = "PPO",
        lookback: int = 20,
        total_timesteps: int = 400_000,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        max_weight: float = 0.05,
        reward_type: str = "composite",
        corr_window: int = 60,
        graph_k: int = 10,
        n_clusters: int = 4,
        diffusion_times: list[float] | None = None,
        seed: int = 42,
        n_seeds: int = 1,
        use_recurrent: bool = False,
        use_vec_normalize: bool = True,
        net_arch: list[int] | None = None,

        lambda_dd: float = 2.0,
        lambda_turnover: float = 0.5,
        dd_threshold: float = 0.05,
        dsr_eta: float = 0.005,
    ):
        self.algorithm = algorithm
        self.lookback = lookback
        self.total_timesteps = total_timesteps
        self.tc_bps = transaction_cost_bps
        self.slip_bps = slippage_bps
        self.max_weight = max_weight
        self.reward_type = reward_type
        self.corr_window = corr_window
        self.graph_k = graph_k
        self.n_clusters = n_clusters
        self.diffusion_times = diffusion_times or [0.5, 1.0, 5.0]
        self.seed = seed
        self.n_seeds = n_seeds
        self.use_recurrent = use_recurrent
        self.use_vec_normalize = use_vec_normalize
        self.net_arch = net_arch
        self.lambda_dd = lambda_dd
        self.lambda_turnover = lambda_turnover
        self.dd_threshold = dd_threshold
        self.dsr_eta = dsr_eta

        self._agent = None
        self._vec_normalize = None
        self._n_assets = 0
        self._graph = None
        self._tda = None
        self._daily_sentiment = None
        self._per_ticker_sentiment = None

    def _get_feature_config(self) -> dict:

        return {
            "use_graph": True,
            "use_sentiment": True,
        }

    def _load_sentiment(self) -> pd.DataFrame | None:

        if self._daily_sentiment is not None:
            return self._daily_sentiment
        try:
            pipe = SentimentPipeline()
            self._daily_sentiment = pipe.build()
            if self._daily_sentiment is not None and len(self._daily_sentiment) > 0:
                logger.info(
                    "Loaded market sentiment: %d days (%s to %s)",
                    len(self._daily_sentiment),
                    self._daily_sentiment.index.min().date(),
                    self._daily_sentiment.index.max().date(),
                )
            return self._daily_sentiment
        except Exception as e:
            logger.warning("Failed to load sentiment data: %s", e)
            return None

    def _load_per_ticker_sentiment(self) -> pd.DataFrame | None:

        if self._per_ticker_sentiment is not None:
            return self._per_ticker_sentiment
        try:
            pipe = SentimentPipeline()
            self._per_ticker_sentiment = pipe.build_per_ticker()
            if self._per_ticker_sentiment is not None:
                n_permnos = self._per_ticker_sentiment.index.get_level_values("permno").nunique()
                logger.info("Loaded per-ticker sentiment: %d rows, %d PERMNOs",
                            len(self._per_ticker_sentiment), n_permnos)
            return self._per_ticker_sentiment
        except Exception as e:
            logger.warning("Failed to load per-ticker sentiment: %s", e)
            return None

    def _make_env(self, returns: np.ndarray, features_pa: np.ndarray,
                  features_global: np.ndarray | None) -> DummyVecEnv | VecNormalize:

        env_kwargs = dict(
            returns=returns,
            features_per_asset=features_pa,
            features_global=features_global,
            lookback=self.lookback,
            transaction_cost_bps=self.tc_bps,
            slippage_bps=self.slip_bps,
            reward_type=self.reward_type,
            max_weight=self.max_weight,
            lambda_dd=self.lambda_dd,
            lambda_turnover=self.lambda_turnover,
            dd_threshold=self.dd_threshold,
            dsr_eta=self.dsr_eta,
        )

        if self.algorithm == "DQN":
            env = DummyVecEnv([lambda: DiscretePortfolioEnv(**env_kwargs)])
        else:
            env = DummyVecEnv([lambda: PortfolioEnv(**env_kwargs)])

        if self.use_vec_normalize and self.algorithm != "DQN":
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0,
                gamma=0.99,
            )

        return env

    def _create_agent(self, env, seed: int):

        net_arch = self.net_arch if self.net_arch is not None else [256, 256]

        policy_kwargs = dict(
            net_arch=net_arch,
            activation_fn=torch.nn.Tanh,
        )

        if self.algorithm == "DQN":

            return DQN(
                "MlpPolicy", env,
                learning_rate=2.1e-5,
                buffer_size=17_000,
                learning_starts=1636,
                batch_size=64,
                gamma=0.996,
                target_update_interval=680,
                exploration_fraction=0.30,
                exploration_final_eps=0.073,
                policy_kwargs=policy_kwargs,
                verbose=0,
                seed=seed,
                device=DEVICE,
            )

        if self.use_recurrent:
            from sb3_contrib import RecurrentPPO
            recurrent_kwargs = dict(
                lstm_hidden_size=128,
                n_lstm_layers=1,
                shared_lstm=False,
                enable_critic_lstm=True,
            )
            return RecurrentPPO(
                "MlpLstmPolicy", env,
                learning_rate=1e-4,
                n_steps=512,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                max_grad_norm=0.5,
                policy_kwargs=recurrent_kwargs,
                verbose=0,
                seed=seed,
                device=DEVICE,
            )

        return PPO(
            "MlpPolicy", env,
            learning_rate=3.5e-4,
            n_steps=512,
            batch_size=32,
            n_epochs=10,
            gamma=0.982,
            gae_lambda=0.976,
            clip_range=0.3,
            ent_coef=8e-4,
            vf_coef=0.47,
            max_grad_norm=0.51,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            device=DEVICE,
        )

    def _evaluate_agent_on_train(self, agent, env, n_eval_episodes: int = 3) -> float:

        total_reward = 0.0
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done_arr, info = env.step(action)
                ep_reward += reward[0]
                done = done_arr[0]
            total_reward += ep_reward
        return total_reward / n_eval_episodes

    def fit(self, train_data: dict) -> None:

        returns = train_data["returns"]
        if isinstance(returns, pd.DataFrame):
            returns_arr = returns.values
        else:
            returns_arr = returns
        returns_arr = np.nan_to_num(returns_arr, nan=0.0).astype(np.float32)

        T, N = returns_arr.shape
        self._n_assets = N
        config = self._get_feature_config()

        price_feats = _build_price_features(returns_arr, self.lookback)
        features_pa = price_feats
        features_global = None

        if config["use_graph"]:
            logger.info("Computing graph/TDA features for training...")
            graph_pa, graph_global = _build_graph_tda_features(
                returns_arr,
                corr_window=self.corr_window,
                graph_k=self.graph_k,
                diffusion_times=self.diffusion_times,
                n_clusters=self.n_clusters,
            )
            features_pa = np.concatenate([features_pa, graph_pa], axis=2)
            features_global = graph_global

        if config["use_sentiment"]:
            logger.info("Building sentiment features for training...")
            daily_sent = self._load_sentiment()
            per_ticker_sent = self._load_per_ticker_sentiment()
            sent_pa, sent_global = _build_sentiment_features(
                config, returns_arr, train_data, daily_sent,
                per_ticker_sentiment=per_ticker_sent,
            )
            if sent_pa is not None:
                features_pa = np.concatenate([features_pa, sent_pa], axis=2)
                logger.info("Sentiment per-asset features added: %d dims", sent_pa.shape[2])
            if sent_global is not None:
                if features_global is not None:
                    features_global = np.concatenate(
                        [features_global, sent_global], axis=1,
                    )
                else:
                    features_global = sent_global
                logger.info("Sentiment global features added: %d dims", sent_global.shape[1])
            if sent_pa is None and sent_global is None:
                logger.warning("No sentiment data available — training without sentiment.")

        best_agent = None
        best_env = None
        best_score = -np.inf

        seeds = [self.seed + i * 1000 for i in range(self.n_seeds)]

        for i, seed in enumerate(seeds):
            if self.n_seeds > 1:
                logger.info("Training seed %d/%d (seed=%d)...", i + 1, self.n_seeds, seed)

            env = self._make_env(returns_arr, features_pa, features_global)
            agent = self._create_agent(env, seed)

            logger.info(
                "Training %s agent (%s%s, reward=%s) for %d timesteps on %d assets...",
                self.__class__.__name__,
                self.algorithm,
                "+LSTM" if self.use_recurrent else "",
                self.reward_type,
                self.total_timesteps, N,
            )
            agent.learn(total_timesteps=self.total_timesteps)

            if self.n_seeds > 1:
                eval_env = self._make_env(returns_arr, features_pa, features_global)
                if isinstance(eval_env, VecNormalize) and isinstance(env, VecNormalize):
                    eval_env.obs_rms = env.obs_rms
                    eval_env.ret_rms = env.ret_rms
                    eval_env.training = False
                    eval_env.norm_reward = False
                score = self._evaluate_agent_on_train(agent, eval_env)
                logger.info("  Seed %d score: %.4f", seed, score)
                if score > best_score:
                    best_score = score
                    best_agent = agent
                    best_env = env
            else:
                best_agent = agent
                best_env = env

        self._agent = best_agent
        if isinstance(best_env, VecNormalize):
            self._vec_normalize = best_env
        logger.info("Training complete.")

        if config["use_graph"]:
            self._graph = CorrelationGraph(
                method="knn", k=self.graph_k,
                diffusion_times=self.diffusion_times,
            )
            self._tda = TDAFeatureExtractor(max_homology_dim=1)

    def predict_weights(self, current_data: dict) -> np.ndarray:

        returns = current_data["returns"]
        if isinstance(returns, pd.DataFrame):
            returns_arr = returns.values
        else:
            returns_arr = returns
        returns_arr = np.nan_to_num(returns_arr, nan=0.0).astype(np.float32)

        n_assets = returns_arr.shape[1]
        weights = np.zeros(n_assets, dtype=np.float32)

        if self._agent is None:
            weights[:] = 1.0 / n_assets
            return weights

        config = self._get_feature_config()

        price_feats = _build_price_features(returns_arr, self.lookback)
        features_pa = price_feats

        if config["use_graph"] and self._graph is not None:
            graph_pa, graph_global = _build_graph_tda_features(
                returns_arr,
                corr_window=self.corr_window,
                graph_k=self.graph_k,
                diffusion_times=self.diffusion_times,
                n_clusters=self.n_clusters,
                recompute_freq=max(1, len(returns_arr) // 10),
            )
            features_pa = np.concatenate([features_pa, graph_pa], axis=2)
            features_global = graph_global
        else:
            features_global = None

        if config["use_sentiment"]:
            daily_sent = self._load_sentiment()
            per_ticker_sent = self._load_per_ticker_sentiment()
            sent_pa, sent_global = _build_sentiment_features(
                config, returns_arr, current_data, daily_sent,
                per_ticker_sentiment=per_ticker_sent,
            )
            if sent_pa is not None:
                features_pa = np.concatenate([features_pa, sent_pa], axis=2)
            if sent_global is not None:
                if features_global is not None:
                    features_global = np.concatenate(
                        [features_global, sent_global], axis=1,
                    )
                else:
                    features_global = sent_global

        env_kwargs = dict(
            returns=returns_arr,
            features_per_asset=features_pa,
            features_global=features_global,
            lookback=self.lookback,
            transaction_cost_bps=self.tc_bps,
            slippage_bps=self.slip_bps,
            max_weight=self.max_weight,
        )

        if self.algorithm == "DQN":
            tmp_env = DiscretePortfolioEnv(**env_kwargs)
        else:
            tmp_env = PortfolioEnv(**env_kwargs)

        tmp_env._step = len(returns_arr) - 1
        tmp_env._weights = np.zeros(n_assets, dtype=np.float32)
        obs = tmp_env._get_obs()

        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)

        action, _ = self._agent.predict(obs, deterministic=True)

        if self.algorithm == "DQN":
            weights = tmp_env._discrete_to_weights(int(action))
        else:
            weights = tmp_env._normalize_weights(action)

        return weights

class RLFullModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("total_timesteps", 400_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("net_arch", [256, 256])
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

class RLNoGraphModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("total_timesteps", 400_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("net_arch", [256, 256])
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": False}

class RLNoSentimentModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("total_timesteps", 400_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("net_arch", [256, 256])
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": False}

class RLDQNModel(_BaseRLModel):

    def __init__(self, **kwargs):
        kwargs.setdefault("algorithm", "DQN")
        kwargs.setdefault("total_timesteps", 400_000)
        kwargs.setdefault("reward_type", "sharpe")
        kwargs.setdefault("use_vec_normalize", False)
        kwargs.setdefault("net_arch", [64, 64])
        super().__init__(**kwargs)

    def _get_feature_config(self) -> dict:
        return {"use_graph": True, "use_sentiment": True}

class SAPPOModel(_BaseRLModel):

    def __init__(self, sappo_alpha: float = 0.1, **kwargs):
        kwargs.setdefault("algorithm", "PPO")
        kwargs.setdefault("total_timesteps", 500_000)
        kwargs.setdefault("reward_type", "composite")
        kwargs.setdefault("use_vec_normalize", True)
        kwargs.setdefault("net_arch", [256, 256])
        super().__init__(**kwargs)
        self.sappo_alpha = sappo_alpha

    def _get_feature_config(self) -> dict:
        return {"use_graph": False, "use_sentiment": True}

    def _make_env(self, returns, features_pa, features_global):

        from src.rl.environment import SAPPOEnv

        env_kwargs = dict(
            returns=returns,
            features_per_asset=features_pa,
            features_global=features_global,
            lookback=self.lookback,
            transaction_cost_bps=self.tc_bps,
            slippage_bps=self.slip_bps,
            reward_type=self.reward_type,
            max_weight=self.max_weight,
            lambda_dd=self.lambda_dd,
            lambda_turnover=self.lambda_turnover,
            dd_threshold=self.dd_threshold,
            dsr_eta=self.dsr_eta,
            sappo_alpha=self.sappo_alpha,
        )

        env = DummyVecEnv([lambda: SAPPOEnv(**env_kwargs)])
        if self.use_vec_normalize:
            env = VecNormalize(
                env, norm_obs=True, norm_reward=True,
                clip_obs=10.0, clip_reward=10.0, gamma=0.99,
            )
        return env
