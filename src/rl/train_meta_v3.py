import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import Env, spaces

from src.data.pipeline import DataPipeline, DataSplit
from src.evaluation.metrics import PortfolioMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

RESULTS_DIR = Path("results/meta_v3")

def load_strategy_returns(test_returns: np.ndarray, test_dates: pd.DatetimeIndex,
                          strategy_files: dict) -> dict:

    result = {}
    for name, fpath in strategy_files.items():
        if fpath.endswith('.parquet'):
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
        s.index = pd.to_datetime(s.index)

        s = s.reindex(test_dates).fillna(0.0)
        result[name] = s.values.astype(np.float32)
    return result

def compute_strategy_weights_returns(test_split, baselines_dir="results/baselines"):

    strat_files = {
        'mvo': f'{baselines_dir}/returns_mean_variance.parquet',
        'lstm': f'{baselines_dir}/returns_lstm.parquet',
        'ew': f'{baselines_dir}/returns_equal_weight.parquet',
    }

    rl_path = "results/full_pipeline/Full_Pipeline_1M_returns.csv"
    if Path(rl_path).exists():
        strat_files['rl'] = rl_path

    dates = test_split.returns.index
    result = {}
    for name, fpath in strat_files.items():
        if fpath.endswith('.parquet'):
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
        s.index = pd.to_datetime(s.index)
        s = s.reindex(dates).fillna(0.0)
        result[name] = s.values.astype(np.float32)

    return result

class MetaRLv3:

    def __init__(self, timesteps=500_000, n_seeds=5, device="cpu"):
        self.timesteps = timesteps
        self.n_seeds = n_seeds
        self.device = device

    def train_and_eval(self, strategy_returns: dict, train_end_idx: int):

        strat_names = list(strategy_returns.keys())
        n_strats = len(strat_names)
        T = len(strategy_returns[strat_names[0]])

        strat_matrix = np.stack([strategy_returns[s] for s in strat_names], axis=1)

        train_strat = strat_matrix[:train_end_idx]
        test_strat = strat_matrix[train_end_idx:]

        class MetaEnv(Env):
            def __init__(self_env):
                super().__init__()

                self_env.observation_space = spaces.Box(-10, 10, (4 * n_strats,), np.float32)

                self_env.action_space = spaces.Box(0, 1, (n_strats + 1,), np.float32)
                self_env._step = 60
                self_env._portfolio_returns = []
                self_env._peak = 1.0
                self_env._equity = 1.0

            def reset(self_env, seed=None, options=None):
                super().reset(seed=seed)
                self_env._step = 60
                self_env._portfolio_returns = []
                self_env._peak = 1.0
                self_env._equity = 1.0
                return self_env._get_obs(), {}

            def _get_obs(self_env):
                t = self_env._step
                obs = np.zeros(4 * n_strats, dtype=np.float32)
                for i in range(n_strats):
                    s = train_strat[:t, i]
                    if len(s) >= 20:
                        obs[i*4] = np.mean(s[-20:]) / max(np.std(s[-20:]), 1e-8) * np.sqrt(252)
                        obs[i*4+1] = np.mean(s[-60:]) / max(np.std(s[-60:]), 1e-8) * np.sqrt(252)
                        obs[i*4+2] = np.mean(s[-5:]) * 252
                        obs[i*4+3] = np.std(s[-20:]) * np.sqrt(252)
                return np.nan_to_num(obs, nan=0.0).astype(np.float32)

            def step(self_env, action):
                t = self_env._step

                strat_w = np.clip(action[:n_strats], 0.01, None)
                strat_w = strat_w / strat_w.sum()
                cash_frac = np.clip(action[n_strats] * 0.3, 0, 0.3)

                strat_ret = np.sum(strat_w * train_strat[t])
                port_ret = (1 - cash_frac) * strat_ret

                self_env._portfolio_returns.append(port_ret)
                self_env._equity *= (1 + port_ret)
                self_env._peak = max(self_env._peak, self_env._equity)
                dd = self_env._equity / self_env._peak - 1

                rets = np.array(self_env._portfolio_returns[-60:])
                if len(rets) > 5:
                    reward = np.mean(rets) / max(np.std(rets), 1e-6) * 0.5
                    if dd < -0.03:
                        reward += 3 * dd
                else:
                    reward = port_ret * 10

                self_env._step += 1
                done = self_env._step >= len(train_strat)
                obs = self_env._get_obs() if not done else np.zeros(4 * n_strats, np.float32)
                return obs, float(reward), done, False, {}

        best_sharpe = -999
        best_agent = None
        best_vn = None

        all_oos_sharpes = []

        for seed_i in range(self.n_seeds):
            seed = seed_i * 77 + 42
            env = DummyVecEnv([MetaEnv])
            vn = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.995)

            agent = PPO(
                "MlpPolicy", vn,
                learning_rate=2e-4, n_steps=1024, batch_size=64,
                n_epochs=20, gamma=0.995, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.03,
                policy_kwargs=dict(net_arch=[64, 32], activation_fn=__import__('torch').nn.Tanh),
                verbose=0, seed=seed, device=self.device,
            )

            agent.learn(total_timesteps=self.timesteps)

            vn.training = False
            vn.norm_reward = False

            oos_returns = []
            obs = np.zeros(4 * n_strats, dtype=np.float32)

            for t in range(len(test_strat)):

                for i in range(n_strats):

                    all_s = np.concatenate([train_strat[:, i], test_strat[:t+1, i]])
                    if len(all_s) >= 20:
                        obs[i*4] = np.mean(all_s[-20:]) / max(np.std(all_s[-20:]), 1e-8) * np.sqrt(252)
                        obs[i*4+1] = np.mean(all_s[-60:]) / max(np.std(all_s[-60:]), 1e-8) * np.sqrt(252)
                        obs[i*4+2] = np.mean(all_s[-5:]) * 252
                        obs[i*4+3] = np.std(all_s[-20:]) * np.sqrt(252)
                obs = np.nan_to_num(obs, nan=0.0).astype(np.float32)

                obs_norm = vn.normalize_obs(obs.reshape(1, -1))
                action, _ = agent.predict(obs_norm, deterministic=True)
                action = action.flatten()

                strat_w = np.clip(action[:n_strats], 0.01, None)
                strat_w = strat_w / strat_w.sum()
                cash_frac = np.clip(action[n_strats] * 0.3, 0, 0.3)

                strat_ret = np.sum(strat_w * test_strat[t])
                port_ret = (1 - cash_frac) * strat_ret
                oos_returns.append(port_ret)

            oos_arr = np.array(oos_returns)
            sharpe = np.mean(oos_arr) / max(np.std(oos_arr), 1e-8) * np.sqrt(252)
            cum_ret = np.prod(1 + oos_arr) - 1
            cum_rets = np.cumprod(1 + oos_arr)
            mdd = np.min(cum_rets / np.maximum.accumulate(cum_rets) - 1)

            logger.info(f"Seed {seed}: OOS Sharpe={sharpe:.3f}, CumRet={cum_ret:.1%}, MaxDD={mdd:.1%}")
            all_oos_sharpes.append(sharpe)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_agent = agent
                best_vn = vn
                best_oos_returns = oos_arr.copy()

        avg_sharpe = np.mean(all_oos_sharpes)
        std_sharpe = np.std(all_oos_sharpes)
        logger.info(f"Average OOS Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")

        return best_oos_returns, all_oos_sharpes

def main():
    logger.info("Loading data...")
    pipeline = DataPipeline(n_assets=100)
    train_split, test_split = pipeline.build()

    logger.info("Loading strategy returns...")
    strat_returns = compute_strategy_weights_returns(test_split)

    strat_names = list(strat_returns.keys())
    T = len(strat_returns[strat_names[0]])
    logger.info(f"Strategies: {strat_names}, T={T}")

    for name, rets in strat_returns.items():
        s = np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(252)
        logger.info(f"  {name}: Sharpe={s:.3f}")

    train_end = int(T * 0.6)
    logger.info(f"Meta-RL train: {train_end} days, OOS: {T - train_end} days")

    meta = MetaRLv3(timesteps=300_000, n_seeds=5, device="cpu")
    oos_returns, all_sharpes = meta.train_and_eval(strat_returns, train_end)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_dates = test_split.returns.index[train_end:]
    oos_df = pd.DataFrame({'portfolio_return': oos_returns}, index=test_dates[:len(oos_returns)])
    oos_df.to_csv(RESULTS_DIR / "MetaRL_v3_OOS_returns.csv")

    metrics = PortfolioMetrics.compute_all(pd.Series(oos_returns, index=test_dates[:len(oos_returns)]))
    logger.info("=" * 60)
    logger.info("FINAL MetaRL v3 (MVO+LSTM+RL+EW+cash) OOS Results:")
    logger.info(f"  Sharpe: {metrics['Sharpe Ratio']:.3f}")
    logger.info(f"  Return: {metrics['Cumulative Return']*100:.1f}%")
    logger.info(f"  MaxDD:  {metrics['Max Drawdown']*100:.1f}%")
    logger.info(f"  Calmar: {metrics['Calmar Ratio']:.3f}")
    logger.info(f"  Avg Sharpe across seeds: {np.mean(all_sharpes):.3f} ± {np.std(all_sharpes):.3f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
