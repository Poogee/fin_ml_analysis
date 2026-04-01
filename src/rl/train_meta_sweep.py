import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import Env, spaces
import torch

from src.data.pipeline import DataPipeline
from src.evaluation.metrics import PortfolioMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

RESULTS_DIR = Path("results/meta_sweep")

def load_all_strategies():

    strats = {}
    files = {
        'mvo': 'results/baselines/returns_mean_variance.parquet',
        'lstm': 'results/baselines/returns_lstm.parquet',
        'ew': 'results/baselines/returns_equal_weight.parquet',
        'rp': 'results/baselines/returns_risk_parity.parquet',
        'rl': 'results/full_pipeline/Full_Pipeline_1M_returns.csv',
        'xgb': 'results/baselines/returns_xgboost.parquet',
        'transformer': 'results/baselines/returns_transformer.parquet',
    }
    for name, fpath in files.items():
        if not Path(fpath).exists():
            continue
        if fpath.endswith('.parquet'):
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
        s.index = pd.to_datetime(s.index)
        strats[name] = s
    return strats

def align_strategies(strats: dict) -> tuple:

    common_idx = None
    for s in strats.values():
        if common_idx is None:
            common_idx = s.index
        else:
            common_idx = common_idx.intersection(s.index)
    common_idx = common_idx.sort_values()
    aligned = {k: v.reindex(common_idx).fillna(0).values.astype(np.float32)
               for k, v in strats.items()}
    return aligned, common_idx

def train_meta_rl(strat_matrix, strat_names, train_end, n_seeds=7, timesteps=500_000,
                  use_cash=True, device="cpu"):

    n_strats = len(strat_names)
    T = strat_matrix.shape[0]
    train_strat = strat_matrix[:train_end]
    test_strat = strat_matrix[train_end:]

    act_dim = n_strats + (1 if use_cash else 0)
    obs_dim = 4 * n_strats

    class MetaEnv(Env):
        def __init__(self_env):
            super().__init__()
            self_env.observation_space = spaces.Box(-10, 10, (obs_dim,), np.float32)
            self_env.action_space = spaces.Box(0, 1, (act_dim,), np.float32)
            self_env._step = 60
            self_env._rets = []
            self_env._peak = 1.0
            self_env._eq = 1.0

        def reset(self_env, seed=None, options=None):
            super().reset(seed=seed)
            self_env._step = 60
            self_env._rets = []
            self_env._peak = 1.0
            self_env._eq = 1.0
            return self_env._get_obs(), {}

        def _get_obs(self_env):
            t = self_env._step
            obs = np.zeros(obs_dim, dtype=np.float32)
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
            w = np.clip(action[:n_strats], 0.01, None)
            w = w / w.sum()
            if use_cash:
                cash = np.clip(action[n_strats] * 0.3, 0, 0.3)
            else:
                cash = 0.0

            port_ret = (1 - cash) * np.sum(w * train_strat[t])
            self_env._rets.append(port_ret)
            self_env._eq *= (1 + port_ret)
            self_env._peak = max(self_env._peak, self_env._eq)
            dd = self_env._eq / self_env._peak - 1

            rets = np.array(self_env._rets[-60:])
            if len(rets) > 5:
                reward = np.mean(rets) / max(np.std(rets), 1e-6) * 0.5
                if dd < -0.03:
                    reward += 3 * dd
            else:
                reward = port_ret * 10

            self_env._step += 1
            done = self_env._step >= len(train_strat)
            obs = self_env._get_obs() if not done else np.zeros(obs_dim, np.float32)
            return obs, float(reward), done, False, {}

    all_sharpes = []
    best_sharpe = -999
    best_oos = None

    for seed_i in range(n_seeds):
        seed = seed_i * 77 + 42
        env = DummyVecEnv([MetaEnv])
        vn = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.995)

        agent = PPO(
            "MlpPolicy", vn,
            learning_rate=2e-4, n_steps=1024, batch_size=64,
            n_epochs=20, gamma=0.995, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.03,
            policy_kwargs=dict(net_arch=[64, 32], activation_fn=torch.nn.Tanh),
            verbose=0, seed=seed, device=device,
        )
        agent.learn(total_timesteps=timesteps)

        vn.training = False
        vn.norm_reward = False
        oos_rets = []
        obs = np.zeros(obs_dim, dtype=np.float32)

        for t in range(len(test_strat)):
            for i in range(n_strats):
                all_s = np.concatenate([train_strat[:, i], test_strat[:t+1, i]])
                if len(all_s) >= 20:
                    obs[i*4] = np.mean(all_s[-20:]) / max(np.std(all_s[-20:]), 1e-8) * np.sqrt(252)
                    obs[i*4+1] = np.mean(all_s[-60:]) / max(np.std(all_s[-60:]), 1e-8) * np.sqrt(252)
                    obs[i*4+2] = np.mean(all_s[-5:]) * 252
                    obs[i*4+3] = np.std(all_s[-20:]) * np.sqrt(252)
            obs = np.nan_to_num(obs, nan=0.0).astype(np.float32)
            obs_n = vn.normalize_obs(obs.reshape(1, -1))
            action, _ = agent.predict(obs_n, deterministic=True)
            action = action.flatten()

            w = np.clip(action[:n_strats], 0.01, None)
            w = w / w.sum()
            cash = np.clip(action[n_strats] * 0.3, 0, 0.3) if use_cash else 0.0
            port_ret = (1 - cash) * np.sum(w * test_strat[t])
            oos_rets.append(port_ret)

        oos_arr = np.array(oos_rets)
        sharpe = np.mean(oos_arr) / max(np.std(oos_arr), 1e-8) * np.sqrt(252)
        cum = np.prod(1 + oos_arr) - 1
        cum_c = np.cumprod(1 + oos_arr)
        mdd = np.min(cum_c / np.maximum.accumulate(cum_c) - 1)
        all_sharpes.append(sharpe)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_oos = oos_arr.copy()

        logger.info(f"  Seed {seed}: Sharpe={sharpe:.3f}, Ret={cum:.1%}, MDD={mdd:.1%}")

    return np.mean(all_sharpes), np.std(all_sharpes), best_sharpe, best_oos, all_sharpes

def main():
    logger.info("Loading strategies...")
    strats = load_all_strategies()
    aligned, dates = align_strategies(strats)
    T = len(dates)
    train_end = int(T * 0.6)

    logger.info(f"Total days: {T}, train: {train_end}, OOS: {T - train_end}")
    for name, rets in aligned.items():
        s = np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(252)
        logger.info(f"  {name}: Sharpe={s:.3f} (full), OOS={np.mean(rets[train_end:]) / max(np.std(rets[train_end:]), 1e-8) * np.sqrt(252):.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = [

        ("MVO+LSTM+cash", ['mvo', 'lstm'], True, 500_000),
        ("MVO+LSTM+RL+cash", ['mvo', 'lstm', 'rl'], True, 500_000),
        ("MVO+LSTM+EW+cash", ['mvo', 'lstm', 'ew'], True, 500_000),
        ("MVO+LSTM+RL+EW+cash", ['mvo', 'lstm', 'rl', 'ew'], True, 500_000),
        ("MVO+LSTM", ['mvo', 'lstm'], False, 500_000),
        ("MVO+LSTM+RL", ['mvo', 'lstm', 'rl'], False, 500_000),
        ("ALL+cash", ['mvo', 'lstm', 'rl', 'ew', 'rp', 'xgb', 'transformer'], True, 500_000),
        ("MVO+LSTM+cash_1M", ['mvo', 'lstm'], True, 1_000_000),
        ("MVO+LSTM+RL+cash_1M", ['mvo', 'lstm', 'rl'], True, 1_000_000),
    ]

    results = []
    for name, keys, use_cash, ts in configs:
        available = [k for k in keys if k in aligned]
        if len(available) < len(keys):
            logger.info(f"SKIP {name}: missing {set(keys) - set(available)}")
            continue

        strat_matrix = np.stack([aligned[k] for k in available], axis=1)
        logger.info(f"\n{'='*60}")
        logger.info(f"CONFIG: {name} ({available}, cash={use_cash}, ts={ts})")
        logger.info(f"{'='*60}")

        avg_s, std_s, best_s, best_oos, all_s = train_meta_rl(
            strat_matrix, available, train_end,
            n_seeds=7, timesteps=ts, use_cash=use_cash
        )

        oos_dates = dates[train_end:train_end + len(best_oos)]
        pd.DataFrame({'portfolio_return': best_oos}, index=oos_dates).to_csv(
            RESULTS_DIR / f"{name.replace('+', '_')}_returns.csv"
        )

        results.append({
            'config': name, 'strategies': '+'.join(available),
            'cash': use_cash, 'timesteps': ts,
            'avg_sharpe': avg_s, 'std_sharpe': std_s, 'best_sharpe': best_s,
            'all_sharpes': all_s,
        })
        logger.info(f">>> {name}: avg={avg_s:.3f}±{std_s:.3f}, best={best_s:.3f}")

    print("\n" + "=" * 90)
    print(f"{'Config':<30s} | {'Avg Sharpe':>12s} | {'Std':>6s} | {'Best':>6s} | {'Cash':>5s} | {'Steps':>7s}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: x['avg_sharpe'], reverse=True):
        print(f"{r['config']:<30s} | {r['avg_sharpe']:>12.3f} | {r['std_sharpe']:>6.3f} | "
              f"{r['best_sharpe']:>6.3f} | {str(r['cash']):>5s} | {r['timesteps']:>7d}")
    print("=" * 90)

    pd.DataFrame(results).to_csv(RESULTS_DIR / "sweep_summary.csv", index=False)

if __name__ == "__main__":
    main()
