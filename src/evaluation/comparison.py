import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from pathlib import Path

from src.evaluation.metrics import PortfolioMetrics
from src.evaluation.backtester import BacktestResult

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

class ModelComparison:

    MODEL_ORDER = [
        "RL Full (PPO+Graph+TDA+NLP)",
        "RL No Graph Features",
        "RL No Sentiment",
        "XGBoost",
        "LightGBM",
        "LSTM",
        "Transformer",
        "Mean-Variance (Markowitz)",
        "Equal Weight",
        "Risk Parity",
    ]

    MODEL_COLORS = {
        "RL Full (PPO+Graph+TDA+NLP)": "#e63946",
        "RL No Graph Features": "#f4a261",
        "RL No Sentiment": "#e9c46a",
        "XGBoost": "#2a9d8f",
        "LightGBM": "#43aa8b",
        "LSTM": "#264653",
        "Transformer": "#6a4c93",
        "Mean-Variance (Markowitz)": "#1d3557",
        "Equal Weight": "#a8dadc",
        "Risk Parity": "#457b9d",
    }

    def __init__(self, results: dict[str, BacktestResult],
                 output_dir: str = "results"):

        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)

    def _get_color(self, name: str) -> str:

        if name in self.MODEL_COLORS:
            return self.MODEL_COLORS[name]

        import colorsys
        h = (hash(name) % 360) / 360.0
        r, g, b = colorsys.hls_to_rgb(h, 0.5, 0.6)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def _ordered_names(self) -> list[str]:

        ordered = [n for n in self.MODEL_ORDER if n in self.results]
        extras = [n for n in self.results if n not in ordered]
        return ordered + extras

    def metrics_table(self, risk_free_rate: float = 0.0) -> pd.DataFrame:

        returns_dict = {name: r.returns for name, r in self.results.items()}
        df = PortfolioMetrics.compute_all_df(returns_dict, risk_free_rate)

        ordered = self._ordered_names()
        df = df.loc[[n for n in ordered if n in df.index]]

        fmt_df = df.copy()
        pct_cols = ["Cumulative Return", "Annualized Return",
                    "Annualized Volatility", "Max Drawdown"]
        for col in pct_cols:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.2%}")
        ratio_cols = ["Sharpe Ratio", "Calmar Ratio", "Sortino Ratio"]
        for col in ratio_cols:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.3f}")

        ci_rows = {}
        for name in ordered:
            if name not in self.results:
                continue
            ci = PortfolioMetrics.bootstrap_sharpe_ci(self.results[name].returns)
            ci_rows[name] = {
                "Sharpe 95% CI": f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]",
                "Sharpe SE": ci["std_error"],
            }
        ci_df = pd.DataFrame(ci_rows).T
        fmt_df = pd.concat([fmt_df, ci_df], axis=1)

        df.to_csv(self.output_dir / "tables" / "metrics_comparison.csv")
        fmt_df.to_csv(self.output_dir / "tables" / "metrics_comparison_formatted.csv")

        ci_df.to_csv(self.output_dir / "tables" / "bootstrap_sharpe_ci.csv")

        return df

    def plot_equity_curves(self, title: str = "Cumulative Portfolio Returns",
                           log_scale: bool = False) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(14, 8))

        for name in self._ordered_names():
            result = self.results[name]
            color = self._get_color(name)
            lw = 2.5 if "RL Full" in name else 1.5
            ax.plot(result.equity_curve.index, result.equity_curve.values,
                    label=name, color=color, linewidth=lw)

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (starting at 1.0)")
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "equity_curves.png",
                    dpi=200, bbox_inches="tight")
        fig.savefig(self.output_dir / "plots" / "equity_curves.pdf",
                    bbox_inches="tight")
        return fig

    def plot_drawdowns(self) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(14, 6))

        for name in self._ordered_names():
            result = self.results[name]
            cumulative = (1 + result.returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            color = self._get_color(name)
            lw = 2.0 if "RL Full" in name else 1.2
            ax.plot(drawdown.index, drawdown.values,
                    label=name, color=color, linewidth=lw, alpha=0.8)

        ax.set_title("Portfolio Drawdowns", fontsize=16, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(loc="lower left", framealpha=0.9)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "drawdowns.png",
                    dpi=200, bbox_inches="tight")
        return fig

    def statistical_tests(self, window: int = 252,
                          baseline: str | None = None) -> pd.DataFrame:

        if baseline is None:
            baseline = "Equal Weight"
        if baseline not in self.results:

            baseline = list(self.results.keys())[0]

        baseline_sharpe = PortfolioMetrics.rolling_sharpe(
            self.results[baseline].returns, window=window
        )

        rows = []
        for name in self._ordered_names():
            if name == baseline:
                continue
            model_sharpe = PortfolioMetrics.rolling_sharpe(
                self.results[name].returns, window=window
            )

            common_idx = baseline_sharpe.index.intersection(model_sharpe.index)
            if len(common_idx) < 10:
                rows.append({
                    "Model": name,
                    "vs Baseline": baseline,
                    "t-statistic": np.nan,
                    "p-value": np.nan,
                    "Mean Diff": np.nan,
                    "Significant (5%)": False,
                })
                continue

            b = baseline_sharpe.loc[common_idx].values
            m = model_sharpe.loc[common_idx].values
            t_stat, p_val = stats.ttest_rel(m, b, nan_policy="omit")

            rows.append({
                "Model": name,
                "vs Baseline": baseline,
                "t-statistic": round(t_stat, 4),
                "p-value": round(p_val, 4),
                "Mean Diff": round(np.nanmean(m - b), 4),
                "Significant (5%)": p_val < 0.05,
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "tables" / "statistical_tests.csv", index=False)
        return df

    def plot_ablation(self) -> plt.Figure:

        ablation_models = {
            "RL Full (PPO+Graph+TDA+NLP)": "Full Model",
            "RL No Graph Features": "- Graph/TDA",
            "RL No Sentiment": "- Sentiment",
        }

        available = {k: v for k, v in ablation_models.items()
                     if k in self.results}
        if len(available) < 2:
            return plt.figure()

        metrics_to_show = ["Sharpe Ratio", "Calmar Ratio", "Cumulative Return"]
        data = {}
        for model_name, label in available.items():
            m = PortfolioMetrics.compute_all(self.results[model_name].returns)
            data[label] = {k: m[k] for k in metrics_to_show}

        df = pd.DataFrame(data).T

        fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(5 * len(metrics_to_show), 6))
        if len(metrics_to_show) == 1:
            axes = [axes]

        colors = ["#e63946", "#f4a261", "#e9c46a", "#2a9d8f"]
        for i, metric in enumerate(metrics_to_show):
            ax = axes[i]
            bars = ax.bar(df.index, df[metric], color=colors[:len(df)])
            ax.set_title(metric, fontsize=13, fontweight="bold")
            ax.set_ylabel(metric)

            for bar, val in zip(bars, df[metric]):
                fmt = f"{val:.2%}" if "Return" in metric else f"{val:.3f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt, ha="center", va="bottom", fontsize=10)
            ax.tick_params(axis="x", rotation=15)

        fig.suptitle("Ablation Analysis: Component Contributions",
                     fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "ablation_analysis.png",
                    dpi=200, bbox_inches="tight")
        return fig

    def plot_rolling_sharpe(self, window: int = 252) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(14, 7))

        for name in self._ordered_names():
            result = self.results[name]
            rs = PortfolioMetrics.rolling_sharpe(result.returns, window=window)
            color = self._get_color(name)
            lw = 2.0 if "RL Full" in name else 1.2
            ax.plot(rs.index, rs.values, label=name,
                    color=color, linewidth=lw, alpha=0.8)

        ax.set_title(f"Rolling Sharpe Ratio ({window}-day window)",
                     fontsize=16, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9)

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "rolling_sharpe.png",
                    dpi=200, bbox_inches="tight")
        return fig

    def plot_metrics_heatmap(self) -> plt.Figure:

        df = self.metrics_table()

        norm_df = df.copy()
        for col in norm_df.columns:
            col_range = norm_df[col].max() - norm_df[col].min()
            if col_range == 0:
                norm_df[col] = 0.5
            elif col == "Max Drawdown":
                norm_df[col] = 1 - (norm_df[col] - norm_df[col].min()) / col_range
            else:
                norm_df[col] = (norm_df[col] - norm_df[col].min()) / col_range

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(norm_df, annot=df.round(4), fmt="", cmap="RdYlGn",
                    ax=ax, linewidths=0.5, cbar_kws={"label": "Normalized Score"})
        ax.set_title("Model Comparison Heatmap", fontsize=16, fontweight="bold")
        ax.set_ylabel("")

        fig.tight_layout()
        fig.savefig(self.output_dir / "plots" / "metrics_heatmap.png",
                    dpi=200, bbox_inches="tight")
        return fig

    def run_full_comparison(self, risk_free_rate: float = 0.0) -> dict:

        print("Computing metrics table...")
        metrics_df = self.metrics_table(risk_free_rate)
        print(metrics_df.to_string())
        print()

        print("Plotting equity curves...")
        eq_fig = self.plot_equity_curves()

        print("Plotting drawdowns...")
        dd_fig = self.plot_drawdowns()

        print("Running statistical tests...")
        stat_df = self.statistical_tests()
        print(stat_df.to_string(index=False))
        print()

        print("Plotting ablation analysis...")
        abl_fig = self.plot_ablation()

        print("Plotting rolling Sharpe...")
        rs_fig = self.plot_rolling_sharpe()

        print("Plotting metrics heatmap...")
        hm_fig = self.plot_metrics_heatmap()

        plt.close("all")

        print(f"\nAll results saved to {self.output_dir}/")
        return {
            "metrics": metrics_df,
            "statistical_tests": stat_df,
            "figures": {
                "equity_curves": eq_fig,
                "drawdowns": dd_fig,
                "ablation": abl_fig,
                "rolling_sharpe": rs_fig,
                "heatmap": hm_fig,
            },
        }
