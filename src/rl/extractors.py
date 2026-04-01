from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def _parse_obs(
    observations: torch.Tensor,
    n_assets: int,
    lookback: int,
    n_channels: int,
    n_global: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    batch_size = observations.shape[0]
    obs_per_asset = lookback * n_channels

    asset_data = observations[:, : n_assets * obs_per_asset]
    prev_weights = observations[
        :, n_assets * obs_per_asset : n_assets * obs_per_asset + n_assets
    ]
    global_feats = observations[:, n_assets * obs_per_asset + n_assets :]

    asset_data = asset_data.view(batch_size, n_assets, n_channels, lookback)

    return asset_data, prev_weights, global_feats

class CNN1DExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        n_assets: int = 115,
        lookback: int = 10,
        n_pa_features: int = 13,
        n_global_features: int = 23,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        self.n_assets = n_assets
        self.lookback = lookback
        self.n_channels = 1 + n_pa_features
        self.n_global = n_global_features

        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        asset_embed_dim = 64 + 1

        self.head = nn.Sequential(
            nn.Linear(n_assets * asset_embed_dim + n_global_features, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        asset_data, prev_weights, global_feats = _parse_obs(
            observations, self.n_assets, self.lookback,
            self.n_channels, self.n_global,
        )

        x = asset_data.reshape(batch_size * self.n_assets, self.n_channels, self.lookback)
        x = self.temporal_cnn(x)
        x = x.squeeze(-1)
        x = x.view(batch_size, self.n_assets, 64)

        prev_w = prev_weights.unsqueeze(-1)
        x = torch.cat([x, prev_w], dim=-1)

        x = x.view(batch_size, -1)
        x = torch.cat([x, global_feats], dim=-1)

        return self.head(x)

class CNNAttentionExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        n_assets: int = 115,
        lookback: int = 10,
        n_pa_features: int = 13,
        n_global_features: int = 23,
        cnn_channels: int = 64,
        n_heads: int = 4,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        self.n_assets = n_assets
        self.lookback = lookback
        self.n_channels = 1 + n_pa_features
        self.n_global = n_global_features

        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        embed_dim = cnn_channels + 1

        if embed_dim % n_heads != 0:
            self._pad_dim = n_heads - (embed_dim % n_heads)
            embed_dim += self._pad_dim
            self.embed_pad = nn.Linear(cnn_channels + 1, embed_dim)
        else:
            self._pad_dim = 0
            self.embed_pad = None

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=0.1, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(n_assets * embed_dim + n_global_features, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
        self._embed_dim = embed_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        asset_data, prev_weights, global_feats = _parse_obs(
            observations, self.n_assets, self.lookback,
            self.n_channels, self.n_global,
        )

        x = asset_data.reshape(
            batch_size * self.n_assets, self.n_channels, self.lookback,
        )
        x = self.temporal_cnn(x).squeeze(-1)
        x = x.view(batch_size, self.n_assets, -1)

        prev_w = prev_weights.unsqueeze(-1)
        x = torch.cat([x, prev_w], dim=-1)

        if self.embed_pad is not None:
            x = self.embed_pad(x)

        attn_out, _ = self.cross_attn(x, x, x)
        x = self.layer_norm(x + attn_out)

        ff_out = self.ff(x)
        x = self.layer_norm2(x + ff_out)

        x = x.view(batch_size, -1)
        x = torch.cat([x, global_feats], dim=-1)

        return self.head(x)

class ASPPBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        atrous_rates: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=3,
                    padding=rate, dilation=rate,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
            for rate in atrous_rates
        ])

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        n_branches = len(atrous_rates) + 1
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * n_branches, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        branch_outs = [branch(x) for branch in self.branches]

        gp = self.global_pool(x)
        gp = gp.expand_as(branch_outs[0])
        branch_outs.append(gp)

        x = torch.cat(branch_outs, dim=1)
        return self.project(x)

class ASPPExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        n_assets: int = 115,
        lookback: int = 10,
        n_pa_features: int = 13,
        n_global_features: int = 23,
        aspp_out_channels: int = 64,
        atrous_rates: tuple[int, ...] = (1, 2, 4, 8),
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        self.n_assets = n_assets
        self.lookback = lookback
        self.n_channels = 1 + n_pa_features
        self.n_global = n_global_features

        self.input_conv = nn.Sequential(
            nn.Conv1d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.aspp = ASPPBlock(32, aspp_out_channels, atrous_rates)

        self.pool = nn.AdaptiveAvgPool1d(1)

        asset_embed_dim = aspp_out_channels + 1

        self.head = nn.Sequential(
            nn.Linear(n_assets * asset_embed_dim + n_global_features, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        asset_data, prev_weights, global_feats = _parse_obs(
            observations, self.n_assets, self.lookback,
            self.n_channels, self.n_global,
        )

        x = asset_data.reshape(
            batch_size * self.n_assets, self.n_channels, self.lookback,
        )

        x = self.input_conv(x)
        x = self.aspp(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = x.view(batch_size, self.n_assets, -1)

        prev_w = prev_weights.unsqueeze(-1)
        x = torch.cat([x, prev_w], dim=-1)

        x = x.view(batch_size, -1)
        x = torch.cat([x, global_feats], dim=-1)

        return self.head(x)

class ManualGATLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.out_per_head = out_features // n_heads

        self.W = nn.Linear(in_features, n_heads * self.out_per_head, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.out_per_head) * 0.01)
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.out_per_head) * 0.01)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        B, N, _ = x.shape
        h = self.W(x)
        h = h.view(B, N, self.n_heads, self.out_per_head)

        src_scores = (h * self.a_src.unsqueeze(0).unsqueeze(0)).sum(-1)
        dst_scores = (h * self.a_dst.unsqueeze(0).unsqueeze(0)).sum(-1)

        e = src_scores.unsqueeze(2) + dst_scores.unsqueeze(1)
        e = self.leaky_relu(e)

        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(mask, float("-inf"))

        alpha = torch.softmax(e, dim=2)
        alpha = alpha.nan_to_num(0)

        h_perm = h.permute(0, 2, 1, 3)
        alpha_perm = alpha.permute(0, 3, 1, 2)

        out = torch.matmul(alpha_perm, h_perm)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)

        return out

class GATExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        n_assets: int = 115,
        lookback: int = 10,
        n_pa_features: int = 13,
        n_global_features: int = 23,
        adj_matrix: torch.Tensor | None = None,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        self.n_assets = n_assets
        self.lookback = lookback
        self.n_channels = 1 + n_pa_features
        self.n_global = n_global_features

        if adj_matrix is not None:
            self.register_buffer("adj", adj_matrix)
        else:
            self.register_buffer("adj", torch.ones(n_assets, n_assets))

        node_input_dim = self.lookback * self.n_channels + 1
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, 64),
            nn.ReLU(),
        )

        self.gat1 = ManualGATLayer(64, 64, n_heads=4)
        self.gat2 = ManualGATLayer(64, 32, n_heads=4)
        self.relu = nn.ReLU()

        self.head = nn.Sequential(
            nn.Linear(n_assets * 32 + n_global_features, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        asset_data, prev_weights, global_feats = _parse_obs(
            observations, self.n_assets, self.lookback,
            self.n_channels, self.n_global,
        )

        x = asset_data.reshape(batch_size, self.n_assets, self.n_channels * self.lookback)
        prev_w = prev_weights.unsqueeze(-1)
        x = torch.cat([x, prev_w], dim=-1)

        x = self.node_proj(x)

        h = self.relu(self.gat1(x, self.adj))
        x = self.gat2(h, self.adj)

        x = x.view(batch_size, -1)
        x = torch.cat([x, global_feats], dim=-1)

        return self.head(x)

def build_knn_adjacency(
    returns: torch.Tensor | None = None,
    corr_matrix: torch.Tensor | None = None,
    k: int = 10,
    n_assets: int | None = None,
) -> torch.Tensor:

    if corr_matrix is None and returns is not None:
        import numpy as np
        if isinstance(returns, torch.Tensor):
            r = returns.cpu().numpy()
        else:
            r = returns
        corr = np.corrcoef(r.T)
        corr = np.nan_to_num(corr, nan=0.0)
        corr_matrix = torch.tensor(corr, dtype=torch.float32)

    if corr_matrix is None:
        if n_assets is not None:
            return torch.ones(n_assets, n_assets)
        raise ValueError("Must provide returns, corr_matrix, or n_assets")

    N = corr_matrix.shape[0]

    sim = corr_matrix.abs()

    sim.fill_diagonal_(0)

    _, topk_idx = sim.topk(min(k, N - 1), dim=1)
    adj = torch.zeros(N, N)
    for i in range(N):
        adj[i, topk_idx[i]] = 1.0

    adj = ((adj + adj.T) > 0).float()
    adj.fill_diagonal_(1.0)

    return adj
