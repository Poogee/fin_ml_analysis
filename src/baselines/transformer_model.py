import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.lstm_model import build_sequences_vectorized, _build_asset_features

class _PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class _TemporalTransformer(nn.Module):

    def __init__(self, input_size: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        last = x[:, -1, :]
        return self.output_head(last)

class TransformerModel:

    N_FEATURES = 3

    def __init__(self, seq_len: int = 20, top_k: int = 50,
                 max_weight: float = 0.05, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1,
                 epochs: int = 10, batch_size: int = 512,
                 learning_rate: float = 1e-3, min_history: int = 60,
                 max_train_samples: int = 300_000):
        self.seq_len = seq_len
        self.top_k = top_k
        self.max_weight = max_weight
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_history = min_history
        self.max_train_samples = max_train_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, train_data: dict) -> None:

        returns = train_data["returns"]
        X, y = build_sequences_vectorized(
            returns, self.seq_len, self.max_train_samples
        )

        if len(X) < 100:
            self.model = None
            return

        y = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))

        self._feat_mean = X.reshape(-1, self.N_FEATURES).mean(axis=0)
        self._feat_std = X.reshape(-1, self.N_FEATURES).std(axis=0)
        self._feat_std[self._feat_std < 1e-10] = 1.0
        X = (X - self._feat_mean) / self._feat_std

        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _TemporalTransformer(
            input_size=self.N_FEATURES,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

    def predict_weights(self, current_data: dict) -> np.ndarray:

        returns = current_data["returns"]
        n_assets = returns.shape[1]
        weights = np.zeros(n_assets)

        if self.model is None:
            mask = ~np.isnan(returns.iloc[-1].values)
            if mask.sum() > 0:
                weights[mask] = 1.0 / mask.sum()
            return weights

        seq_len = min(self.seq_len, len(returns))
        window = returns.iloc[-seq_len:].values
        valid_count = np.sum(~np.isnan(window), axis=0)
        valid_mask = valid_count >= (seq_len // 2)

        if "presence_mask" in current_data:
            mask = current_data["presence_mask"].iloc[-1].values.astype(bool)
            valid_mask = valid_mask & mask

        if valid_mask.sum() == 0:
            return weights

        valid_idx = np.where(valid_mask)[0]
        w = np.nan_to_num(window, nan=0.0)

        X_list = []
        for ai in valid_idx:
            X_list.append(_build_asset_features(w[:, ai]))
        X = np.array(X_list)

        if hasattr(self, "_feat_mean"):
            X = (X - self._feat_mean) / self._feat_std

        X_t = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t).squeeze(1).cpu().numpy()

        k = min(self.top_k, len(valid_idx))
        top_indices = np.argsort(preds)[-k:]
        selected = valid_idx[top_indices]

        w_val = min(1.0 / k, self.max_weight)
        weights[selected] = w_val
        if weights.sum() > 0:
            weights /= weights.sum()
        return weights
