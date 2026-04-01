import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class _LSTMNet(nn.Module):

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

def _build_asset_features(ret_seq: np.ndarray) -> np.ndarray:

    cum_ret = np.cumsum(ret_seq)

    vol = np.zeros_like(ret_seq)
    for s in range(len(ret_seq)):
        start = max(0, s - 4)
        vol[s] = np.std(ret_seq[start:s + 1]) if s > 0 else 0.0
    return np.column_stack([ret_seq, cum_ret, vol])

def build_sequences_vectorized(returns: pd.DataFrame, seq_len: int = 20,
                                max_samples: int = 300_000
                                ) -> tuple[np.ndarray, np.ndarray]:

    ret_vals = returns.values
    n_dates, n_assets = ret_vals.shape

    if n_dates <= seq_len + 1:
        return np.empty((0, seq_len, 3)), np.empty(0)

    all_X = []
    all_y = []

    for t in range(seq_len, n_dates - 1):
        window = ret_vals[t - seq_len:t]
        target = ret_vals[t]

        valid_count = np.sum(~np.isnan(window), axis=0)
        valid = (valid_count >= seq_len // 2) & ~np.isnan(target)

        if valid.sum() < 5:
            continue

        valid_idx = np.where(valid)[0]
        w = np.nan_to_num(window, nan=0.0)

        batch_feats = np.zeros((len(valid_idx), seq_len, 3))
        for i, ai in enumerate(valid_idx):
            batch_feats[i] = _build_asset_features(w[:, ai])

        all_X.append(batch_feats)
        all_y.append(target[valid_idx])

    if not all_X:
        return np.empty((0, seq_len, 3)), np.empty(0)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    return X, y

class LSTMModel:

    N_FEATURES = 3

    def __init__(self, seq_len: int = 20, top_k: int = 50,
                 max_weight: float = 0.05, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 epochs: int = 10, batch_size: int = 512,
                 learning_rate: float = 1e-3, min_history: int = 60,
                 max_train_samples: int = 300_000):
        self.seq_len = seq_len
        self.top_k = top_k
        self.max_weight = max_weight
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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

        self.model = _LSTMNet(
            input_size=self.N_FEATURES,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
