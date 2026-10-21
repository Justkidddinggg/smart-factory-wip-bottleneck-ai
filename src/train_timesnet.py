"""
========================================================================================
🚀 SMART FACTORY AI: 24-HOUR MULTI-HORIZON WIP FORECASTING (TIMESNET + REVIN)
========================================================================================
Architecture: TimesNet (ICLR 2023) with Reversible Instance Normalization (RevIN)
Transforms 1D time-series into 2D intra-/inter-period tensor blocks for multi-period capture.
========================================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.fft
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class RevIN(nn.Module):
    """Reversible Instance Normalization (ICLR 2022) to handle non-stationary distribution shift."""
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            x = x * self.stdev + self.mean
            return x


class TimesBlock(nn.Module):
    """Captures 2D multi-periodicity through FFT top-k period decomposition and Inception-style 2D conv."""
    def __init__(self, seq_len: int, pred_len: int, top_k: int = 3, d_model: int = 32, d_ff: int = 64):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, kernel_size=(3, 3), padding=1)
        )

    def forward(self, x):
        B, T, N = x.size()
        # FFT to extract dominant intra-day / inter-day frequencies
        xf = torch.fft.rfft(x, dim=1)
        freq_amps = torch.abs(xf).mean(dim=-1).mean(dim=0)
        freq_amps[0] = 0.0  # Ignore DC component
        _, top_indices = torch.topk(freq_amps, self.top_k)
        periods = torch.clamp((T // top_indices), min=2, max=T // 2)

        out_list = []
        for period in periods:
            p = int(period.item())
            length = ((T + p - 1) // p) * p
            padding = torch.zeros(B, length - T, N, device=x.device)
            padded = torch.cat([x, padding], dim=1)
            # Reshape 1D into 2D tensor (Batch, d_model, Period, Length//Period)
            x_2d = padded.reshape(B, length // p, p, N).permute(0, 3, 1, 2)
            conv_out = self.conv(x_2d)
            back_1d = conv_out.permute(0, 2, 3, 1).reshape(B, -1, N)[:, :T, :]
            out_list.append(back_1d)

        return torch.stack(out_list, dim=-1).mean(dim=-1) + x


class TimesNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, seq_len: int = 72, pred_len: int = 24, d_model: int = 32):
        super(TimesNet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.revin = RevIN(in_dim)
        self.embedding = nn.Linear(in_dim, d_model)
        self.block1 = TimesBlock(seq_len, pred_len, top_k=3, d_model=d_model)
        self.block2 = TimesBlock(seq_len, pred_len, top_k=2, d_model=d_model)
        self.projection = nn.Linear(seq_len, pred_len)
        self.out_layer = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # 1. Reversible Normalization
        x_norm = self.revin(x, mode="norm")
        # 2. Project to feature space
        emb = self.embedding(x_norm)
        # 3. 2D Periodic Transformation
        feat = self.block1(emb)
        feat = self.block2(feat)
        # 4. Temporal Projection across 24 horizons
        out = self.projection(feat.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.out_layer(out)
        # 5. Denormalize back to actual physical scale
        pred = self.revin(out, mode="denorm")
        return pred


def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def run_timesnet_forecasting():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    csv_path = os.path.join(base_dir, "data", "mock_sample_timeseries.csv")

    if not os.path.exists(csv_path):
        from data.generate_mock_data import generate_synthetic_factory_data
        generate_synthetic_factory_data(output_path=csv_path)

    df = pd.read_csv(csv_path)
    target_cols = ["Total_WIP", "WIP_Process_A", "WIP_Process_B", "WIP_Process_C", "WIP_Process_D", "WIP_Process_E"]
    values = df[target_cols].values.astype(np.float32)

    seq_len = 72   # 3-Day historical lookback
    pred_len = 24  # 24-Hour forecast horizon

    # Create sliding windows
    X, Y = [], []
    for i in range(len(values) - seq_len - pred_len + 1):
        X.append(values[i:i + seq_len])
        Y.append(values[i + seq_len:i + seq_len + pred_len])
    X, Y = np.array(X), np.array(Y)

    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)

    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size + val_size:], Y[train_size + val_size:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesNet(in_dim=len(target_cols), out_dim=len(target_cols), seq_len=seq_len, pred_len=pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.L1Loss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train).to(device), torch.tensor(Y_train).to(device)),
        batch_size=32, shuffle=True
    )

    print(f"🚀 Training TimesNet (Seq={seq_len}h -> Pred={pred_len}h) on {len(X_train)} windows | Device: {device}")
    model.train()
    for epoch in range(1, 11):
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch {epoch:02d}/10 | L1 Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(X_test).to(device)).cpu().numpy()

    print("\n" + "=" * 80)
    print("🏆 TIMESNET 24-HOUR WIP FORECASTING BENCHMARK (TEST SET)")
    print("=" * 80)
    print(f"{'Station':<16} | {'WMAPE (%)':>12} | {'Accuracy (%)':>14} | {'MAE':>10}")
    print("-" * 80)

    for idx, col in enumerate(target_cols):
        true_col = Y_test[:, :, idx]
        pred_col = test_pred[:, :, idx]
        wmape = calculate_wmape(true_col, pred_col)
        acc = max(0.0, 100.0 - wmape)
        mae = np.mean(np.abs(true_col - pred_col))
        st_name = col.replace("WIP_", "")
        print(f"{st_name:<16} | {wmape:>11.2f}% | {acc:>13.2f}% | {mae:>10.1f}")
    print("=" * 80)


if __name__ == "__main__":
    run_timesnet_forecasting()
