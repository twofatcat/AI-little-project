import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1) Volatility estimators (no look-ahead)
# -----------------------------
def ewma_vol(returns: np.ndarray, alpha: float = 0.94, eps: float = 1e-8, demean: bool = True) -> np.ndarray:
    """EWMA volatility using only information up to time t (inclusive)."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = r.size
    if n == 0:
        return r

    if demean:
        mean = np.zeros(n, dtype=np.float64)
        var = np.zeros(n, dtype=np.float64)
        mean[0] = r[0]
        var[0] = max(r[0] * r[0], eps)
        for t in range(1, n):
            mean[t] = alpha * mean[t - 1] + (1 - alpha) * r[t]
            resid = r[t] - mean[t]
            var[t] = alpha * var[t - 1] + (1 - alpha) * resid * resid
            if var[t] < eps:
                var[t] = eps
    else:
        var = np.zeros(n, dtype=np.float64)
        var[0] = max(r[0] * r[0], eps)
        for t in range(1, n):
            var[t] = alpha * var[t - 1] + (1 - alpha) * r[t] * r[t]
            if var[t] < eps:
                var[t] = eps

    return np.sqrt(var)


def rolling_vol(returns: np.ndarray, window: int = 50, eps: float = 1e-8) -> np.ndarray:
    """Rolling std volatility. sigma_t uses returns up to t (inclusive)."""
    r = pd.Series(np.asarray(returns, dtype=np.float64).reshape(-1))
    sigma = r.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()
    return np.maximum(sigma, eps)


# -----------------------------
# 2) RevIN for RETURNS CHANNEL ONLY
# -----------------------------
class RevIN1D(nn.Module):
    """Reversible Instance Normalization for a single feature channel.

    We only apply RevIN to the returns channel r (shape B,L,1). The (global) sigma channel
    coming from the dataset is *not* normalized.

    Notes:
      - We cache per-batch mean/std for denormalization.
      - When affine=True, we implement a sign-preserving inverse so that gamma<0 does not
        break reversibility.
    """

    def __init__(self, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self._mean: Optional[torch.Tensor] = None  # (B,1,1)
        self._std: Optional[torch.Tensor] = None   # (B,1,1)

    def norm(self, r: torch.Tensor) -> torch.Tensor:
        """r: (B,L,1) -> r_hat: (B,L,1)."""
        if r.ndim != 3 or r.shape[-1] != 1:
            raise ValueError(f"RevIN1D expects r with shape (B,L,1), got {tuple(r.shape)}")

        mean = r.mean(dim=1, keepdim=True)  # (B,1,1)
        std = r.std(dim=1, keepdim=True, unbiased=False)  # (B,1,1)
        std = torch.clamp(std, min=self.eps)

        self._mean = mean
        self._std = std

        r_hat = (r - mean) / std
        if self.affine:
            r_hat = r_hat * self.gamma + self.beta
        return r_hat

    def denorm(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Denormalize scalar predictions using cached stats.

        y_hat: (B,1) or (B,) -> y_raw: (B,1)
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("RevIN1D cache is empty. Call norm(r) before denorm(y_hat).")

        y = y_hat.view(-1, 1, 1)  # (B,1,1)
        if self.affine:
            # Invert affine: (y - beta) / gamma, but keep sign to preserve reversibility.
            gamma = self.gamma
            sign = torch.where(gamma >= 0, torch.ones_like(gamma), -torch.ones_like(gamma))
            denom = sign * torch.clamp(gamma.abs(), min=self.eps)
            y = (y - self.beta) / denom

        y = y * self._std + self._mean
        return y.view(-1, 1)


# -----------------------------
# 3) Dataset
# -----------------------------
class TSReturnDataset(Dataset):
    """Map-style dataset for next-step return prediction.

    The dataset always returns the "baseline" (global-vol) target:
      y_global = r_{t+1} / sigma_t_global

    In v4, when use_revin=1, we *override* the volatility inside the model using the
    *current window* volatility (std of the lookback window). In that mode:
      y_window = r_{t+1} / sigma_window
    and the sigma channel fed to the base model is replaced with sigma_window.
    """

    def __init__(
        self,
        returns: np.ndarray,
        sigma: np.ndarray,
        lookback: int = 60,
        eps: float = 1e-8,
    ):
        assert len(returns) == len(sigma)
        self.r = np.asarray(returns, dtype=np.float32).reshape(-1)
        self.sigma = np.maximum(np.asarray(sigma, dtype=np.float32).reshape(-1), eps)
        self.lookback = int(lookback)
        self.eps = float(eps)

        # valid t needs enough lookback and needs t+1 to exist
        self.idxs = list(range(self.lookback - 1, len(self.r) - 1))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i: int):
        t = self.idxs[i]
        r_seq = self.r[t - self.lookback + 1 : t + 1]     # (L,)
        s_seq = self.sigma[t - self.lookback + 1 : t + 1] # (L,)

        x = np.stack([r_seq, s_seq], axis=-1)  # (L,2)

        scale = self.sigma[t]       # sigma_t_global
        raw_target = self.r[t + 1]  # r_{t+1}
        y_global = raw_target / scale

        return (
            torch.from_numpy(x),
            torch.tensor([y_global], dtype=torch.float32),
            torch.tensor([scale], dtype=torch.float32),
            torch.tensor([raw_target], dtype=torch.float32),
        )


# -----------------------------
# 4) Base networks (NO RevIN inside)
# -----------------------------
class LinearNet(nn.Module):
    def __init__(self, lookback: int, num_features: int):
        super().__init__()
        self.lookback = int(lookback)
        self.num_features = int(num_features)
        self.fc = nn.Linear(self.lookback * self.num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, f = x.shape
        if l != self.lookback or f != self.num_features:
            raise ValueError(f"Expected x shape (B,{self.lookback},{self.num_features}), got {tuple(x.shape)}")
        return self.fc(x.reshape(b, l * f))


class MLPNet(nn.Module):
    def __init__(self, lookback: int, num_features: int, hidden: int = 128):
        super().__init__()
        self.lookback = int(lookback)
        self.num_features = int(num_features)
        self.net = nn.Sequential(
            nn.Linear(self.lookback * self.num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, f = x.shape
        if l != self.lookback or f != self.num_features:
            raise ValueError(f"Expected x shape (B,{self.lookback},{self.num_features}), got {tuple(x.shape)}")
        return self.net(x.reshape(b, l * f))


class LSTMNet(nn.Module):
    def __init__(self, num_features: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


# -----------------------------
# 5) Wrapper with two schemes
# -----------------------------
class VolNormWrapper(nn.Module):
    """Two operating modes (target_mode='volnorm' and x contains 2 channels):

    - use_revin = False (baseline):
        x = [r_seq, sigma_seq_global]
        base predicts z = r_{t+1} / sigma_t_global
        r_pred = z_pred * sigma_t_global

    - use_revin = True (RevIN + window volatility override)  **v4 change**:
        1) Compute per-window mean/std from r_seq (RevIN stats).
        2) Normalize r_seq -> r_hat.
        3) Override the sigma channel with sigma_window (std) repeated across the window.
        4) base predicts r_hat_{t+1}; denorm -> r_pred (raw return).
        5) z_pred = r_pred / sigma_window.

        This makes the model "see" the current window volatility, and y is also
        normalized by the same sigma_window.
    """

    def __init__(self, base: nn.Module, use_revin: bool = False, revin_affine: bool = True):
        super().__init__()
        self.base = base
        self.use_revin = bool(use_revin)
        self.revin = RevIN1D(affine=revin_affine) if self.use_revin else None

        # cache the last per-batch window scale used in forward() (B,1)
        self.last_scale: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (z_pred, r_pred).

        x: (B,L,2) with channels [returns, sigma]
        scale:
          - baseline: sigma_t_global (B,1), default uses last sigma from x.
          - revin: ignored (we use sigma_window computed from the current window).
        """
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"Expected x shape (B,L,2) with [r,sigma], got {tuple(x.shape)}")

        if not self.use_revin:
            if scale is None:
                scale = x[:, -1, 1:2]  # sigma_t_global
            z_pred = self.base(x)
            r_pred = z_pred * scale
            self.last_scale = scale
            return z_pred, r_pred

        # --- RevIN mode: override sigma with window std ---
        assert self.revin is not None

        r = x[..., 0:1]  # (B,L,1)
        r_hat = self.revin.norm(r)  # caches mean/std

        if self.revin._std is None:
            raise RuntimeError("RevIN std cache missing after norm().")
        sigma_window = self.revin._std  # (B,1,1)
        self.last_scale = sigma_window.view(-1, 1)  # (B,1)

        # sigma feature is constant across the window (broadcast)
        b, l, _ = r_hat.shape
        s_feat = sigma_window.expand(b, l, 1)
        x_in = torch.cat([r_hat, s_feat], dim=-1)  # (B,L,2)

        r_pred_hat = self.base(x_in)         # predict in normalized-return space
        r_pred = self.revin.denorm(r_pred_hat)  # raw return
        z_pred = r_pred / self.last_scale     # volnorm by *window* volatility
        return z_pred, r_pred


# -----------------------------
# 6) Differentiable correlation (IC-like) loss helpers
# -----------------------------
def batch_pearsonr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable Pearson correlation computed over the batch dimension."""
    x = pred.reshape(-1)
    y = target.reshape(-1)
    x = x - x.mean()
    y = y - y.mean()

    cov = (x * y).mean()
    x_std = torch.sqrt((x * x).mean() + eps)
    y_std = torch.sqrt((y * y).mean() + eps)
    corr = cov / (x_std * y_std + eps)
    return torch.clamp(corr, -1.0, 1.0)


# -----------------------------
# 7) Train / Eval
# -----------------------------
@torch.no_grad()
def eval_epoch(model: VolNormWrapper, loader: DataLoader, device: torch.device):
    """Evaluate one epoch.

    Returns:
      - mse_y:      MSE in the *model-space* vol-normalized target.
                   (baseline uses global sigma; RevIN uses window sigma)
      - mse_y_global: MSE in the *global* vol-normalized target r/sigma_global (comparable across modes)
      - mse_r:      MSE in raw-return space (comparable across modes)
      - ic:         Pearson corr(pred_r, true_r) in raw-return space (comparable across modes)
    """
    model.eval()
    mse_y = 0.0
    mse_y_global = 0.0
    mse_r = 0.0
    n = 0

    preds_r = []
    trues_r = []

    for x, y_global, scale_global, raw_target in loader:
        x = x.to(device)
        y_global = y_global.to(device)
        scale_global = scale_global.to(device)
        raw_target = raw_target.to(device)

        if not model.use_revin:
            z_pred, r_pred = model(x, scale=scale_global)
            y_model = y_global
        else:
            z_pred, r_pred = model(x)
            if model.last_scale is None:
                raise RuntimeError("model.last_scale is None in RevIN mode.")
            y_model = raw_target / model.last_scale

        # (1) model-space y MSE (baseline: global; revin: local)
        mse_y += ((z_pred - y_model) ** 2).sum().item()

        # (2) global y MSE: compare r_pred/sigma_global against y_global  (apples-to-apples across modes)
        y_pred_global = r_pred / scale_global
        mse_y_global += ((y_pred_global - y_global) ** 2).sum().item()

        # (3) raw-return MSE (apples-to-apples across modes)
        mse_r += ((r_pred - raw_target) ** 2).sum().item()

        preds_r.append(r_pred.detach().cpu().view(-1))
        trues_r.append(raw_target.detach().cpu().view(-1))
        n += x.shape[0]

    preds_r = torch.cat(preds_r).numpy()
    trues_r = torch.cat(trues_r).numpy()

    if np.std(preds_r) < 1e-12 or np.std(trues_r) < 1e-12:
        ic = 0.0
    else:
        ic = float(np.corrcoef(preds_r, trues_r)[0, 1])

    return {"mse_y": mse_y / n, "mse_y_global": mse_y_global / n, "mse_r": mse_r / n, "ic": ic}


def train(args):
    # --- hard constraints kept from v3 for safety ---
    if args.target_mode != "volnorm":
        raise ValueError("This script is written for target_mode='volnorm' only.")
    if args.add_sigma_feature != 1:
        raise ValueError("This script is written for add_sigma_feature=1 only (x contains [r,sigma]).")

    df = pd.read_csv(args.csv)
    price = df[args.price_col].astype(float)
    r = np.log(price).diff().fillna(0.0).to_numpy(dtype=np.float64)

    # Baseline volatility (no look-ahead). In RevIN mode it is still computed to keep
    # dataset shape consistent, but the model will override it with per-window volatility.
    if args.vol_method == "ewma":
        sigma = ewma_vol(r, alpha=args.ewma_alpha)
    else:
        sigma = rolling_vol(r, window=args.vol_window)

    # chronological split
    n = len(r)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    # Constant to scale raw-return MSE so its magnitude is comparable to vol-normalized MSE.
    # (Scaling by a constant does NOT change the optimum; it's only for stable gradients/logs.)
    r_loss_norm = float(np.mean(sigma[:n_train] ** 2)) if n_train > 0 else 1.0

    lookback = args.lookback
    r_train, s_train = r[:n_train], sigma[:n_train]
    r_val, s_val = r[n_train - lookback : n_train + n_val], sigma[n_train - lookback : n_train + n_val]
    r_test, s_test = r[n_train + n_val - lookback :], sigma[n_train + n_val - lookback :]

    ds_train = TSReturnDataset(r_train, s_train, lookback=lookback)
    ds_val = TSReturnDataset(r_val, s_val, lookback=lookback)
    ds_test = TSReturnDataset(r_test, s_test, lookback=lookback)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    # base model sees (B,L,2)
    num_features = 2
    if args.model == "linear":
        base = LinearNet(lookback=lookback, num_features=num_features)
    elif args.model == "mlp":
        base = MLPNet(lookback=lookback, num_features=num_features, hidden=args.hidden)
    else:
        base = LSTMNet(num_features=num_features, hidden=args.hidden, num_layers=args.num_layers)

    model = VolNormWrapper(base=base, use_revin=bool(args.use_revin), revin_affine=bool(args.revin_affine))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    # Model selection
    if args.select_metric in ("mse_y", "mse_y_global", "mse_r"):
        best_score = float("inf")
    else:
        best_score = -float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse_y = 0.0
        total_mse_r = 0.0
        m = 0

        for x, y_global, scale_global, raw_target in dl_train:
            x = x.to(device)
            raw_target = raw_target.to(device)

            if not model.use_revin:
                y_model = y_global.to(device)
                scale = scale_global.to(device)
                z_pred, r_pred = model(x, scale=scale)
            else:
                z_pred, r_pred = model(x)
                if model.last_scale is None:
                    raise RuntimeError("model.last_scale is None in RevIN mode.")
                y_model = raw_target / model.last_scale
                scale = model.last_scale

            # model-space y loss (baseline: global; revin: local)
            loss_mse_y = loss_fn(z_pred, y_model)
            # raw-return loss (apples-to-apples across modes)
            loss_mse_r = loss_fn(r_pred, raw_target)

            # Choose a single loss space to make runs comparable
            if args.loss_space == "y":
                loss = loss_mse_y
                if args.lambda_mse_r > 0.0:
                    # add raw-return MSE as an auxiliary term (scaled by a constant)
                    loss = loss + args.lambda_mse_r * (loss_mse_r / (r_loss_norm + 1e-8))
            elif args.loss_space == "r":
                # pure raw-return objective (scaled by a constant; scaling does NOT change the optimum)
                loss = loss_mse_r / (r_loss_norm + 1e-8)
            else:  # both
                loss = loss_mse_y
                if args.lambda_mse_r > 0.0:
                    loss = loss + args.lambda_mse_r * (loss_mse_r / (r_loss_norm + 1e-8))

            # Optional: add a differentiable IC proxy aligned with eval_epoch (raw-return IC)
            if args.lambda_ic > 0.0:
                corr = batch_pearsonr(r_pred, raw_target)
                loss = loss + args.lambda_ic * (1.0 - corr)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += float(loss.item()) * x.shape[0]
            total_mse_y += float(loss_mse_y.item()) * x.shape[0]
            total_mse_r += float(loss_mse_r.item()) * x.shape[0]
            m += x.shape[0]

        train_loss = total_loss / m
        train_mse_y = total_mse_y / m
        train_mse_r = total_mse_r / m
        val_metrics = eval_epoch(model, dl_val, device)

        # compute selection score
        if args.select_metric == "mse_y":
            # WARNING: mse_y is in *model-space* (baseline: global; revin: local). Not apples-to-apples across modes.
            score = val_metrics["mse_y"]
            improved = score < (best_score - args.min_delta)
        elif args.select_metric == "mse_y_global":
            # Global vol-normalized target r/sigma_global (apples-to-apples across modes)
            score = val_metrics["mse_y_global"]
            improved = score < (best_score - args.min_delta)
        elif args.select_metric == "mse_r":
            # Raw-return MSE (apples-to-apples across modes)
            score = val_metrics["mse_r"]
            improved = score < (best_score - args.min_delta)
        elif args.select_metric == "ic":
            score = val_metrics["ic"]
            improved = score > (best_score + args.min_delta)
        elif args.select_metric == "combo_r":
            # score = IC - w * raw MSE
            score = val_metrics["ic"] - args.combo_mse_weight * val_metrics["mse_r"]
            improved = score > (best_score + args.min_delta)
        else:  # combo (y)
            # score = IC - w * mse_y (model-space)
            score = val_metrics["ic"] - args.combo_mse_weight * val_metrics["mse_y"]
            improved = score > (best_score + args.min_delta)

        if improved:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # early stopping
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[EarlyStop] epoch={epoch} best_score={best_score:.6g} (metric={args.select_metric})")
            break

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:04d}] "
                f"train_loss={train_loss:.6g}  "
                f"train_mse_y={train_mse_y:.6g}  "
                f"train_mse_r={train_mse_r:.6g}  "
                f"val_mse_y={val_metrics['mse_y']:.6g}  "
                f"val_mse_y_g={val_metrics['mse_y_global']:.6g}  "
                f"val_mse_r={val_metrics['mse_r']:.6g}  "
                f"val_ic={val_metrics['ic']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_epoch(model, dl_test, device)
    print(f"[TEST] mse_y={test_metrics['mse_y']:.6g}  mse_y_g={test_metrics['mse_y_global']:.6g}  mse_r={test_metrics['mse_r']:.6g}  ic={test_metrics['ic']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--price_col", type=str, default="close")

    p.add_argument("--model", type=str, choices=["linear", "mlp", "lstm"], default="lstm")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)

    # fixed by request (still exposed for safety checks)
    p.add_argument("--target_mode", type=str, choices=["volnorm"], default="volnorm")
    p.add_argument("--add_sigma_feature", type=int, choices=[1], default=1)

    p.add_argument("--use_revin", type=int, choices=[0, 1], default=0)
    p.add_argument("--revin_affine", type=int, choices=[0, 1], default=1)

    p.add_argument("--vol_method", type=str, choices=["ewma", "rolling"], default="ewma")
    p.add_argument("--ewma_alpha", type=float, default=0.94)
    p.add_argument("--vol_window", type=int, default=50)

    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--loss_space", type=str, choices=["y", "r", "both"], default="y",
                   help="Training loss space: y=vol-normalized, r=raw return, both=combine (y + lambda_mse_r * r).")
    p.add_argument("--lambda_ic", type=float, default=0.0,
                   help="Weight for (1 - batch Pearson corr) term (IC proxy). 0 disables.")
    p.add_argument("--lambda_mse_r", type=float, default=0.0,
                   help="Weight for auxiliary raw-return MSE term. Used when loss_space is y/both. 0 disables.")

    p.add_argument("--select_metric", type=str,
                   choices=["mse_y", "mse_y_global", "mse_r", "ic", "combo", "combo_r"],
                   default="mse_y")
    p.add_argument("--combo_mse_weight", type=float, default=0.05,
                   help="For select_metric=combo: score = ic - w*mse_y; for combo_r: score = ic - w*mse_r.")
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping patience in epochs (0 disables).")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="Minimum improvement required to reset patience.")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--cpu", action="store_true")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
