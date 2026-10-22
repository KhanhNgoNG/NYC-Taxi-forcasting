# Runtime definitions extracted verbatim from reference/02-graph-wavenet-adaptive-graph.ipynb
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class CFG:
    asset_dir: str = "/kaggle/input/notebooks/khanhngnguynl/graphdl/graph_assets"
    out_dir: str = "/kaggle/working"

    # data
    seq_len: int = 48
    lag_steps: int = 336
    lag_taps: tuple = (-1, 0, 1)      # lấy t-335, t-336, t-337 (bước 30 phút)
    strict_comparable: bool = True    # True: cùng tập mẫu val/test với các baseline cũ

    # model
    res_ch: int = 32
    skip_ch: int = 128
    emb_dim: int = 16
    head_hidden: int = 128
    order: int = 2
    dropout: float = 0.2
    residual_head: bool = True
    use_fixed: bool = True
    use_adaptive: bool = True
    use_lag: bool = True
    use_cal: bool = True
    use_holiday: bool = True          # 4 cờ ngày lễ trong calendar
    use_analog: bool = True           # trung bình các ngày cùng loại (2 kênh: mean, last)

    # holiday
    holiday_country: str = "US"
    holiday_subdiv: str = "NY"        # None nếu không cần
    holiday_extra: tuple = ()         # thêm ngày, ví dụ ("2024-11-29",)
    holiday_exclude: tuple = ("Election",)   # loại theo tên chứa chuỗi (Election Day của NY không phải ngày nghỉ thực sự)
    analog_k: int = 4                 # số ngày cùng loại gần nhất để lấy trung bình
    holiday_loss_weight: float = 3.0  # trọng số loss cho mẫu có target là ngày lễ (1.0 = tắt)

    # training
    batch_size: int = 64
    epochs: int = 60
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    l1_weight: float = 0.5            # loss = MSE + l1_weight * L1
    ema_decay: float = 0.999          # 0 = tắt EMA
    clip_norm: float = 5.0
    amp: bool = True
    seed: int = 42

def row_normalize_adjacency(adj):
    adj = adj.astype(np.float32) + np.eye(adj.shape[0], dtype=np.float32)
    deg = adj.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return (adj / deg).astype(np.float32)

def build_fixed_supports(adj):
    """Đồ thị tương quan thường đối xứng -> chỉ cần 1 hướng. Nếu bất đối xứng thì thêm hướng ngược."""
    supports = [row_normalize_adjacency(adj)]
    if not np.allclose(adj, adj.T, atol=1e-6):
        supports.append(row_normalize_adjacency(adj.T))
    return [torch.tensor(s) for s in supports]

class CausalConv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(cin, cout, (1, kernel_size), dilation=(1, dilation))

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0, 0, 0)))

class DiffusionGraphConv(nn.Module):
    """out[n] = sum_m A[n,m] x[m]  (A hàng-chuẩn-hóa => trung bình láng giềng)."""

    def __init__(self, cin, cout, num_supports, order=2, dropout=0.1):
        super().__init__()
        self.order, self.dropout = order, dropout
        self.proj = nn.Conv2d(cin * (1 + num_supports * order), cout, 1)

    def forward(self, x, supports):
        outs = [x]
        for A in supports:
            xk = x
            for _ in range(self.order):
                xk = torch.matmul(A.to(xk.dtype), xk)    # (N,N) @ (B,C,N,T)
                outs.append(xk)
        h = self.proj(torch.cat(outs, dim=1))
        return F.dropout(h, p=self.dropout, training=self.training)

class GWNBlock(nn.Module):
    def __init__(self, res_ch, skip_ch, dilation, num_supports, order, dropout):
        super().__init__()
        self.filter_conv = CausalConv2d(res_ch, res_ch, 2, dilation)
        self.gate_conv = CausalConv2d(res_ch, res_ch, 2, dilation)
        self.skip_conv = nn.Conv2d(res_ch, skip_ch, 1)
        self.graph_conv = DiffusionGraphConv(res_ch, res_ch, num_supports, order, dropout)
        self.bn = nn.BatchNorm2d(res_ch)

    def forward(self, x, supports):
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        skip = self.skip_conv(h)
        h = self.graph_conv(h, supports)
        return self.bn(h + x), skip

class AdaptiveGWNV3(nn.Module):
    def __init__(self, num_nodes, fixed_supports, cfg, n_lag):
        super().__init__()
        self.cfg, self.N = cfg, num_nodes
        fixed = fixed_supports if cfg.use_fixed else []
        self.register_buffer("fixed", torch.stack(fixed) if fixed else torch.zeros(0, num_nodes, num_nodes))
        n_sup = len(fixed) + (1 if cfg.use_adaptive else 0)

        # randn (không nhân 0.1) để đồ thị thích nghi không bị đồng đều lúc khởi tạo
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, cfg.emb_dim))
        self.nodevec2 = nn.Parameter(torch.randn(cfg.emb_dim, num_nodes))

        # chọn cột calendar / kênh lag theo cờ (data luôn tạo đủ để ablation dùng chung)
        n_cal = (11 if cfg.use_holiday else 7) if cfg.use_cal else 0
        self.cal_cols = list(range(n_cal))
        self.lag_cols = (list(range(n_lag - 2)) if cfg.use_lag else []) + \
                        ([n_lag - 2, n_lag - 1] if cfg.use_analog else [])

        in_ch = 1 + n_cal
        self.input_proj = nn.Conv2d(in_ch, cfg.res_ch, 1)
        self.blocks = nn.ModuleList([
            GWNBlock(cfg.res_ch, cfg.skip_ch, d, n_sup, cfg.order, cfg.dropout)
            for d in (1, 2, 4, 8, 16, 32)
        ])

        head_in = cfg.skip_ch + 1 + cfg.emb_dim + len(self.lag_cols) + n_cal
        self.head1 = nn.Conv1d(head_in, cfg.head_hidden, 1)
        self.head2 = nn.Conv1d(cfg.head_hidden, 1, 1)
        self.head_drop = nn.Dropout(cfg.dropout)

    def get_adaptive_adj(self):
        a = F.relu(self.nodevec1 @ self.nodevec2) / (self.cfg.emb_dim ** 0.5)
        return F.softmax(a, dim=1)

    def forward(self, x, cal_w, cal_t, lag):
        B, T, N = x.shape
        h = x.permute(0, 2, 1).unsqueeze(1)                              # B,1,N,T
        if self.cfg.use_cal:
            c = cal_w[..., self.cal_cols].permute(0, 2, 1).unsqueeze(2).expand(-1, -1, N, -1)
            h = torch.cat([h, c], dim=1)
        h = self.input_proj(h)

        supports = list(self.fixed)
        if self.cfg.use_adaptive:
            supports.append(self.get_adaptive_adj())

        skip_sum = None
        for blk in self.blocks:
            h, s = blk(h, supports)
            skip_sum = s if skip_sum is None else skip_sum + s
        s_last = F.relu(skip_sum)[..., -1]                               # B,skip,N

        x_last = x[:, -1, :]                                             # B,N
        feats = [s_last, x_last.unsqueeze(1).to(s_last.dtype)]
        if self.lag_cols:
            feats.append(lag[:, self.lag_cols].to(s_last.dtype))         # B,*,N
        if self.cfg.use_cal:
            feats.append(cal_t[:, self.cal_cols].unsqueeze(-1).expand(-1, -1, N).to(s_last.dtype))
        feats.append(self.nodevec1.t().unsqueeze(0).expand(B, -1, -1).to(s_last.dtype))

        z = F.relu(self.head1(torch.cat(feats, dim=1)))
        out = self.head2(self.head_drop(z)).squeeze(1)                   # B,N
        if self.cfg.residual_head:
            out = out + x_last.to(out.dtype)
        return out
