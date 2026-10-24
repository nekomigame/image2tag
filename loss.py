"""マルチラベル分類用の損失関数モジュール.

タグ予測における正例・負例の極端な不均衡を是正するため、
Asymmetric Loss (ASL) を提供します。
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for Multi-Label Classification.

    正例（Positive）と負例（Negative）で異なるフォーカスパラメータ（gamma）を適用し、
    負例に対して確率マージン（margin）を導入することで、
    大多数を占める簡単な負例からの勾配を大幅に減衰させます。
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        """AsymmetricLoss を初期化します.

        Args:
            gamma_neg: 負例の減衰パラメータ (大きいほど負例の影響を抑制)
            gamma_pos: 正例の減衰パラメータ
            clip: 負例の確率マージン (不要なノイズ負例のカットオフ)
            eps: 数値安定性のための微小値
            reduction: 損失集約方法 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.gamma_neg: float = gamma_neg
        self.gamma_pos: float = gamma_pos
        self.clip: float = clip
        self.eps: float = eps
        self.reduction: str = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """損失を計算します.

        Args:
            logits: モデルの出力ロジット (形状: [BatchSize, NumClasses])
            targets: マルチホット正解ラベル (形状: [BatchSize, NumClasses], 値: 0または1)

        Returns:
            集約されたスカラー損失 (reduction='mean'/'sum'の場合)
        """
        targets = targets.float()
        probs = torch.sigmoid(logits)

        # 正例に対する計算
        pos_probs = probs
        pos_loss = targets * torch.log(pos_probs.clamp(min=self.eps))
        if self.gamma_pos > 0:
            pos_weight = (1.0 - pos_probs) ** self.gamma_pos
            pos_loss = pos_loss * pos_weight

        # 負例に対する計算（非対称なクリッピングマージンとフォーカス）
        neg_probs = 1.0 - probs
        if self.clip > 0:
            # 負例の確率をマージンでクリップ
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)
        neg_loss = (1.0 - targets) * torch.log(neg_probs.clamp(min=self.eps))
        if self.gamma_neg > 0:
            neg_weight = (1.0 - neg_probs) ** self.gamma_neg
            neg_loss = neg_loss * neg_weight

        loss = -(pos_loss + neg_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
