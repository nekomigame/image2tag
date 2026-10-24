"""Swin Transformer v2 モデル定義モジュール.

timm ライブラリを利用し、Swin Transformer v2 をバックボーンとした
マルチラベル画像タグ分類モデルを構築します。
省メモリプリセットと高精度プリセットを切り替え可能です。
"""

from typing import Dict, Tuple
import timm
import torch
import torch.nn as nn


# プリセット定義
PRESETS: Dict[str, Dict[str, any]] = {
    "low_vram": {
        "model_name": "swinv2_tiny_window16_256",
        "default_image_size": 256,
        "description": "省メモリモード (SwinV2-Tiny 256px, VRAM 6~8GB 向け)",
    },
    "high_quality": {
        "model_name": "swinv2_base_window12to16_192to256",
        "default_image_size": 256,
        "description": "標準/高精度モード (SwinV2-Base 256px, VRAM 12~24GB 向け)",
    },
}


class SwinV2TagClassifier(nn.Module):
    """Swin Transformer v2 を用いたマルチラベルタグ分類モデル."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "swinv2_tiny_window16_256",
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ) -> None:
        """モデルを初期化します.

        Args:
            num_classes: 分類対象のタグ総数
            model_name: timm バックボーン名
            pretrained: 事前学習済み重み（ImageNet）を使用するか
            drop_rate: ドロップアウト率
        """
        super().__init__()
        self.num_classes: int = num_classes
        self.model_name: str = model_name

        # timm から Swin Transformer v2 バックボーンを作成
        # num_classes=0 でプーリング後の特徴ベクトル出力を取得
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )

        num_features: int = self.backbone.num_features

        # マルチラベル分類ヘッド
        self.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(p=drop_rate),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播計算を行います.

        Args:
            x: 入力画像テンソル (形状: [BatchSize, 3, Height, Width])

        Returns:
            出力ロジット (形状: [BatchSize, num_classes])
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def build_model(
    num_classes: int,
    preset: str = "low_vram",
    custom_model_name: str = "",
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> Tuple[SwinV2TagClassifier, int]:
    """設定やプリセットに基づいて分類モデルを構築します.

    Args:
        num_classes: タグ数
        preset: 'low_vram' または 'high_quality'
        custom_model_name: 独自に timm モデル名を指定する場合の上書き指定
        pretrained: 事前学習済み重みのロード有無
        drop_rate: ドロップアウト率

    Returns:
        (モデルインスタンス, 推奨入力画像サイズ)
    """
    if custom_model_name:
        selected_model = custom_model_name
        image_size = 256
    else:
        preset_info = PRESETS.get(preset, PRESETS["low_vram"])
        selected_model = preset_info["model_name"]
        image_size = preset_info["default_image_size"]

    model = SwinV2TagClassifier(
        num_classes=num_classes,
        model_name=selected_model,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )

    return model, image_size
