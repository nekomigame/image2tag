"""学習済み Swin Transformer v2 モデルによるタグ予測・推論モジュール.

画像（ローカルファイル、URL、PIL Image）を入力として受け取り、
チェックポイント（.pth）からモデルとタグ語彙を復元して、
信頼度付きタグ一覧やカンマ区切りプロンプト形式で出力します。
"""

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import requests
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor

from model import build_model


class TagPredictor:
    """学習済み Swin Transformer v2 タグ分類モデルを用いた推論クラス."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        vocab_path: Optional[str] = None,
    ) -> None:
        """推論クラスを初期化し、モデルとタグ辞書をロードします.

        Args:
            checkpoint_path: 学習済み重みファイル (.pth) のパス
            device: 実行デバイス ('cuda' または 'cpu'。省略時は自動判定)
            vocab_path: 語彙ファイル (tag_vocab.json) のパス (チェックポイント内に辞書がない場合のフォールバック)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {checkpoint_path}")

        # デバイス決定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"モデルロード中: {checkpoint_path} (デバイス: {self.device})")

        # チェックポイントの読み込み
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # メタデータの復元
        model_name = checkpoint.get("model_name", "swinv2_tiny_window16_256")
        preset = checkpoint.get("preset", "low_vram")
        self.image_size: int = checkpoint.get("image_size", 256)
        num_classes: int = checkpoint.get("num_classes", 0)

        # タグ語彙辞書の復元
        if "id_to_tag" in checkpoint:
            # チェックポイント内の id_to_tag (キーが整数の場合と文字列の場合を考慮)
            raw_id_to_tag = checkpoint["id_to_tag"]
            self.id_to_tag: Dict[int, str] = {int(k): str(v) for k, v in raw_id_to_tag.items()}
            self.tag_to_id: Dict[str, int] = {v: k for k, v in self.id_to_tag.items()}
        elif vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            self.tag_to_id = vocab_data.get("tag_to_id", {})
            self.id_to_tag = {int(v): k for k, v in self.tag_to_id.items()}
        else:
            raise ValueError(
                "チェックポイント内にタグ語彙辞書が含まれておらず、vocab_path も指定されていません。"
            )

        if num_classes == 0:
            num_classes = len(self.id_to_tag)

        # モデルの構築と重み適用
        self.model, _ = build_model(
            num_classes=num_classes,
            preset=preset,
            custom_model_name=model_name,
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # 正規化用テンソル (ImageNet 基準)
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        print(f"モデル初期化完了 (クラス数: {num_classes}, 入力解像度: {self.image_size}x{self.image_size})")

    def _load_image(self, image_input: Union[str, Path, Image.Image, io.BytesIO]) -> Image.Image:
        """多様な入力形式から PIL Image (RGB) を取得します."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, (io.BytesIO, bytes)):
            buf = image_input if isinstance(image_input, io.BytesIO) else io.BytesIO(image_input)
            return Image.open(buf).convert("RGB")

        if isinstance(image_input, (str, Path)):
            input_str = str(image_input).strip()
            # HTTP/HTTPS URL の場合
            if input_str.startswith("http://") or input_str.startswith("https://"):
                resp = requests.get(input_str, timeout=20)
                resp.raise_for_status()
                with io.BytesIO(resp.content) as buf:
                    return Image.open(buf).convert("RGB")
            # ローカルファイルパスの場合
            if os.path.exists(input_str):
                return Image.open(input_str).convert("RGB")
            raise FileNotFoundError(f"画像ファイルが見つかりません: {input_str}")

        raise TypeError(f"未対応の画像入力タイプです: {type(image_input)}")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL Image をモデル入力テンソルに変換・GPU正規化します."""
        # リサイズ
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # テンソル化: [1, 3, H, W]
        tensor = pil_to_tensor(image).unsqueeze(0)

        # GPU 転送 & 正規化
        tensor = tensor.to(self.device, non_blocking=True).float().div_(255.0)
        return tensor.sub_(self.norm_mean).div_(self.norm_std)

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image, io.BytesIO],
        threshold: float = 0.35,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """画像に対してタグ予測を行い、(タグ名, 信頼度スコア) のリストを返します.

        Args:
            image_input: 画像パス、URL、または PIL Image
            threshold: 検出とみなす最小確率しきい値 (0.0 ~ 1.0)
            top_k: 上位何件まで返すか (省略時はしきい値以上の全件)

        Returns:
            スコアの高い順にソートされた [(タグ名, 確率スコア), ...]
        """
        pil_img = self._load_image(image_input)
        tensor = self._preprocess(pil_img)

        # 順伝播 & シグモイド確率変換
        logits = self.model(tensor)
        probs = torch.sigmoid(logits)[0]  # [num_classes]

        # 信頼度スコアの高い順にインデックスをソート
        sorted_indices = torch.argsort(probs, descending=True).cpu().tolist()

        results: List[Tuple[str, float]] = []
        for idx in sorted_indices:
            score = float(probs[idx].item())
            if score < threshold:
                break
            tag_name = self.id_to_tag.get(idx, f"tag_{idx}")
            results.append((tag_name, score))
            if top_k is not None and len(results) >= top_k:
                break

        return results

    def predict_to_prompt(
        self,
        image_input: Union[str, Path, Image.Image, io.BytesIO],
        threshold: float = 0.35,
        top_k: Optional[int] = None,
        separator: str = ", ",
    ) -> str:
        """タグ予測結果をカンマ区切りのプロンプト形式文字列として返します.

        Args:
            image_input: 画像パス、URL、または PIL Image
            threshold: 信頼度しきい値
            top_k: 上位何件まで取得するか
            separator: 区切り文字列 (デフォルト: ', ')

        Returns:
            例: '1girl, solo, smile, blue_eyes, blonde_hair'
        """
        predictions = self.predict(image_input, threshold=threshold, top_k=top_k)
        return separator.join([tag for tag, _ in predictions])


def main() -> None:
    """CLI からの単体実行エントリーポイント."""
    parser = argparse.ArgumentParser(description="Swin Transformer v2 タグ予測ツール")
    parser.add_argument("--checkpoint", "-c", required=True, help="チェックポイント (.pth) のパス")
    parser.add_argument("--image", "-i", required=True, help="画像ファイルパスまたは画像 URL")
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.35, help="検出確率しきい値 (デフォルト: 0.35)"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=None, help="出力する最大タグ数 (省略時はしきい値以上の全件)"
    )
    parser.add_argument(
        "--prompt-only", action="store_true", help="カンマ区切りのタグプロンプトのみを出力"
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON 形式で出力"
    )

    args = parser.parse_args()

    predictor = TagPredictor(checkpoint_path=args.checkpoint)
    predictions = predictor.predict(args.image, threshold=args.threshold, top_k=args.top_k)

    if args.prompt_only:
        print(", ".join([tag for tag, _ in predictions]))
        return

    if args.json:
        output_data = [{"tag": tag, "score": round(score, 4)} for tag, score in predictions]
        print(json.dumps(output_data, ensure_ascii=False, indent=2))
        return

    print(f"\n予測結果 (検出タグ数: {len(predictions)} 件, しきい値: {args.threshold}):")
    print("=" * 60)
    for tag, score in predictions:
        pct = score * 100.0
        bar = "█" * int(pct // 5)
        print(f" {pct:5.1f}% | {bar:<20} | {tag}")
    print("=" * 60)
    print("\nプロンプト形式:")
    print(", ".join([tag for tag, _ in predictions]))


if __name__ == "__main__":
    main()
