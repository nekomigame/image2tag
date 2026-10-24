"""WebDAV インメモリストリーミングデータセットモジュール.

Cryptomator の内蔵 WebDAV サーバーから直接画像をストリーミング取得し、
ローカルディスク（SSD）に平文ファイルを一切書き出すことなく
PyTorch Tensor へ変換します。
"""

from collections import OrderedDict
import io
import json
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from webdav_client import WebDAVClient


class WebDAVStreamingDataset(Dataset):
    """Cryptomator WebDAV から直接インメモリストリーミングを行うマルチラベルデータセット."""

    def __init__(
        self,
        client: WebDAVClient,
        dataset_name: str,
        tag_to_id: Dict[str, int],
        image_size: int = 256,
        is_training: bool = True,
        in_memory_cache_size: int = 500,
        transform: Optional[Callable] = None,
    ) -> None:
        """データセットを初期化します.

        Args:
            client: WebDAV クライアントインスタンス
            dataset_name: 対象データセット識別名 (例: 'solo_1girl_v1')
            tag_to_id: タグ名からクラスIDへのマッピング辞書
            image_size: モデル入力画像サイズ
            is_training: 学習用データ拡張を適用するか
            in_memory_cache_size: RAM内に保持する最大画像数 (0の場合はキャッシュなし)
            transform: 独自の torchvision 変換処理 (省略時はデフォルト構築)
        """
        self.client: WebDAVClient = client
        self.dataset_name: str = dataset_name
        self.tag_to_id: Dict[str, int] = tag_to_id
        self.num_classes: int = len(tag_to_id)
        self.image_size: int = image_size
        self.in_memory_cache_size: int = in_memory_cache_size

        # RAM内LRUキャッシュ（SSDへの書き込みは一切行わない）
        self._ram_cache: OrderedDict[str, Image.Image] = OrderedDict()

        # データセットのルートパス (例: datasets/solo_1girl_v1)
        self.base_path: str = f"datasets/{dataset_name}"
        self.images_dir: str = f"{self.base_path}/images"

        # metadata.jsonl を WebDAV からインメモリで取得
        metadata_bytes = self.client.get_bytes(f"{self.base_path}/metadata.jsonl")
        if metadata_bytes is None:
            raise FileNotFoundError(
                f"WebDAV 内にメタデータが見つかりません: {self.base_path}/metadata.jsonl"
            )

        # メタデータをメモリ上に展開
        self.samples: List[Dict[str, Any]] = []
        for line in metadata_bytes.decode("utf-8").splitlines():
            line_str = line.strip()
            if line_str:
                self.samples.append(json.loads(line_str))

        if not self.samples:
            raise ValueError(f"データセット '{dataset_name}' に画像サンプルが存在しません。")

        # 前処理パイプライン
        if transform is not None:
            self.transform = transform
        else:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])

    def __len__(self) -> int:
        """サンプル総数を返します."""
        return len(self.samples)

    def _fetch_image(self, filename: str) -> Image.Image:
        """WebDAV から画像バイトストリームを取得し、PIL Image に変換します (SSD非書き込み)."""
        # RAM キャッシュ確認
        if self.in_memory_cache_size > 0 and filename in self._ram_cache:
            self._ram_cache.move_to_end(filename)
            return self._ram_cache[filename].copy()

        # WebDAV から直接 HTTP GET
        image_path = f"{self.images_dir}/{filename}"
        img_bytes = self.client.get_bytes(image_path)
        if img_bytes is None:
            # 取得失敗時は黒塗り画像を代替
            img = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
            return img

        # io.BytesIO でメモリ上でのみデコード
        with io.BytesIO(img_bytes) as buf:
            img = Image.open(buf).convert("RGB")

        # RAM キャッシュ管理
        if self.in_memory_cache_size > 0:
            if len(self._ram_cache) >= self.in_memory_cache_size:
                self._ram_cache.popitem(last=False)
            self._ram_cache[filename] = img.copy()

        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """指定インデックスのサンプル (画像テンソル, 正解マルチホットベクトル) を取得します."""
        sample = self.samples[idx]
        filename = sample["filename"]
        tags = sample.get("tags", [])

        # 画像のインメモリ取得
        img = self._fetch_image(filename)
        img_tensor = self.transform(img)

        # マルチホットターゲットベクトルの構築 (形状: [num_classes])
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for tag in tags:
            tag_id = self.tag_to_id.get(tag)
            if tag_id is not None:
                target[tag_id] = 1.0

        return img_tensor, target
