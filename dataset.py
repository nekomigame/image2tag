"""WebDAV インメモリストリーミング & RAM プリロード高速データセットモジュール.

Cryptomator 内蔵 WebDAV サーバーから直接インメモリで画像を並列ロードし、
ローカル SSD に平文ファイルを一切書き出すことなく、
RAM 上の uint8 テンソルとして保持して GPU への最高速パイプラインを提供します。
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

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
        preload_to_ram: bool = True,
        preload_workers: int = 16,
        in_memory_cache_size: int = 2000,
    ) -> None:
        """データセットを初期化します.

        Args:
            client: WebDAV クライアントインスタンス
            dataset_name: 対象データセット識別名 (例: 'test_overwatch')
            tag_to_id: タグ名からクラスIDへのマッピング辞書
            image_size: モデル入力画像サイズ
            is_training: 学習用データ拡張を適用するか
            preload_to_ram: 初回に全画像をマルチスレッドで RAM 上にプリロードするか (推奨: True)
            preload_workers: WebDAV 並列取得スレッド数
            in_memory_cache_size: preload_to_ram=False 時に RAM 内に保持する最大画像数
        """
        self.client: WebDAVClient = client
        self.dataset_name: str = dataset_name
        self.tag_to_id: Dict[str, int] = tag_to_id
        self.num_classes: int = len(tag_to_id)
        self.image_size: int = image_size
        self.is_training: bool = is_training
        self.preload_to_ram: bool = preload_to_ram
        self.preload_workers: int = preload_workers
        self.in_memory_cache_size: int = in_memory_cache_size

        # データセットのルートパス
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

        # 事前に全サンプルのマルチホットベクトル（ターゲット）を RAM 上に作成
        self.targets: torch.Tensor = torch.zeros(
            (len(self.samples), self.num_classes), dtype=torch.float32
        )
        for i, sample in enumerate(self.samples):
            for tag in sample.get("tags", []):
                tag_id = self.tag_to_id.get(tag)
                if tag_id is not None:
                    self.targets[i, tag_id] = 1.0

        # RAM 内テンソルストレージ (SSD 非保存)
        # preload_to_ram=True の場合: 形状 [N, 3, image_size, image_size] の uint8 テンソル
        # (1000枚の256x256画像でわずか約187MBのメモリ消費)
        self._ram_tensors: Optional[torch.Tensor] = None
        self._lru_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        if self.preload_to_ram:
            self._execute_preload()

    def _execute_preload(self) -> None:
        """マルチスレッドで全画像を WebDAV から RAM 上へ並列吸い上げします (SSD 書き込みゼロ)."""
        num_samples = len(self.samples)
        print(f"\n[高速化] WebDAV から RAM 上へ全 {num_samples} 枚を並列プリロード中 ({self.preload_workers} スレッド)...")

        # RAM 内テンソル配列を事前確保 (uint8: 0~255)
        self._ram_tensors = torch.zeros(
            (num_samples, 3, self.image_size, self.image_size), dtype=torch.uint8
        )

        def fetch_and_process(idx: int) -> Tuple[int, Optional[torch.Tensor]]:
            filename = self.samples[idx]["filename"]
            image_path = f"{self.images_dir}/{filename}"
            img_bytes = self.client.get_bytes(image_path)
            if img_bytes is None:
                return idx, None

            try:
                with io.BytesIO(img_bytes) as buf:
                    with Image.open(buf) as img:
                        img_rgb = img.convert("RGB")
                        if img_rgb.size != (self.image_size, self.image_size):
                            img_rgb = img_rgb.resize(
                                (self.image_size, self.image_size), Image.BILINEAR
                            )
                        tensor = pil_to_tensor(img_rgb)  # [3, H, W] uint8
                        return idx, tensor
            except Exception:
                return idx, None

        with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
            futures = [executor.submit(fetch_and_process, i) for i in range(num_samples)]
            for f in tqdm(as_completed(futures), total=num_samples, desc="RAM プリロード進捗"):
                idx, tensor = f.result()
                if tensor is not None:
                    self._ram_tensors[idx] = tensor

        approx_mb = (self._ram_tensors.element_size() * self._ram_tensors.nelement()) / (1024 * 1024)
        print(f"[完了] RAM プリロード完了！ 使用メモリ: 約 {approx_mb:.1f} MB (ローカル SSD 書き込み 0 バイト)")
        print(f"[高速化] 以降のエポックでは WebDAV 通信および CPU デコード待ちが一切発生しません。\n")

    def __len__(self) -> int:
        """サンプル総数を返します."""
        return len(self.samples)

    def _fetch_single_image(self, idx: int) -> torch.Tensor:
        """非プリロード時のインメモリ取得."""
        filename = self.samples[idx]["filename"]
        if filename in self._lru_cache:
            self._lru_cache.move_to_end(filename)
            return self._lru_cache[filename]

        image_path = f"{self.images_dir}/{filename}"
        img_bytes = self.client.get_bytes(image_path)
        if img_bytes is None:
            tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.uint8)
        else:
            try:
                with io.BytesIO(img_bytes) as buf:
                    with Image.open(buf) as img:
                        img_rgb = img.convert("RGB")
                        if img_rgb.size != (self.image_size, self.image_size):
                            img_rgb = img_rgb.resize(
                                (self.image_size, self.image_size), Image.BILINEAR
                            )
                        tensor = pil_to_tensor(img_rgb)
            except Exception:
                tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.uint8)

        if len(self._lru_cache) >= self.in_memory_cache_size:
            self._lru_cache.popitem(last=False)
        self._lru_cache[filename] = tensor
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """指定インデックスのサンプル (uint8テンソル, ターゲットベクトル) を高速返却."""
        if self._ram_tensors is not None:
            img_tensor = self._ram_tensors[idx]
        else:
            img_tensor = self._fetch_single_image(idx)

        target = self.targets[idx]
        return img_tensor, target
