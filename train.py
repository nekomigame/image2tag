"""Swin Transformer v2 タグ分類学習モジュール.

Cryptomator 内蔵 WebDAV サーバーから直接インメモリストリーミングでデータを読み込み、
Swin Transformer v2 モデルの学習を行います。
"""

from collections import Counter
import json
import os
from typing import Any, Dict, List, Tuple
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import WebDAVStreamingDataset
from loss import AsymmetricLoss
from model import build_model
from webdav_client import WebDAVClient


def prepare_tag_vocabulary(
    client: WebDAVClient,
    dataset_name: str,
    min_tag_frequency: int = 10,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """WebDAV 内の metadata.jsonl を集計し、出現回数条件に基づくタグ語彙を生成します.

    Args:
        client: WebDAV クライアント
        dataset_name: データセット識別名
        min_tag_frequency: 最小タグ出現回数しきい値

    Returns:
        (tag_to_id 辞書, id_to_tag 辞書)
    """
    metadata_path = f"datasets/{dataset_name}/metadata.jsonl"
    metadata_bytes = client.get_bytes(metadata_path)
    if metadata_bytes is None:
        raise FileNotFoundError(f"メタデータが見つかりません: {metadata_path}")

    # 出現回数カウント
    tag_counts: Counter = Counter()
    for line in metadata_bytes.decode("utf-8").splitlines():
        if line.strip():
            sample = json.loads(line)
            tags = sample.get("tags", [])
            tag_counts.update(tags)

    # しきい値以上のタグをフィルタリング（出現頻度順）
    filtered_tags = [
        tag for tag, count in tag_counts.most_common() if count >= min_tag_frequency
    ]

    tag_to_id: Dict[str, int] = {tag: i for i, tag in enumerate(filtered_tags)}
    id_to_tag: Dict[int, str] = {i: tag for i, tag in enumerate(filtered_tags)}

    print(
        f"タグ語彙生成完了: 検出タグ総数 {len(tag_counts)} 種類中、"
        f"出現回数 {min_tag_frequency} 回以上のタグ {len(filtered_tags)} 種類を採用"
    )

    # 語彙辞書を WebDAV に暗号化保存（再学習・推論用）
    vocab_data = {
        "dataset_name": dataset_name,
        "min_tag_frequency": min_tag_frequency,
        "num_classes": len(filtered_tags),
        "tag_to_id": tag_to_id,
        "tag_frequencies": {tag: tag_counts[tag] for tag in filtered_tags},
    }
    client.put_bytes(
        f"datasets/{dataset_name}/tag_vocab.json",
        json.dumps(vocab_data, indent=2, ensure_ascii=False).encode("utf-8"),
    )

    return tag_to_id, id_to_tag


def train_model(config: Dict[str, Any]) -> None:
    """設定情報に基づき Swin Transformer v2 の学習を実行します.

    Args:
        config: 設定辞書
    """
    # 接続設定
    storage_cfg = config.get("storage", {})
    client = WebDAVClient(
        base_url=storage_cfg.get("webdav_url", "http://127.0.0.1:42427/MyVault"),
        username=os.getenv("WEBDAV_USERNAME") or storage_cfg.get("username"),
        password=os.getenv("WEBDAV_PASSWORD") or storage_cfg.get("password"),
        auth_type=storage_cfg.get("auth_type", "basic"),
        verify_ssl=storage_cfg.get("verify_ssl", False),
    )

    # 学習パラメータ
    train_cfg = config.get("training", {})
    target_dataset = train_cfg.get("target_dataset", "default")
    preset = train_cfg.get("preset", "low_vram")
    min_tag_frequency = train_cfg.get("min_tag_frequency", 10)
    batch_size = train_cfg.get("batch_size", 32)
    epochs = train_cfg.get("epochs", 20)
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    num_workers = train_cfg.get("num_workers", 0)  # Windows環境での安定性のためデフォルト0推奨
    cache_size = train_cfg.get("in_memory_cache_size", 500)
    output_dir = train_cfg.get("output_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"学習開始: データセット='{target_dataset}', プリセット='{preset}'")
    print(f"=======================================================")

    # 1. タグ語彙辞書の準備
    tag_to_id, id_to_tag = prepare_tag_vocabulary(
        client, target_dataset, min_tag_frequency=min_tag_frequency
    )
    num_classes = len(tag_to_id)
    if num_classes == 0:
        raise ValueError("有効なタグが 0 件です。min_tag_frequency を下げるか画像数を増やしてください。")

    # 2. モデル構築 (プリセット適用)
    model, image_size = build_model(
        num_classes=num_classes,
        preset=preset,
        custom_model_name=train_cfg.get("model_name", ""),
        pretrained=train_cfg.get("pretrained", True),
    )
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)
    model = model.to(device)
    print(f"デバイス: {device}, 入力解像度: {image_size}x{image_size}, モデル: {model.model_name}")

    # 3. WebDAV ストリーミングデータセットの準備 (SSD非保存 & RAMプリロード)
    preload_to_ram = train_cfg.get("preload_to_ram", True)
    preload_workers = train_cfg.get("preload_workers", 16)

    full_dataset = WebDAVStreamingDataset(
        client=client,
        dataset_name=target_dataset,
        tag_to_id=tag_to_id,
        image_size=image_size,
        is_training=True,
        preload_to_ram=preload_to_ram,
        preload_workers=preload_workers,
        in_memory_cache_size=cache_size,
    )

    # 訓練 / 検証スプリット (90% / 10%)
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. 損失関数 & 最適化
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # GPU 高速前処理用テンソル (ImageNet 正規化)
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess_on_gpu(imgs: torch.Tensor, augment: bool = True) -> torch.Tensor:
        """GPU 上で uint8 テンソルを一括正規化 & 水平反転 (CPU 負荷ゼロ)."""
        x = imgs.to(device, non_blocking=True).float().div_(255.0)
        if augment and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])
        return x.sub_(norm_mean).div_(norm_std)

    # 5. 学習ループ
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, targets in pbar:
            targets = targets.to(device, non_blocking=True)
            # GPU 側で一括前処理
            images = preprocess_on_gpu(images, augment=True)

            optimizer.zero_grad()
            with autocast(device_type, enabled=(device_type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # 検証ループ
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                targets = targets.to(device, non_blocking=True)
                images = preprocess_on_gpu(images, augment=False)

                with autocast(device_type, enabled=(device_type == "cuda")):
                    logits = model(images)
                    loss = criterion(logits, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} 終了 - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # ベストチェックポイント保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(output_dir, f"{target_dataset}_best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "model_name": model.model_name,
                    "preset": preset,
                    "image_size": image_size,
                    "num_classes": num_classes,
                    "tag_to_id": tag_to_id,
                    "id_to_tag": id_to_tag,
                },
                ckpt_path,
            )
            print(f"ベストモデルを保存しました -> {ckpt_path}")

    print("\n学習が正常に完了しました。")
