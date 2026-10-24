"""画像・タグ学習システム CLI エントリーポイント.

Rule34 からのデータ収集（Cryptomator WebDAV への直接暗号化保存）、
保存済み複数データセットの一覧表示、
Swin Transformer v2 を用いたマルチラベル分類学習を統合実行します。
"""

import argparse
import json
import os
import sys
from typing import Any, Dict
import yaml

from collector import DatasetCollector
from train import train_model
from webdav_client import WebDAVClient


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """設定ファイルを読み込みます.

    Args:
        config_path: YAML 設定ファイルのパス

    Returns:
        設定辞書
    """
    if not os.path.exists(config_path):
        print(f"警告: 設定ファイル {config_path} が見つかりません。デフォルト値を使用します。")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def handle_download(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """データ収集処理を実行します."""
    storage_cfg = config.get("storage", {})
    client = WebDAVClient(
        base_url=args.webdav_url or storage_cfg.get("webdav_url", "http://127.0.0.1:42427/MyVault"),
        username=storage_cfg.get("username"),
        password=storage_cfg.get("password"),
        auth_type=storage_cfg.get("auth_type", "basic"),
        verify_ssl=storage_cfg.get("verify_ssl", False),
    )

    collector_cfg = config.get("collector", {})
    dataset_name = args.dataset_name or collector_cfg.get("dataset_name", "dataset_v1")
    tags = args.tags.split() if args.tags else collector_cfg.get("query_tags", ["1girl"])
    limit = args.limit or collector_cfg.get("limit", 500)
    min_score = args.min_score if args.min_score is not None else collector_cfg.get("min_score", 0)

    collector = DatasetCollector(client)
    collector.collect(
        dataset_name=dataset_name,
        query_tags=tags,
        limit=limit,
        min_score=min_score,
    )


def handle_train(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """学習処理を実行します."""
    if args.dataset_name:
        config.setdefault("training", {})["target_dataset"] = args.dataset_name
    if args.preset:
        config.setdefault("training", {})["preset"] = args.preset
    if args.min_frequency is not None:
        config.setdefault("training", {})["min_tag_frequency"] = args.min_frequency
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.webdav_url:
        config.setdefault("storage", {})["webdav_url"] = args.webdav_url

    train_model(config)


def handle_list(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """WebDAV 内に保存されているデータセット一覧を表示します."""
    storage_cfg = config.get("storage", {})
    client = WebDAVClient(
        base_url=args.webdav_url or storage_cfg.get("webdav_url", "http://127.0.0.1:42427/MyVault"),
        username=storage_cfg.get("username"),
        password=storage_cfg.get("password"),
        auth_type=storage_cfg.get("auth_type", "basic"),
        verify_ssl=storage_cfg.get("verify_ssl", False),
    )

    print("\nCryptomator 内のデータセットを検索中...")
    datasets = client.list_dir("datasets")
    if not datasets:
        print("保存されているデータセットは見つかりませんでした。")
        return

    print(f"\n検出されたデータセット ({len(datasets)} 件):")
    print("-" * 60)
    for ds_name in datasets:
        info_bytes = client.get_bytes(f"datasets/{ds_name}/dataset_info.json")
        if info_bytes:
            info = json.loads(info_bytes.decode("utf-8"))
            tags_str = ", ".join(info.get("query_tags", []))
            total = info.get("total_images", "不明")
            updated = info.get("updated_at", "不明")
            print(f"・ データセット名: {ds_name}")
            print(f"    - 画像枚数: {total} 枚")
            print(f"    - 収集タグ: {tags_str}")
            print(f"    - 更新日時: {updated}")
        else:
            print(f"・ データセット名: {ds_name} (メタデータ未取得)")
    print("-" * 60)


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="Swin Transformer v2 画像・タグ学習システム (Cryptomator WebDAV連携)"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="設定ファイルのパス (デフォルト: config.yaml)"
    )
    parser.add_argument(
        "--webdav-url", default=None, help="Cryptomator WebDAV ルート URL (設定値を上書き)"
    )

    subparsers = parser.add_subparsers(dest="command", help="実行サブコマンド")

    # サブコマンド: download
    dl_parser = subparsers.add_parser("download", help="Rule34 から画像を収集して WebDAV に暗号化保存")
    dl_parser.add_argument("--dataset-name", "-n", help="保存先データセット識別名")
    dl_parser.add_argument("--tags", "-t", help="検索タグ (スペース区切り, 例: '1girl solo')")
    dl_parser.add_argument("--limit", "-l", type=int, help="取得上限枚数")
    dl_parser.add_argument("--min-score", "-s", type=int, help="最小スコアフィルター")

    # サブコマンド: train
    tr_parser = subparsers.add_parser("train", help="WebDAV 内のデータセットから学習を実行")
    tr_parser.add_argument("--dataset-name", "-n", help="学習対象データセット名")
    tr_parser.add_argument(
        "--preset", "-p", choices=["low_vram", "high_quality"], help="モデルプリセット"
    )
    tr_parser.add_argument(
        "--min-frequency", "-f", type=int, help="タグ最小出現回数しきい値"
    )
    tr_parser.add_argument("--epochs", "-e", type=int, help="エポック数")
    tr_parser.add_argument("--batch-size", "-b", type=int, help="バッチサイズ")

    # サブコマンド: list
    subparsers.add_parser("list", help="Cryptomator 内の保存済みデータセット一覧を表示")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "download":
        handle_download(args, config)
    elif args.command == "train":
        handle_train(args, config)
    elif args.command == "list":
        handle_list(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
