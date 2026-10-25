"""画像・タグ学習システム CLI エントリーポイント.

Rule34 からのデータ収集（Cryptomator WebDAV への直接暗号化保存）、
保存済み複数データセットの一覧表示、
Swin Transformer v2 を用いたマルチラベル分類学習を統合実行します。
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import yaml

from collector import DatasetCollector
from predict import TagPredictor
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
        if config_path == "config.yaml" and os.path.exists("example.config.yaml"):
            print("情報: config.yaml が存在しないため、example.config.yaml を参照します。")
            config_path = "example.config.yaml"
        else:
            print(f"警告: 設定ファイル {config_path} が見つかりません。デフォルト値を使用します。")
            return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_webdav_client(config: Dict[str, Any], url_override: Optional[str] = None) -> WebDAVClient:
    """設定および環境変数から WebDAV クライアントを初期化します."""
    storage_cfg = config.get("storage", {})
    base_url = url_override or storage_cfg.get("webdav_url", "http://127.0.0.1:42427/MyVault")
    username = os.getenv("WEBDAV_USERNAME") or storage_cfg.get("username")
    password = os.getenv("WEBDAV_PASSWORD") or storage_cfg.get("password")
    auth_type = storage_cfg.get("auth_type", "basic")
    verify_ssl = storage_cfg.get("verify_ssl", False)

    return WebDAVClient(
        base_url=base_url,
        username=username,
        password=password,
        auth_type=auth_type,
        verify_ssl=verify_ssl,
    )


def handle_download(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """データ収集処理を実行します."""
    client = get_webdav_client(config, args.webdav_url)

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
    if args.preload is not None:
        config.setdefault("training", {})["preload_to_ram"] = args.preload
    if args.preload_workers is not None:
        config.setdefault("training", {})["preload_workers"] = args.preload_workers
    if args.webdav_url:
        config.setdefault("storage", {})["webdav_url"] = args.webdav_url

    train_model(config)


def handle_list(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """WebDAV 内に保存されているデータセット一覧を表示します."""
    client = get_webdav_client(config, args.webdav_url)

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


def handle_predict(args: argparse.Namespace) -> None:
    """学習済みモデルによるタグ予測を実行します."""
    predictor = TagPredictor(checkpoint_path=args.checkpoint, vocab_path=args.vocab)
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
        bar = "#" * int(pct // 5)
        print(f" {pct:5.1f}% | {bar:<20} | {tag}")
    print("=" * 60)
    print("\nプロンプト形式:")
    print(", ".join([tag for tag, _ in predictions]))


def main() -> None:
    """メイン関数."""
    # .env 環境変数を読み込み
    load_dotenv()

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
    dl_parser.add_argument("--dataset-name", "--dataset", "-n", help="保存先データセット識別名")
    dl_parser.add_argument("--tags", "-t", help="検索タグ (スペース区切り, 例: '1girl solo')")
    dl_parser.add_argument("--limit", "-l", type=int, help="取得上限枚数")
    dl_parser.add_argument("--min-score", "-s", type=int, help="最小スコアフィルター")

    # サブコマンド: train
    tr_parser = subparsers.add_parser("train", help="WebDAV 内のデータセットから学習を実行")
    tr_parser.add_argument("--dataset-name", "--dataset", "-n", help="学習対象データセット名")
    tr_parser.add_argument(
        "--preset", "-p", choices=["low_vram", "high_quality"], help="モデルプリセット"
    )
    tr_parser.add_argument(
        "--min-frequency", "-f", type=int, help="タグ最小出現回数しきい値"
    )
    tr_parser.add_argument("--epochs", "-e", type=int, help="エポック数")
    tr_parser.add_argument("--batch-size", "-b", type=int, help="バッチサイズ")
    tr_parser.add_argument(
        "--preload",
        dest="preload",
        action="store_true",
        default=None,
        help="全画像をマルチスレッドで RAM にプリロードする (SSD非保存高速化)",
    )
    tr_parser.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        help="RAM プリロードを無効化し逐次ストリーミングする",
    )
    tr_parser.add_argument(
        "--preload-workers", type=int, default=None, help="RAM プリロード時の並列スレッド数 (デフォルト: 16)"
    )

    # サブコマンド: list
    subparsers.add_parser("list", help="Cryptomator 内の保存済みデータセット一覧を表示")

    # サブコマンド: predict
    pred_parser = subparsers.add_parser("predict", help="学習済みモデルを用いて画像からタグを予測")
    pred_parser.add_argument("--checkpoint", "-c", required=True, help="チェックポイント (.pth) のパス")
    pred_parser.add_argument("--image", "-i", required=True, help="画像ファイルパスまたは画像 URL")
    pred_parser.add_argument(
        "--vocab", "-v", default=None, help="タグ語彙ファイル (tag_vocab.json) のパス (未指定時はWebDAVから自動取得)"
    )
    pred_parser.add_argument(
        "--threshold", "-t", type=float, default=0.35, help="検出確率しきい値 (デフォルト: 0.35)"
    )
    pred_parser.add_argument(
        "--top-k", "-k", type=int, default=None, help="出力する最大タグ数 (省略時はしきい値以上の全件)"
    )
    pred_parser.add_argument(
        "--prompt-only", action="store_true", help="カンマ区切りのタグプロンプトのみを出力"
    )
    pred_parser.add_argument("--json", action="store_true", help="JSON 形式で出力")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "download":
        handle_download(args, config)
    elif args.command == "train":
        handle_train(args, config)
    elif args.command == "list":
        handle_list(args, config)
    elif args.command == "predict":
        handle_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
