"""Rule34 データセット収集モジュール.

rule34py を使用してメタデータおよび画像バイナリを取得し、
ローカルディスク（SSD）を一切経由せずに Cryptomator の内蔵 WebDAV サーバーへ
直接暗号化保存します。
"""

from datetime import datetime, timezone
import io
import json
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import requests
from rule34Py import rule34Py
from tqdm import tqdm

from webdav_client import WebDAVClient


class DatasetCollector:
    """Rule34 から画像を収集し、WebDAV へ直接暗号化保存するコレクター."""

    def __init__(self, webdav_client: WebDAVClient) -> None:
        """コレクターを初期化します.

        Args:
            webdav_client: WebDAV クライアントインスタンス
        """
        # .env から環境変数をロード
        load_dotenv()

        self.client: WebDAVClient = webdav_client
        self.r34 = rule34Py()

        # .env から API キーとユーザー ID を適用
        api_key = os.getenv("RULE34_API_KEY", "").strip()
        user_id = os.getenv("RULE34_USER_ID", "").strip()
        if api_key:
            self.r34.api_key = api_key
            print("Rule34 API キーを設定しました。")
        if user_id:
            self.r34.user_id = user_id
            print(f"Rule34 ユーザー ID ({user_id}) を設定しました。")

        self.http_session = requests.Session()

    def collect(
        self,
        dataset_name: str,
        query_tags: List[str],
        limit: int = 1000,
        min_score: int = 0,
        rate_limit_delay: float = 0.5,
    ) -> Dict[str, Any]:
        """指定したタグ条件で画像を検索し、Cryptomator に直接保存します.

        Args:
            dataset_name: 保存先データセット名 (英数字・アンダースコア推奨)
            query_tags: 検索タグのリスト (例: ['1girl', 'solo'])
            limit: 収集上限枚数
            min_score: 最小スコアフィルター
            rate_limit_delay: 各ダウンロード間の待機秒数

        Returns:
            収集結果の統計情報辞書
        """
        # 保存先パスの準備
        dataset_path = f"datasets/{dataset_name}"
        images_path = f"{dataset_path}/images"
        metadata_file = f"{dataset_path}/metadata.jsonl"
        info_file = f"{dataset_path}/dataset_info.json"

        # WebDAV 上にディレクトリ構造を作成
        self.client.mkdir_p(images_path)

        # 既存メタデータがある場合は取得して重複を防止（再開対応）
        existing_ids = set()
        existing_metadata_bytes = self.client.get_bytes(metadata_file)
        metadata_records: List[Dict[str, Any]] = []

        if existing_metadata_bytes:
            for line in existing_metadata_bytes.decode("utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    existing_ids.add(record["id"])
                    metadata_records.append(record)
            print(f"既存のデータセット '{dataset_name}' を検出しました。既存登録画像数: {len(existing_ids)} 枚")

        print(f"Rule34 検索開始: tags={query_tags}, 取得目標: {limit} 枚 (スコア >={min_score})")

        downloaded_count = 0
        page_id = 0
        pbar = tqdm(total=limit, initial=len(existing_ids), desc="画像収集 & WebDAV直接保存")

        def save_metadata_checkpoint() -> None:
            """収集途中のメタデータを WebDAV に保存して再開可能にします."""
            if not metadata_records:
                return
            lines = [json.dumps(r, ensure_ascii=False) for r in metadata_records]
            content = "\n".join(lines) + "\n"
            self.client.put_bytes(metadata_file, content.encode("utf-8"))

            info_data = {
                "dataset_name": dataset_name,
                "query_tags": query_tags,
                "total_images": len(metadata_records),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.client.put_bytes(
                info_file,
                json.dumps(info_data, indent=2, ensure_ascii=False).encode("utf-8"),
            )

        try:
            PAGE_FETCH_LIMIT = 1000  # Rule34 API の 1 ページあたり最大取得件数 (1000 件)

            while len(existing_ids) < limit:
                posts = None
                max_retries = 5
                retry_delay = 60.0  # 429検知時はバケット回復のため60秒待機

                for retry in range(max_retries):
                    try:
                        # API レート制限ガード: 検索リクエスト間にインターバルを確保
                        time.sleep(1.5)
                        posts = self.r34.search(
                            tags=query_tags,
                            limit=PAGE_FETCH_LIMIT,  # 常に最大 1000 件を一括取得
                            page_id=page_id,
                        )
                        break
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "Too Many Requests" in err_str:
                            print(f"\nRule34 レートリミット (429) を検知。API バケット回復のため {retry_delay:.0f}秒待機してリトライします ({retry + 1}/{max_retries})...")
                            time.sleep(retry_delay)
                        else:
                            print(f"\nRule34 API 検索エラー (ページ {page_id}): {e}")
                            break

                if not posts:
                    print("\nこれ以上の画像が見つからないか、検索を終了しました。")
                    break

                for post in posts:
                    if len(existing_ids) >= limit:
                        break

                    post_id = post.id
                    if post_id in existing_ids:
                        continue

                    if post.score < min_score:
                        continue

                    file_url = getattr(post, "image", None) or getattr(post, "sample", None)
                    if not file_url:
                        continue

                    # 拡張子判定
                    ext = file_url.split(".")[-1].split("?")[0].lower()
                    if ext not in ("jpg", "jpeg", "png", "webp"):
                        continue

                    filename = f"{post_id}.{ext}"
                    target_webdav_path = f"{images_path}/{filename}"

                    # 画像をインメモリ（RAM）で取得し、即座に WebDAV へ直接 PUT（SSD非保存）
                    try:
                        resp = self.http_session.get(file_url, timeout=30)
                        if resp.status_code == 200:
                            success = self.client.put_bytes(target_webdav_path, resp.content)
                            if success:
                                # タグリストを文字列として安全に抽出
                                tags_list = []
                                if hasattr(post, "tags") and post.tags:
                                    if isinstance(post.tags, list):
                                        for t in post.tags:
                                            # Tag オブジェクトのプロパティまたは文字列として取得
                                            tag_name = getattr(t, "tag", None) or getattr(t, "name", None) or str(t)
                                            tags_list.append(str(tag_name).strip())
                                    elif isinstance(post.tags, str):
                                        tags_list = post.tags.split()

                                record = {
                                    "id": post_id,
                                    "filename": filename,
                                    "tags": tags_list,
                                    "score": post.score,
                                }
                                metadata_records.append(record)
                                existing_ids.add(post_id)
                                downloaded_count += 1
                                pbar.update(1)

                                # 25枚ごとにメタデータを WebDAV に定期書き込み（クラッシュ対策）
                                if downloaded_count % 25 == 0:
                                    save_metadata_checkpoint()
                    except Exception:
                        pass

                    time.sleep(rate_limit_delay)

                # 取得件数が上限未満の場合はこれ以上のページが存在しないため終了
                if len(posts) < PAGE_FETCH_LIMIT:
                    print("\nこれ以上の画像（最終ページ）に到達したため検索を終了します。")
                    break

                page_id += 1

        finally:
            pbar.close()
            # 完了または中断時に最終メタデータを確実に WebDAV に保存
            save_metadata_checkpoint()

        info_data = {
            "dataset_name": dataset_name,
            "query_tags": query_tags,
            "total_images": len(metadata_records),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.client.put_bytes(info_file, json.dumps(info_data, indent=2, ensure_ascii=False).encode("utf-8"))

        if not metadata_records:
            print(f"\n警告: 条件に一致する画像が 1 件も取得できませんでした。タグスペルや min_score を確認してください。")
        else:
            print(f"\nデータセット '{dataset_name}' の保存完了: 総計 {len(metadata_records)} 枚 (今回新規追加: {downloaded_count} 枚)")
        return info_data
