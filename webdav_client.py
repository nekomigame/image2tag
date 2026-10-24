"""Cryptomator 内蔵 WebDAV サーバーとの直接 HTTP 通信クライアント.

ローカルディスク（SSD）に平文の一時ファイルを生成することなく、
すべてインメモリ（RAM）上でストリーミング送受信を行います。
"""

import io
import urllib.parse
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth


class WebDAVClient:
    """Cryptomator 内蔵 WebDAV と直接 HTTP 通信を行うクライアントクラス."""

    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_type: str = "basic",
        verify_ssl: bool = False,
    ) -> None:
        """WebDAV クライアントを初期化します.

        Args:
            base_url: WebDAVのルートURL (例: 'http://127.0.0.1:42427/MyVault')
            username: 認証ユーザー名 (省略可能)
            password: 認証パスワード (省略可能)
            auth_type: 認証方式 ('basic' または 'digest')
            verify_ssl: SSL証明書検証の有無
        """
        self.base_url: str = base_url.rstrip("/")
        self.session: requests.Session = requests.Session()
        self.verify_ssl: bool = verify_ssl

        if username and password:
            if auth_type.lower() == "digest":
                self.session.auth = HTTPDigestAuth(username, password)
            else:
                self.session.auth = HTTPBasicAuth(username, password)

        # アダプターのコネクションプール設定（ストリーミング高速化）
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=3,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _build_url(self, path: str) -> str:
        """相対パスから完全な URL を組み立てます.

        Args:
            path: WebDAV 内の相対パス

        Returns:
            完全な URL 文字列
        """
        clean_path = path.strip("/")
        return f"{self.base_url}/{clean_path}"

    def mkdir(self, path: str) -> bool:
        """ディレクトリを作成します (MKCOL).

        すでに存在する場合は True を返します。

        Args:
            path: 作成するディレクトリパス

        Returns:
            成功または既に存在する場合は True
        """
        url = self._build_url(path)
        try:
            resp = self.session.request("MKCOL", url, verify=self.verify_ssl, timeout=10)
            if resp.status_code in (201, 405):
                return True
            return resp.ok
        except requests.RequestException:
            return False

    def mkdir_p(self, path: str) -> bool:
        """親ディレクトリを含めて再帰的に作成します.

        Args:
            path: 作成する階層パス

        Returns:
            すべてのディレクトリ作成に成功した場合は True
        """
        parts = [p for p in path.strip("/").split("/") if p]
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else part
            if not self.mkdir(current):
                return False
        return True

    def exists(self, path: str) -> bool:
        """ファイルまたはディレクトリが存在するか確認します (PROPFIND Depth 0).

        Args:
            path: 確認対象パス

        Returns:
            存在する場合は True
        """
        url = self._build_url(path)
        headers = {"Depth": "0"}
        try:
            resp = self.session.request(
                "PROPFIND", url, headers=headers, verify=self.verify_ssl, timeout=10
            )
            return resp.status_code in (200, 207)
        except requests.RequestException:
            return False

    def put_bytes(self, path: str, data: bytes) -> bool:
        """メモリ上のバイト列を直接アップロードします (HTTP PUT).

        ローカルディスクには一切書き込みを行いません。

        Args:
            path: アップロード先パス
            data: アップロードするバイナリデータ

        Returns:
            成功した場合は True
        """
        url = self._build_url(path)
        try:
            resp = self.session.put(url, data=data, verify=self.verify_ssl, timeout=30)
            return resp.status_code in (200, 201, 204)
        except requests.RequestException:
            return False

    def put_stream(self, path: str, stream: io.BytesIO) -> bool:
        """BytesIO ストリームを直接アップロードします (HTTP PUT).

        Args:
            path: アップロード先パス
            stream: バイナリストリーム

        Returns:
            成功した場合は True
        """
        stream.seek(0)
        return self.put_bytes(path, stream.read())

    def get_bytes(self, path: str) -> Optional[bytes]:
        """ファイルをメモリ上に直接ダウンロードします (HTTP GET).

        ローカルディスクに保存せず、バイト列をそのまま返します。

        Args:
            path: 取得対象パス

        Returns:
            取得したバイト列 (失敗時は None)
        """
        url = self._build_url(path)
        try:
            resp = self.session.get(url, stream=True, verify=self.verify_ssl, timeout=30)
            if resp.status_code == 200:
                return resp.content
            return None
        except requests.RequestException:
            return None

    def get_stream(self, path: str) -> Optional[io.BytesIO]:
        """ファイルを BytesIO ストリームとして取得します.

        Args:
            path: 取得対象パス

        Returns:
            io.BytesIO ストリーム (失敗時は None)
        """
        data = self.get_bytes(path)
        if data is not None:
            return io.BytesIO(data)
        return None

    def list_dir(self, path: str) -> List[str]:
        """ディレクトリ内のアイテム一覧を取得します (PROPFIND Depth 1).

        名前空間プレフィックスの相違 (d:, D:, デフォルト名前空間) を吸収して
        安全に解析します。

        Args:
            path: 一覧を取得するディレクトリパス

        Returns:
            ファイル・ディレクトリ名のリスト
        """
        url = self._build_url(path)
        headers = {"Depth": "1"}
        try:
            resp = self.session.request(
                "PROPFIND", url, headers=headers, verify=self.verify_ssl, timeout=15
            )
            if resp.status_code not in (200, 207):
                return []
        except requests.RequestException:
            return []

        items: List[str] = []
        try:
            root = ET.fromstring(resp.content)
            target_path_norm = urllib.parse.urlparse(url).path.rstrip("/")

            # XML名前空間に依存せず、要素のローカル名で探索
            for elem in root.iter():
                elem_tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if elem_tag == "response":
                    href_elem = None
                    for child in elem:
                        child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                        if child_tag == "href":
                            href_elem = child
                            break

                    if href_elem is not None and href_elem.text:
                        item_path = urllib.parse.unquote(href_elem.text).rstrip("/")
                        if item_path != target_path_norm:
                            name = item_path.split("/")[-1]
                            if name and name not in items:
                                items.append(name)
        except Exception:
            return []

        return items
