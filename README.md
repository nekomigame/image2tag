# Swin Transformer v2 画像・タグ学習システム

Swin Transformer v2 を用いたイラスト画像に対するマルチラベルタグ分類モデルの学習システムです。

## 特徴
1. **SSD平文非保存 (Zero-Disk Persistence)**:
   - ダウンロードした画像バイナリをローカルSSD（一時フォルダー等）に一切保存せず、RAMバッファ（`io.BytesIO`）から Cryptomator の内蔵 WebDAV サーバーへ直接 `HTTP PUT` で暗号化保存します。
   - 学習時も WebDAV から直接 `HTTP GET` でストリーミング取得し、RAM 上でのみデコード・学習を行います。
2. **複数パターンのデータセット保存 & 再学習**:
   - データセット名（`--dataset-name`）ごとに独立したフォルダー構造で保存・管理。
   - Rule34 に再アクセスすることなく、Cryptomator 内の蓄積データから過去のデータセットを指定して何度でも再学習が可能です。
3. **プリセット切り替え**:
   - 省メモリモード (`low_vram`: SwinV2-Tiny, 256px, VRAM 6~8GB)
   - 標準/高精度モード (`high_quality`: SwinV2-Base, 256px, VRAM 12~24GB)
4. **タグ出現回数による絞り込み**:
   - 最小出現回数（`--min-frequency`）を指定して、ノイズタグを自動除外。

---

## 準備

### 1. API キー・環境変数の設定 (`.env`)
`.env.example` を参考に、プロジェクトルートの `.env` ファイルに Rule34 の API キーを設定します（`.env` は Git 追跡対象外です）。

```env
RULE34_API_KEY="あなたのAPIキー"
RULE34_USER_ID="あなたのユーザーID"
```

### 2. 設定ファイルの作成
`example.config.yaml` をコピーして `config.yaml` を作成します（`config.yaml` はローカル専用で Git 追跡対象外です）。

```powershell
copy example.config.yaml config.yaml
```

### 3. Cryptomator の設定
1. Cryptomator を起動し、対象の保管庫（Vault）をアンロックします。
2. Cryptomator の「設定」または保管庫オプションを開き、仮想ドライブタイプを「WebDAV」に設定（または内蔵 WebDAV サーバーを有効化）。
3. 表示される WebDAV URL（例: `http://127.0.0.1:42427/MyVault`）を確認し、`config.yaml` の `storage.webdav_url` に設定します。

---

## 使い方

### 1. データセットの収集・Cryptomator への直接保存 (`download`)
Rule34 からタグと枚数を指定してダウンロードし、Cryptomator の WebDAV へ直接暗号化保存します。

```powershell
# 例: '1girl solo' タグで 500枚収集し、データセット名 'solo_1girl_v1' として保存
uv run python main.py download --dataset-name "solo_1girl_v1" --tags "1girl solo" --limit 500

# 別のパターンを保存したい場合
uv run python main.py download --dataset-name "vtuber_blue_eyes" --tags "vtuber blue_eyes" --limit 1000
```

### 2. 保存済みデータセットの一覧確認 (`list`)
Cryptomator 内に暗号化保存されているデータセット一覧、枚数、収集タグを確認できます。

```powershell
uv run python main.py list
```

### 3. 学習・再学習の実行 (`train`)
保存済みのデータセットを指定して、Swin Transformer v2 の学習を開始します。

```powershell
# 省メモリモード（VRAM 6~8GB向け）で、出現回数 5回以上のタグを学習
uv run python main.py train --dataset-name "solo_1girl_v1" --preset low_vram --min-frequency 5 --epochs 20

# 標準/高精度モードで再学習する場合
uv run python main.py train --dataset-name "solo_1girl_v1" --preset high_quality --min-frequency 10 --epochs 30
```

### 4. 学習済みモデルを使ったタグ予測・推論 (`predict`)
学習したチェックポイント（`.pth`）を使って、画像（ローカルファイルまたは Web URL）からタグを予測します。

```powershell
# 信頼度付きで予測結果を表示
uv run python main.py predict --checkpoint "./checkpoints/test_overwatch_best_model.pth" --image "sample.jpg"

# Web上の画像URLから直接予測
uv run python main.py predict --checkpoint "./checkpoints/test_overwatch_best_model.pth" --image "https://example.com/image.jpg"

# カンマ区切りのプロンプト形式（Danbooru / Stable Diffusion 形式）のみを出力
uv run python main.py predict --checkpoint "./checkpoints/test_overwatch_best_model.pth" --image "sample.jpg" --prompt-only

# 検出確率のしきい値や最大取得数を指定
uv run python main.py predict --checkpoint "./checkpoints/test_overwatch_best_model.pth" --image "sample.jpg" --threshold 0.4 --top-k 15
```

#### Python コードから直接呼び出す場合
```python
from predict import TagPredictor

# モデルの読み込み
predictor = TagPredictor(checkpoint_path="./checkpoints/test_overwatch_best_model.pth")

# 1. 信頼度付きリストを取得: [('tracer_(overwatch)', 0.98), ('1girl', 0.95), ...]
tags_with_scores = predictor.predict("sample.jpg", threshold=0.35)

# 2. カンマ区切りのプロンプト文字列を取得: "tracer_(overwatch), 1girl, solo, goggles"
prompt_text = predictor.predict_to_prompt("sample.jpg", threshold=0.35)
print(prompt_text)
```

### 5. Google Colab での実行 (`image2tag_colab.ipynb`)
Google Colab 上で Google Drive に配置した Cryptomator Vault をアンロックし、GPU（T4 / V100 / A100）を活用してストリーミング学習・推論を行うことができます。

1. [image2tag_colab.ipynb](file:///C:/Users/nekom/Desktop/python/image2tag/image2tag_colab.ipynb) を Google Colab にアップロードまたは開きます。
2. ノートブック内の手順に従ってセルを上から順に実行します：
   - Google Drive をマウント（`/content/drive`）
   - Linux 向け Cryptomator CLI（`cryptomator-cli`）を自動ダウンロード & セットアップ
   - Google Drive 内の Vault パス（例: `/content/drive/MyDrive/MyVault`）とパスワードを入力して WebDAV をバックグラウンド起動
   - WebDAV URL が自動検出され、`config.yaml` に自動設定されます
   - Colab のローカルディスクに平文画像を一切書き込むことなく、RAM プリロード経由で高速に Swin Transformer v2 を学習・推論します。

---

## モジュール構成
- [main.py](file:///C:/Users/nekom/Desktop/python/image2tag/main.py): CLI 統合エントリーポイント
- [collector.py](file:///C:/Users/nekom/Desktop/python/image2tag/collector.py): Rule34 画像収集 & WebDAV 直接 PUT
- [webdav_client.py](file:///C:/Users/nekom/Desktop/python/image2tag/webdav_client.py): Cryptomator 内蔵 WebDAV クライアント（インメモリ直接通信）
- [dataset.py](file:///C:/Users/nekom/Desktop/python/image2tag/dataset.py): WebDAV ストリーミング PyTorch Dataset
- [model.py](file:///C:/Users/nekom/Desktop/python/image2tag/model.py): Swin Transformer v2 & 分類ヘッド
- [loss.py](file:///C:/Users/nekom/Desktop/python/image2tag/loss.py): Asymmetric Loss (不均衡対応マルチラベル損失)
- [train.py](file:///C:/Users/nekom/Desktop/python/image2tag/train.py): 学習パイプライン
- [config.yaml](file:///C:/Users/nekom/Desktop/python/image2tag/config.yaml): 共通設定ファイル
