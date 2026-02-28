# Development Guide

## プロジェクト目的
- 形状入力（エッジ重み付きグラフ + 境界ノード指定）からラプラシアンスペクトラム特徴を計算する。
- 主出力は固有値スペクトラムと固有値ヒストグラム。
- 境界条件は Dirichlet / Neumann を切り替え可能。
- 複数サンプルの一括解析（バッチ特徴、PCA/MDS 可視化）を想定。

## スコープ
### 入力
- `WeightedGraph`（ノード、重み付きエッジ、境界ノード）
- 変換ユーティリティ
  - 二値画像 -> `WeightedGraph`
  - 点群 -> `WeightedGraph`

### 出力
- 固有値スペクトラム（昇順）
- 固有値ヒストグラム（bin / range 指定可能）
- バッチ解析向け特徴量（固定長スペクトラム + ヒストグラム）
- 距離行列（spectrum: L2/Wasserstein, histogram: L1/JS）

## セットアップ
### 1) インストール
```bash
pip install -e .
```

開発依存込み:
```bash
pip install -e ".[dev]"
```

## テスト
```bash
python -m pytest -q
```

## コード構成
### ライブラリ本体
- `src/lapspec/`
  - `types.py`: `WeightedGraph`
  - `graph.py`: エッジリスト -> グラフ
  - `laplacian.py`: 正規化ラプラシアン構築（Dirichlet/Neumann）
  - `spectrum.py`: 固有値計算、ヒストグラム（範囲は実験・設定で指定可能）
  - `metrics.py`: 距離行列（spectrum/histogram）
  - `batch.py`: バッチ特徴抽出
  - `visualization.py`: PCA / MDS 可視化
  - `converters/`: 画像・点群変換

### 実験
- `experiments/shared.py`
  - 実験共通処理（CSV/画像/PCA/H0-H1）
- `experiments/holed_disk.py`
- `experiments/cell_division.py`
- `experiments/pointcloud_torus.py`

### テスト
- `tests/`

## 実験を追加するときの推奨手順
1. 新規スクリプトを `experiments/<name>.py` に追加
2. 形状生成とメタ情報定義のみ実装
3. 出力処理は `experiments/shared.py` を再利用
4. `README.md` と `docs/EXPERIMENTS.md` に実行方法を追記

## 共通処理の責務
`experiments/shared.py` は以下の責務を持ちます。
- overlay 画像生成
- 重み付きグラフ一覧 CSV 出力
- Dirichlet/Neumann 特徴計算
- ヒストグラム・固有値・PCA/MDS CSV 出力
- 距離行列 CSV 出力
- H0/H1 計算と色付け PCA 出力

新規実験で同種の処理が必要な場合は、各実験へコピーせず shared 側へ追加してください。

## 実装上の判断基準
- 優先事項
  - 再現性（乱数 seed、bin/range 設定）
  - 拡張性（境界条件・特徴量・距離指標の追加容易性）
  - 計算効率（疎行列、バッチ処理）
- 注意点
  - 境界ノード定義の曖昧さ
  - 固有値スケール依存性
  - 点群グラフ化時のパラメータ感度（`k` / `radius`）
