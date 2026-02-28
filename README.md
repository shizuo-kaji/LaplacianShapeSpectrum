# Shape Analysis with Laplacian Spectrum (`lapspec`)

形状をラプラシアンスペクトラムで特徴化する Python ライブラリです。
入力は「エッジ重み付きグラフ + 境界ノード指定」を基本形式とし、二値画像・点群からの変換もサポートします。

## Documentation
- API 仕様: [docs/API.md](docs/API.md)
- 実験ガイド: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- 開発ガイド: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## 主な機能
- 重み付きグラフから正規化ラプラシアン（Normalized Laplacian）を構築
- 境界条件を選択可能:
  - Dirichlet（境界固定）
  - Neumann（全ノード対象）
- 固有値スペクトラム・ヒストグラム計算（正規化ラプラシアンのため固有値は理論上 `[0, 2]`）
- ヒストグラム範囲は指定可能（既定は `[0, 2]`）
- バッチ特徴抽出（`固定長スペクトラム + ヒストグラム`）
- PCA 可視化
- 距離行列計算（スペクトラム: `L2` / `Wasserstein`, ヒストグラム: `L1` / `JS divergence`）
- 距離行列の MDS 可視化
- synthetic 実験（穴あき円盤、くびれ分裂楕円、3D点群の楕円体→トーラス）

## インストール
```bash
pip install -e .
```

開発用:
```bash
pip install -e ".[dev]"
```

## 最小使用例
```python
import numpy as np
from lapspec import from_edge_list, build_laplacian, compute_spectrum

graph = from_edge_list(
    num_nodes=4,
    edges=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
    boundary_nodes=np.array([0, 3], dtype=np.int64),
)
lap = build_laplacian(graph, boundary="dirichlet").matrix
eigs = compute_spectrum(lap, k=None)
print(eigs)
```

## 実験実行
```bash
python experiments/holed_disk.py
python experiments/cell_division.py
python experiments/pointcloud_torus.py
```
