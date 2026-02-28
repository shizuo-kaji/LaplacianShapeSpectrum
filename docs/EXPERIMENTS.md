# Experiments Guide

実験スクリプトは `experiments/` にあり、共通処理は [shared.py](../experiments/shared.py) に集約されています。

## 実験一覧
### 1) Holed Disk
スクリプト: [holed_disk.py](../experiments/holed_disk.py)

概要:
- 円盤領域に最大 5 個の穴が同時に開く系列
  - 複数中心 `c_i` を固定し、各半径 `r_i(t)` を時間パラメータ `t` で変化
  - `r_i(t)=0` をまたぐ設定で穴の生成/消失を発生させる
- 各サンプルに対して画像 -> 重み付きグラフ変換
- Dirichlet / Neumann のスペクトラム特徴を比較
- H0 / H1 を計算し、PCA 色付け図を出力

実行:
```bash
python experiments/holed_disk.py
```

既定出力:
- `experiments/output/holed_disk/`

### 2) Cell Division
スクリプト: [cell_division.py](../experiments/cell_division.py)

概要:
- 楕円がくびれて分裂していく形状系列
- 5 ケース（軸長・角度・くびれ進行・非対称度が異なる）
- Dirichlet / Neumann 比較、H0 / H1 計算、PCA 出力

実行:
```bash
python experiments/cell_division.py
```

既定出力:
- `experiments/output/cell_division/`

### 3) Pointcloud Torus Transition
スクリプト: [pointcloud_torus.py](../experiments/pointcloud_torus.py)

概要:
- 3D 点群で「楕円体 -> くぼみ形成 -> 貫通 -> トーラス」に遷移する系列
- 5 ケース（初期楕円体軸長・くぼみ半径・進行速度が異なる）
- 点群から重み付きグラフへ変換し、Dirichlet / Neumann の特徴を比較
- H0/H1（H1 は中心軸貫通判定の proxy）で PCA 色付け

実行:
```bash
python experiments/pointcloud_torus.py
```

既定出力:
- `experiments/output/pointcloud_torus/`

## 共通出力フォーマット
各実験の出力ディレクトリには以下が生成されます。

特徴量計算の前提:
- ラプラシアンは正規化ラプラシアンを使用
- ヒストグラム範囲は実験ごとに指定（データ分布に合わせて選択）

- `test_images/*.png`
  - 二値形状 + グラフ overlay（エッジ/境界）
- `test_images_overview.png`
  - ID 付きサムネイル一覧
- `test_images_index.csv`
  - 各サンプルの ID と画像パス、実験メタ情報
- `weighted_graph_summary.csv`
  - サンプルごとのノード数/エッジ数/密度/重み統計
- `weighted_graph_index.csv`
  - `graph_XXX_edges.csv` / `graph_XXX_boundary_nodes.csv` への索引
- `pointcloud_index.csv`, `pointcloud_summary.csv`（pointcloud 実験）
  - 各サンプル点群 CSV への索引・点群統計
- `dirichlet_features.csv`, `neumann_features.csv`
  - 連結特徴量（固定長スペクトラム + ヒストグラム）
- `dirichlet_histograms.csv`, `neumann_histograms.csv`
- `dirichlet_eigenvalues.csv`, `neumann_eigenvalues.csv`
- `*_distance_matrix.csv`
  - スペクトラム距離（`l2`, `wasserstein`）
  - ヒストグラム距離（`l1`, `js`）
- `pca_dirichlet.csv`, `pca_neumann.csv`
  - PCA 座標 + メタ情報
- `pca_trajectory_id_labeled.png`
  - サンプル ID 注記入り PCA
- `mds_*_dirichlet.csv`, `mds_*_neumann.csv`
  - 距離行列から得た MDS 座標
- `mds_*_id_labeled.png`
  - サンプル ID 注記入り MDS
- `homology.csv`
  - `sample_id,H0,H1`
- `pca_homology_h0_h1.png`
  - H0/H1 で色付けした PCA 図（Dirichlet/Neumann）
- `summary.txt`
  - 実験サマリ（次元数、分散寄与率、ステップ距離平均など）

## H0 / H1 の定義（実装準拠）
実装: [shared.py](../experiments/shared.py#L333)

- `H0`
  - 前景連結成分数（サイズ閾値あり）
- `H1`
  - `binary_fill_holes` を使って抽出した穴成分数（サイズ閾値あり）

補足:
- `holed_disk` は `min_component_pixels=1, min_hole_pixels=1`
- `cell_division` はノイズ抑制のため `min_component_pixels=40, min_hole_pixels=40`
- `case_id` を持つ実験では、PCA/MDS 図に case ごとの marker を付与
- `spectrum_histogram_list.png` では左端に case 括りを表示
