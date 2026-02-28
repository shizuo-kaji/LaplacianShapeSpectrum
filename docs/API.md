# API Reference

本ドキュメントは `src/lapspec` の公開 API と挙動をまとめたものです。

## Data Model
### `WeightedGraph`
定義: [types.py](../src/lapspec/types.py#L12)

- `num_nodes: int`
- `edges: NDArray[int64]` shape `(m, 2)`
- `weights: NDArray[float64]` shape `(m,)`
- `boundary_nodes: NDArray[int64]` shape `(b,)`
- `node_positions: NDArray[float64] | None` shape `(num_nodes, dim)`

制約:
- ノード番号は `0..num_nodes-1`
- 自己ループ禁止
- エッジ重みは正
- `boundary_nodes` も有効範囲内

## Graph Construction
### `from_edge_list(...) -> WeightedGraph`
定義: [graph.py](../src/lapspec/graph.py#L23)

機能:
- 入力配列を `numpy` 配列へ正規化
- 重複エッジを自動統合（無向化のため `(u,v)` と `(v,u)` は同一扱い）
- 重複重みは合算

## Laplacian
### `build_adjacency(graph) -> csr_matrix`
定義: [laplacian.py](../src/lapspec/laplacian.py#L22)

### `build_laplacian(graph, boundary="neumann") -> LaplacianResult`
定義: [laplacian.py](../src/lapspec/laplacian.py#L40)

境界条件:
- `neumann`: 全ノードの正規化ラプラシアン `L = I - D^{-1/2} A D^{-1/2}`
- `dirichlet`: 上記行列から境界ノードを除いた内部ノード部分行列へ縮約

戻り値 `LaplacianResult`:
- `matrix`: 計算対象ラプラシアン（`csr_matrix`）
- `active_nodes`: 行列に対応する元ノード ID
- `boundary_condition`: `"dirichlet"` or `"neumann"`

## Spectrum
### `compute_spectrum(matrix, k=None, tol=1e-10, eigsh_kwargs=None) -> ndarray`
定義: [spectrum.py](../src/lapspec/spectrum.py#L16)

仕様:
- `k=None` または `k>=n` のとき全固有値（密行列で `eigvalsh`）
- 疎行列 + `k<n` のとき `scipy.sparse.linalg.eigsh` を利用
- 数値誤差近傍（`abs(x)<tol`）は `0.0` に丸め

### `spectrum_histogram(eigenvalues, bins=32, value_range=None, density=True)`
定義: [spectrum.py](../src/lapspec/spectrum.py#L76)

戻り値:
- `hist`, `edges`（どちらも `float64`）

補足:
- 正規化ラプラシアンの固有値は理論上 `[0, 2]`
- 本プロジェクトの特徴量計算では `hist_range` で範囲指定可能（既定は `[0, 2]`）

### `fixed_length_spectrum(eigenvalues, length, fill_value=0.0)`
定義: [spectrum.py](../src/lapspec/spectrum.py#L88)

## Converters
### `image_to_graph(mask, connectivity=4|8, weight_mode="unit"|"distance")`
定義: [image.py](../src/lapspec/converters/image.py#L31)

仕様:
- 前景画素をノード化
- `connectivity` で近傍辺を構築
- `weight_mode="distance"` では重み `1 / pixel_distance`
- 境界ノードは 4-neighborhood で背景/外周に接する画素として推定

### `pointcloud_to_graph(...)`
定義: [pointcloud.py](../src/lapspec/converters/pointcloud.py#L41)

主な引数:
- `method`: `"knn"` or `"radius"`
- `weight_mode`: `"unit" | "inverse_distance" | "gaussian"`
- `boundary_mode`: `"outer_percentile" | "none"`

## Batch Features
### `FeatureConfig`
定義: [batch.py](../src/lapspec/batch.py#L16)

補足:
- `hist_range` を指定するとヒストグラム範囲を変更可能
- `hist_range=None` の場合は既定 `(0.0, 2.0)` を使用

### `graph_feature_vector(graph, config)`
定義: [batch.py](../src/lapspec/batch.py#L26)

出力ベクトル:
- `[固定長スペクトラム(spectrum_k), ヒストグラム(hist_bins)]`

### `batch_features(graphs, ...)`
定義: [batch.py](../src/lapspec/batch.py#L50)

戻り値:
- shape `(num_graphs, spectrum_k + hist_bins)`

## Distance Metrics
### `spectrum_distance_matrix(spectra, metric="l2"|"wasserstein")`
定義: [metrics.py](../src/lapspec/metrics.py)

仕様:
- 入力 `spectra` は shape `(num_samples, spectrum_dim)` の2次元配列
- 戻り値は shape `(num_samples, num_samples)` の対称距離行列

### `histogram_distance_matrix(histograms, metric="l1"|"js", normalize=True)`
定義: [metrics.py](../src/lapspec/metrics.py)

仕様:
- 入力 `histograms` は shape `(num_samples, num_bins)` の2次元配列
- `metric="l1"` は L1 距離、`metric="js"` は Jensen-Shannon divergence
- 戻り値は shape `(num_samples, num_samples)` の対称距離行列

## PCA / Visualization
### `pca_projection(features, n_components=2, standardize=True, random_state=0)`
定義: [visualization.py](../src/lapspec/visualization.py#L12)

### `mds_projection(distance_matrix, n_components=2, random_state=0)`
定義: [visualization.py](../src/lapspec/visualization.py)

### `plot_pca(features, labels=None, save_path=None, title=...)`
定義: [visualization.py](../src/lapspec/visualization.py#L35)
