# Changelog

## [1.4.3] — 2026-07-17

### Performance

- `erase_first_loop_sweep_numba` now applies a **2D AABB filter** (x-axis sweep-line break + y-axis overlap rejection) instead of only the x-axis filter. The y-axis check eliminates most candidate pairs that survive the x-filter for typical contours, reducing cross-product calls by another 6–8×.

  Combined speedup over the original O(K²) AABB scan:

  | K | Before (1.4.0 AABB) | After (1.4.3 sweep+2D) | Speedup |
  |---|---|---|---|
  | 100 | 30 µs | 1.8 µs | 17× |
  | 200 | 123 µs | 3.3 µs | 37× |
  | 500 | 772 µs | 9.2 µs | 84× |
  | 1000 | 3.1 ms | 19.5 µs | 159× |
  | 2000 | 12.5 ms | 43.6 µs | 286× |

---

## [1.4.2] — 2026-07-17

### Performance

- `CleanContours.remove_small_loops` now uses a **sweep-line algorithm** (`erase_first_loop_sweep_numba`) instead of the previous O(K²) AABB scan. Segments are sorted once by `min_x` (O(K log K)); the inner loop breaks as soon as the x-ranges of two segments stop overlapping, reducing the number of pair checks from O(K²) to O(K·ε) where ε is the average number of segments with overlapping x-range (typically 2–10 for well-distributed contours).

  Measured speedups on a single contour:

  | K | Before (AABB) | After (sweep) | Speedup |
  |---|---|---|---|
  | 200 | 0.12 ms | 0.03 ms | 4× |
  | 500 | 0.77 ms | 0.07 ms | 10× |
  | 1000 | 3.1 ms | 0.14 ms | 22× |
  | 2000 | 12.5 ms | 0.30 ms | 42× |

  The old `erase_first_encounter_loop_numba` (AABB) is kept for backward compatibility.

---

## [1.4.1] — 2026-07-17

### Bug fixes / performance

- Removed `ThreadPoolExecutor` from `CleanContours.clean_contours` and `clean_contours_and_interpolate`. Benchmarks showed the thread pool overhead consistently exceeded the compute time per contour at all tested scales, making it slower than the straightforward sequential loop. The real performance gains are: numba `cache=True`, AABB pre-filter, and early exit in `is_intersecting_numba`.

---

## [1.4.0] — 2026-07-16

### New features

- **`ContourToSDF`** — differentiable signed distance function layer (positive inside, negative outside, zero on boundary).
- **`sample_features_on_contour`** — differentiable bilinear feature sampling at contour node coordinates using `grid_sample`; gradients flow to both the feature map and the contour coordinates. Output shape `(B, N, K, C)`.
- **`normals`** — unit outward normal vectors at each contour node via central finite differences (output `(B, N, K, 2)`).
- **`compactness`** — isoperimetric compactness `4π·area/perimeter²` ∈ (0, 1]; equals 1 for a perfect circle (output `(B, N)`).
- **`iou`** — differentiable Intersection over Union on soft masks `(B, N, H, W)` (output `(B, N)`).
- **`dice`** — differentiable Dice coefficient on soft masks `(B, N, H, W)` (output `(B, N)`).
- **`spacing_loss`** — penalises variance in inter-node arc lengths to encourage uniform spacing (output `(B, N)`).
- **`smoothness_loss`** — penalises high curvature via second-order finite differences (output `(B, N)`).
- **`balloon_loss`** — signed area alias; minimise to shrink, negate to expand (output `(B, N)`).

### Architecture improvements

- Introduced `_ContourBase` base class: shared mesh buffer (`register_buffer`) and `_compute()` winding-number core, eliminating ~100 lines of duplicated code across `ContourToMask`, `ContourToDistanceMap`, `ContourToIsolines`, and the new `ContourToSDF`.
- PascalCase renaming: `ContourToMask`, `ContourToDistanceMap`, `ContourToIsolines`, `DrawContour`. Old underscore names (`Contour_to_mask`, etc.) kept as backward-compatible aliases.
- `Smoothing.kernel` registered as a buffer via `register_buffer` — device placement is now automatic.
- `ContourToIsolines.vars` (isoline variances) registered as a buffer — no more numpy→tensor conversion on every forward call.
- `CleanContours` methods converted to `@staticmethod`; class is now a pure namespace.
- `_validate_contour()` helper: raises `ValueError` on wrong shape or out-of-range values.

### Performance improvements

- Numba JIT functions now compiled with `cache=True` — no recompilation cost across Python sessions.
- AABB bounding-box pre-filter in `erase_first_encounter_loop_numba` — skips cross-product tests for clearly disjoint segments.
- Early exit in `is_intersecting_numba` — avoids the second pair of cross products when the first test already rules out intersection.
- `CleanContours.clean_contours` and `clean_contours_and_interpolate` now process the batch in parallel via `ThreadPoolExecutor` (numba releases the GIL).
- Mesh `expand` instead of `repeat` — no extra memory allocation.
- `torch.norm` replaced with `torch.linalg.vector_norm` throughout.

### Bug fixes

- **`Sobel.forward`**: in-place border-zeroing (`x[:, :, 0, :] = 0` etc.) on a tensor that is part of the autograd graph caused a `RuntimeError` during `backward()`. Fixed by replacing with an out-of-place `F.pad` operation.
- **`ContourToIsolines.forward`**: the raw winding-number mask can exceed 1.0 near contour edges; the output was no longer bounded. Fixed by clamping the mask to `[0, 1]` before the Gaussian multiplication.
- **`normals`**: rotation direction was `(−ty, tx)`, which produces *inward* normals for clockwise (image-convention) contours. Corrected to `(ty, −tx)`.
- **`Gaussian kernel` in `Smoothing`**: formula was `(−1/2) · x² / (2σ²) = −x²/(4σ²)`, making the effective standard deviation `√2 · σ`. Fixed to `−x²/(2σ²)`.
- **`hausdorff_distance`**: `torch.cdist` with the default MM-accelerated mode gives tiny non-zero values on the diagonal for identical inputs. Fixed by passing `compute_mode='donot_use_mm_for_euclid_dist'`.
- **`DrawContour` for N > 1**: `Sobel` has `in_channels=1` but the mask had shape `(B, N, H, W)`. Fixed by reshaping to `(B*N, 1, H, W)` before the Sobel pass.
- **`DrawContour` normalisation**: min/max had shape `(B, N)` which broadcast incorrectly against `(B, N, H*W)`. Fixed with `.unsqueeze(-1)`.
- `torch.pi = torch.acos(...) * 2` was overwriting a PyTorch module attribute on every forward call. Replaced with `import math` / `math.pi`.
- `hausdorff_distance` and `iou`/`dice` now raise `ValueError` on shape mismatch between the two inputs.

### Documentation

- README fully rewritten with six sections: contour-to-image layers, feature sampling, geometric descriptors, similarity metrics, regularization losses, and NumPy utilities.
- Corrected `curvature` output shape: `(B, N, K)` (padding is internal), not `(B, N, K−6)` as previously documented.
- Updated `normals` convention to match the fix: outward normals for clockwise contours (image space, y-axis down).

### Tests

- Added `tests/test_torch_contour.py` with **87 tests** covering every public API function and class:
  - Output shapes, value ranges, mathematical correctness (shoelace area, circle perimeter/compactness, constant curvature).
  - Differentiability (`loss.backward()` + gradient checks) for all differentiable operations.
  - Input validation (out-of-range values, wrong ndim, shape mismatches).
  - `CleanContours` utilities (loop removal, interpolation, batch processing).

---

## [1.3.1] and earlier

See git history.
