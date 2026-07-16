"""Tests for torch_contour."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torch_contour import (
    ContourToMask,
    ContourToDistanceMap,
    ContourToSDF,
    ContourToIsolines,
    DrawContour,
    Smoothing,
    Sobel,
    area,
    perimeter,
    compactness,
    normals,
    curvature,
    hausdorff_distance,
    iou,
    dice,
    sample_features_on_contour,
    spacing_loss,
    smoothness_loss,
    balloon_loss,
    CleanContours,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIZE = 32   # small image size — keeps tests fast
EPS  = 1e-5


def make_square(b: int = 1, n: int = 1) -> torch.Tensor:
    """Square with corners at (0.1,0.1)–(0.9,0.9), shape (b, n, 4, 2).

    Exact area  = 0.64, exact perimeter = 3.2.
    """
    nodes = torch.tensor(
        [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        dtype=torch.float32,
    )
    return nodes.unsqueeze(0).unsqueeze(0).expand(b, n, -1, -1).clone()


def make_circle(k: int = 40, r: float = 0.3, b: int = 1, n: int = 1) -> torch.Tensor:
    """Uniformly-sampled circle centred at (0.5, 0.5), shape (b, n, k, 2)."""
    t = torch.linspace(0, 2 * math.pi, k + 1)[:-1]
    pts = torch.stack([0.5 + r * torch.cos(t), 0.5 + r * torch.sin(t)], dim=-1)
    return pts.unsqueeze(0).unsqueeze(0).expand(b, n, -1, -1).clone()


# ---------------------------------------------------------------------------
# Contour-to-image layers
# ---------------------------------------------------------------------------

class TestContourToMask:
    ctm = ContourToMask(size=SIZE)

    def test_output_shape(self):
        c = make_circle(b=2, n=3)
        assert self.ctm(c).shape == (2, 3, SIZE, SIZE)

    def test_values_in_range(self):
        c = make_circle()
        out = self.ctm(c)
        assert out.min() >= -EPS
        assert out.max() <= 1 + EPS

    def test_inside_is_one(self):
        """Centre of a large circle should be masked as inside (≈ 1)."""
        c = make_circle(k=60, r=0.4)
        mask = self.ctm(c)
        centre = mask[0, 0, SIZE // 2, SIZE // 2]
        assert centre > 0.9, f"expected centre ≈ 1, got {centre:.3f}"

    def test_outside_is_zero(self):
        """Corner pixel should be masked as outside (≈ 0) for a small circle."""
        c = make_circle(r=0.2)
        mask = self.ctm(c)
        corner = mask[0, 0, 0, 0]
        assert corner < 0.1, f"expected corner ≈ 0, got {corner:.3f}"

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        out = self.ctm(c)
        out.sum().backward()
        assert c.grad is not None
        assert not torch.isnan(c.grad).any()

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            self.ctm(make_circle() * 2)   # values > 1


class TestContourToDistanceMap:
    ctd = ContourToDistanceMap(size=SIZE)

    def test_output_shape(self):
        c = make_circle(b=2, n=3)
        assert self.ctd(c).shape == (2, 3, SIZE, SIZE)

    def test_values_in_range(self):
        c = make_circle()
        out = self.ctd(c)
        assert out.min() >= -EPS
        assert out.max() <= 1 + EPS

    def test_return_mask(self):
        c = make_circle()
        dmap, mask = self.ctd(c, return_mask=True)
        assert dmap.shape == (1, 1, SIZE, SIZE)
        assert mask.shape == (1, 1, SIZE, SIZE)

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        self.ctd(c).sum().backward()
        assert c.grad is not None
        assert not torch.isnan(c.grad).any()


class TestContourToSDF:
    sdf_layer = ContourToSDF(size=SIZE)

    def test_output_shape(self):
        c = make_circle(b=2, n=3)
        assert self.sdf_layer(c).shape == (2, 3, SIZE, SIZE)

    def test_positive_inside(self):
        """SDF at the centre of the circle must be positive (inside)."""
        c = make_circle(k=60, r=0.4)
        sdf = self.sdf_layer(c)
        assert sdf[0, 0, SIZE // 2, SIZE // 2].item() > 0

    def test_negative_outside(self):
        """SDF at the corner must be negative (outside)."""
        c = make_circle(r=0.2)
        sdf = self.sdf_layer(c)
        assert sdf[0, 0, 0, 0].item() < 0

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        self.sdf_layer(c).sum().backward()
        assert c.grad is not None
        assert not torch.isnan(c.grad).any()


class TestContourToIsolines:
    iso_layer = ContourToIsolines(size=SIZE, isolines=[0.2, 0.5, 0.8])

    def test_output_shape(self):
        c = make_circle(b=2, n=3)
        # (B, N, I, H, W)  with I = 3
        assert self.iso_layer(c).shape == (2, 3, 3, SIZE, SIZE)

    def test_values_in_range(self):
        c = make_circle()
        out = self.iso_layer(c)
        assert out.min() >= -EPS
        assert out.max() <= 1 + EPS

    def test_invalid_isoline_raises(self):
        with pytest.raises(ValueError):
            ContourToIsolines(size=SIZE, isolines=[0.5, 1.5])

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        self.iso_layer(c).sum().backward()
        assert c.grad is not None
        assert not torch.isnan(c.grad).any()


class TestDrawContour:
    draw = DrawContour(size=SIZE)

    def test_output_shape(self):
        c = make_circle(b=2, n=3)
        assert self.draw(c).shape == (2, 3, SIZE, SIZE)

    def test_values_in_range(self):
        c = make_circle()
        out = self.draw(c)
        assert out.min() >= -EPS
        assert out.max() <= 1 + EPS

    def test_multiple_polygons(self):
        """DrawContour must work with N > 1 (Sobel in_channels=1 bug check)."""
        c = make_circle(b=1, n=4)
        out = self.draw(c)
        assert out.shape == (1, 4, SIZE, SIZE)

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        self.draw(c).sum().backward()
        assert c.grad is not None
        assert not torch.isnan(c.grad).any()


class TestSmoothing:
    smoother = Smoothing(sigma=2)

    def test_output_shape_preserved(self):
        c = make_circle(k=40, b=2, n=3)
        assert self.smoother(c).shape == c.shape

    def test_reduces_smoothness_loss(self):
        """Smoothing should lower the smoothness loss."""
        torch.manual_seed(0)
        c = make_circle(k=40) + torch.randn(1, 1, 40, 2) * 0.02
        c = c.clamp(0.05, 0.95)
        loss_before = smoothness_loss(c).item()
        loss_after  = smoothness_loss(self.smoother(c)).item()
        assert loss_after < loss_before

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        self.smoother(c).sum().backward()
        assert c.grad is not None
        assert not torch.isnan(c.grad).any()


class TestSobel:
    sobel = Sobel()

    def test_output_shape(self):
        img = torch.rand(2, 1, SIZE, SIZE)
        assert self.sobel(img).shape == (2, 1, SIZE, SIZE)

    def test_constant_image_is_zero(self):
        """A constant image has no edges."""
        img = torch.ones(1, 1, SIZE, SIZE)
        out = self.sobel(img)
        # border pixels are zeroed; interior should also be zero for a constant img
        assert out[:, :, 1:-1, 1:-1].abs().max().item() < EPS


# ---------------------------------------------------------------------------
# Geometric descriptors
# ---------------------------------------------------------------------------

class TestArea:
    def test_output_shape(self):
        assert area(make_circle(b=2, n=3)).shape == (2, 3)

    def test_square_exact(self):
        """Shoelace formula must give 0.64 for the unit square."""
        a = area(make_square())
        assert torch.allclose(a, torch.tensor([[0.64]]), atol=1e-5)

    def test_circle_approx(self):
        """Area of a dense circle ≈ π r²."""
        r = 0.3
        a = area(make_circle(k=200, r=r)).item()
        assert abs(a - math.pi * r ** 2) < 0.005

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        area(c).sum().backward()
        assert c.grad is not None


class TestPerimeter:
    def test_output_shape(self):
        assert perimeter(make_circle(b=2, n=3)).shape == (2, 3)

    def test_square_exact(self):
        """Perimeter of the unit square must be 3.2."""
        p = perimeter(make_square())
        assert torch.allclose(p, torch.tensor([[3.2]]), atol=1e-5)

    def test_circle_approx(self):
        """Perimeter of a dense circle ≈ 2πr."""
        r = 0.3
        p = perimeter(make_circle(k=200, r=r)).item()
        assert abs(p - 2 * math.pi * r) < 0.005

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        perimeter(c).sum().backward()
        assert c.grad is not None


class TestCompactness:
    def test_output_shape(self):
        assert compactness(make_circle(b=2, n=3)).shape == (2, 3)

    def test_circle_near_one(self):
        """Compactness of a dense circle must be close to 1."""
        c = compactness(make_circle(k=200)).item()
        assert abs(c - 1.0) < 0.01

    def test_square_less_than_circle(self):
        """Square is less compact than a circle."""
        c_circle = compactness(make_circle(k=200)).item()
        c_square = compactness(make_square()).item()
        assert c_square < c_circle

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        compactness(c).sum().backward()
        assert c.grad is not None


class TestNormals:
    def test_output_shape(self):
        c = make_circle(k=40, b=2, n=3)
        assert normals(c).shape == (2, 3, 40, 2)

    def test_unit_length(self):
        """All normal vectors must have unit length."""
        n = normals(make_circle(k=40))
        norms = torch.linalg.vector_norm(n, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_outward_for_circle(self):
        """Normals of a CCW circle should point away from the centre."""
        c   = make_circle(k=40)                          # (1, 1, 40, 2)
        n   = normals(c)                                 # (1, 1, 40, 2)
        # vector from centre to node
        rad = c - 0.5                                    # (1, 1, 40, 2)
        # dot product of normal with radial direction should be positive
        dot = (n * rad).sum(dim=-1)                      # (1, 1, 40)
        assert (dot > 0).all()

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        normals(c).sum().backward()
        assert c.grad is not None


class TestCurvature:
    def test_output_shape(self):
        k = 20
        c = make_circle(k=k, b=2, n=3)
        # padding is internal; output has same K as input
        assert curvature(c).shape == (2, 3, k)

    def test_circle_constant_curvature(self):
        """Curvature of a circle is constant = 1/r."""
        r = 0.3
        c   = make_circle(k=100, r=r)
        kap = curvature(c)          # (1, 1, 94)
        # All values should be close to the theoretical curvature
        expected = 1.0 / r
        assert kap.std().item() / expected < 0.05, "curvature not constant on circle"

    def test_differentiable(self):
        c = make_circle(k=20).requires_grad_(True)
        curvature(c).sum().backward()
        assert c.grad is not None


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

class TestHausdorffDistance:
    def test_output_shape(self):
        c = make_circle(b=2, n=3)
        assert hausdorff_distance(c, c).shape == (2, 3)

    def test_zero_for_identical(self):
        c  = make_circle()
        hd = hausdorff_distance(c, c)
        assert torch.allclose(hd, torch.zeros(1, 1), atol=1e-5)

    def test_positive_for_different(self):
        c1 = make_circle(r=0.2)
        c2 = make_circle(r=0.3)
        assert hausdorff_distance(c1, c2).item() > 0

    def test_shape_mismatch_raises(self):
        c1 = make_circle(k=20)
        c2 = make_circle(k=30)
        with pytest.raises(ValueError):
            hausdorff_distance(c1, c2)

    def test_differentiable(self):
        c1 = make_circle(r=0.2).requires_grad_(True)
        c2 = make_circle(r=0.3)
        hausdorff_distance(c1, c2).sum().backward()
        assert c1.grad is not None


class TestIoU:
    def test_output_shape(self):
        m = torch.rand(2, 3, SIZE, SIZE)
        assert iou(m, m).shape == (2, 3)

    def test_one_for_identical(self):
        # IoU == 1 only for binary masks (soft masks give iou < 1 in general)
        m = torch.zeros(2, 1, SIZE, SIZE)
        m[:, :, : SIZE // 2, :] = 1.0
        assert torch.allclose(iou(m, m), torch.ones(2, 1), atol=1e-5)

    def test_zero_for_non_overlapping(self):
        m1 = torch.zeros(1, 1, SIZE, SIZE)
        m2 = torch.zeros(1, 1, SIZE, SIZE)
        m1[:, :, : SIZE // 2, :] = 1.0
        m2[:, :, SIZE // 2 :, :] = 1.0
        assert iou(m1, m2).item() < 1e-4

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            iou(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 64, 64))

    def test_differentiable(self):
        m1 = torch.rand(1, 1, SIZE, SIZE, requires_grad=True)
        m2 = torch.rand(1, 1, SIZE, SIZE)
        iou(m1, m2).sum().backward()
        assert m1.grad is not None


class TestDice:
    def test_output_shape(self):
        m = torch.rand(2, 3, SIZE, SIZE)
        assert dice(m, m).shape == (2, 3)

    def test_one_for_identical(self):
        # Dice == 1 only for binary masks (soft masks give dice < 1 in general)
        m = torch.zeros(2, 1, SIZE, SIZE)
        m[:, :, : SIZE // 2, :] = 1.0
        assert torch.allclose(dice(m, m), torch.ones(2, 1), atol=1e-5)

    def test_zero_for_non_overlapping(self):
        m1 = torch.zeros(1, 1, SIZE, SIZE)
        m2 = torch.zeros(1, 1, SIZE, SIZE)
        m1[:, :, : SIZE // 2, :] = 1.0
        m2[:, :, SIZE // 2 :, :] = 1.0
        assert dice(m1, m2).item() < 1e-4

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            dice(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 64, 64))

    def test_differentiable(self):
        m1 = torch.rand(1, 1, SIZE, SIZE, requires_grad=True)
        m2 = torch.rand(1, 1, SIZE, SIZE)
        dice(m1, m2).sum().backward()
        assert m1.grad is not None


# ---------------------------------------------------------------------------
# Feature sampling
# ---------------------------------------------------------------------------

class TestSampleFeaturesOnContour:
    def test_output_shape(self):
        fm = torch.rand(2, 64, SIZE, SIZE)
        c  = make_circle(k=30, b=2, n=3)
        out = sample_features_on_contour(fm, c)
        assert out.shape == (2, 3, 30, 64)

    def test_gradient_flows_to_contour(self):
        """Loss on sampled features must propagate back to contour coords."""
        fm = torch.rand(1, 8, SIZE, SIZE)
        c  = make_circle(k=20).requires_grad_(True)
        sample_features_on_contour(fm, c).sum().backward()
        assert c.grad is not None
        assert not torch.isnan(c.grad).any()

    def test_gradient_flows_to_feature_map(self):
        fm = torch.rand(1, 8, SIZE, SIZE, requires_grad=True)
        c  = make_circle(k=20)
        sample_features_on_contour(fm, c).sum().backward()
        assert fm.grad is not None

    def test_invalid_contour_raises(self):
        fm = torch.rand(1, 8, SIZE, SIZE)
        with pytest.raises(ValueError):
            sample_features_on_contour(fm, make_circle() * 2)


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------

class TestSpacingLoss:
    def test_output_shape(self):
        assert spacing_loss(make_circle(b=2, n=3)).shape == (2, 3)

    def test_near_zero_for_uniform_circle(self):
        """A uniformly-sampled circle has near-zero spacing variance."""
        loss = spacing_loss(make_circle(k=100)).item()
        assert loss < 1e-8

    def test_higher_for_non_uniform(self):
        torch.manual_seed(0)
        uniform = make_circle(k=40)
        # perturb some nodes to break uniformity
        non_uniform = uniform.clone()
        non_uniform[0, 0, ::5] += 0.05
        non_uniform = non_uniform.clamp(0.01, 0.99)
        assert spacing_loss(non_uniform).item() > spacing_loss(uniform).item()

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        spacing_loss(c).sum().backward()
        assert c.grad is not None


class TestSmoothnessLoss:
    def test_output_shape(self):
        assert smoothness_loss(make_circle(b=2, n=3)).shape == (2, 3)

    def test_lower_for_smooth(self):
        """A smooth circle has lower smoothness loss than a perturbed one."""
        torch.manual_seed(42)
        smooth = make_circle(k=40)
        jagged = (smooth + torch.randn_like(smooth) * 0.03).clamp(0.01, 0.99)
        assert smoothness_loss(smooth).item() < smoothness_loss(jagged).item()

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        smoothness_loss(c).sum().backward()
        assert c.grad is not None


class TestBalloonLoss:
    def test_output_shape(self):
        assert balloon_loss(make_circle(b=2, n=3)).shape == (2, 3)

    def test_equals_area(self):
        """balloon_loss is identical to area."""
        c = make_circle(k=50)
        assert torch.allclose(balloon_loss(c), area(c))

    def test_larger_contour_has_larger_loss(self):
        small = make_circle(r=0.1)
        large = make_circle(r=0.4)
        assert balloon_loss(large).item() > balloon_loss(small).item()

    def test_differentiable(self):
        c = make_circle().requires_grad_(True)
        balloon_loss(c).sum().backward()
        assert c.grad is not None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_out_of_range_raises(self):
        c_bad = make_circle() + 1.0   # values > 1
        for layer in [ContourToMask(SIZE), ContourToDistanceMap(SIZE), ContourToSDF(SIZE)]:
            with pytest.raises(ValueError, match="range"):
                layer(c_bad)

    def test_wrong_ndim_raises(self):
        c_bad = make_circle().squeeze(0)   # 3-D instead of 4-D
        with pytest.raises(ValueError):
            ContourToMask(SIZE)(c_bad)

    def test_wrong_last_dim_raises(self):
        c_bad = make_circle().unsqueeze(-1)   # last dim = 1, not 2
        with pytest.raises(ValueError):
            ContourToMask(SIZE)(c_bad)

    def test_hausdorff_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            hausdorff_distance(make_circle(k=10), make_circle(k=20))

    def test_iou_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            iou(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 64, 64))


# ---------------------------------------------------------------------------
# CleanContours
# ---------------------------------------------------------------------------

class TestCleanContours:
    def test_clean_contours_returns_list(self):
        c = make_circle(k=30).numpy()
        result = CleanContours.clean_contours(c)
        assert isinstance(result, list)
        assert len(result) == 1   # b*n = 1

    def test_clean_and_interpolate_shape(self):
        c = make_circle(k=30).numpy()
        out = CleanContours.clean_contours_and_interpolate(c)
        assert out.shape == (1, 1, 30, 2)

    def test_clean_and_interpolate_batch(self):
        c = make_circle(k=30, b=2, n=3).numpy()
        out = CleanContours.clean_contours_and_interpolate(c)
        assert out.shape == (2, 3, 30, 2)

    def test_contour_without_loops_unchanged_length(self):
        """A clean circle has no loops — length should be preserved."""
        c = make_circle(k=30).numpy()[0, 0]   # (30, 2)
        length_before = CleanContours.contour_length(c)
        cleaned = CleanContours.remove_small_loops(c, length_before)
        assert cleaned.shape[0] == c.shape[0]

    def test_make_strictly_increasing(self):
        seq = np.array([0.0, 0.5, 0.5, 0.5, 1.0])
        result = CleanContours.make_strictly_increasing(seq)
        assert np.all(np.diff(result) > 0)

    def test_interpolate_output_shape(self):
        c = make_circle(k=30).numpy()[0, 0]   # (30, 2)
        out = CleanContours.interpolate(c, n=50)
        assert out.shape == (50, 2)
