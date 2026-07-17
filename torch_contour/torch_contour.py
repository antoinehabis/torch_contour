from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_contour(contour: torch.Tensor) -> None:
    """Raise ValueError if *contour* is not a valid (B, N, K, 2) tensor in [0, 1]."""
    if contour.ndim != 4:
        raise ValueError(
            f"contour must be a 4-D tensor of shape (B, N, K, 2), got {contour.ndim}-D"
        )
    if contour.shape[-1] != 2:
        raise ValueError(
            f"contour last dimension must be 2 (x, y coordinates), got {contour.shape[-1]}"
        )
    if (contour < 0).any() or (contour > 1).any():
        raise ValueError("Tensor values should be in the range [0, 1]")


# ---------------------------------------------------------------------------
# Base class — shared mesh buffer and winding-number core
# ---------------------------------------------------------------------------

class _ContourBase(nn.Module):
    """Shared initialisation (mesh buffer) and core winding-number computation.

    Parameters
    ----------
    size : int
        Output image size (square: *size* × *size*).
    k : float
        Sharpness of the soft-sign approximation (default 1e5).
    eps : float
        Numerical stability term (default 1e-5).
    """

    def __init__(self, size: int, k: float = 1e5, eps: float = 1e-5) -> None:
        super().__init__()
        self.k = k
        self.eps = eps
        self.size = size
        mesh = (
            torch.unsqueeze(
                torch.stack(
                    torch.meshgrid(
                        torch.arange(self.size),
                        torch.arange(self.size),
                        indexing="ij",
                    ),
                    dim=-1,
                ).reshape(-1, 2),
                dim=1,
            )
            / self.size
        )
        self.register_buffer("mesh", mesh)

    def _compute(
        self, contour: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Core winding-number computation shared by all subclasses.

        Parameters
        ----------
        contour : torch.Tensor
            Shape (B, N, K, 2), values in [0, 1].

        Returns
        -------
        mask : torch.Tensor, shape (B*N, size, size)
            Per-pixel winding-number approximation (values in [0, 1]).
        min_diff : torch.Tensor, shape (B*N, size, size)
            Minimum distance from each pixel to any contour vertex.
        b, n : int
        """
        _validate_contour(contour)
        b, n, k, _ = contour.shape
        bn = b * n
        contour = contour.reshape(bn, k, -1)                        # (B*N, K, 2)
        mesh = self.mesh.unsqueeze(0).expand(bn, -1, -1, -1)       # (B*N, P, 1, 2)

        contour = torch.unsqueeze(contour, dim=1)                   # (B*N, 1, K, 2)
        diff = -mesh + contour                                       # (B*N, P, K, 2)

        # Minimum distance from each pixel to the nearest vertex
        vertex_norms = torch.linalg.vector_norm(diff, dim=-1)       # (B*N, P, K)
        min_diff = torch.min(vertex_norms, dim=2)[0]                # (B*N, P)
        min_diff = min_diff.reshape(bn, self.size, self.size)

        # Winding-number soft sign: keeps the same sign convention as the
        # original formulation (diff_y * roll_x − diff_x * roll_y)
        roll_diff = torch.roll(diff, -1, dims=2)                    # (B*N, P, K, 2)
        sign = diff * torch.roll(roll_diff, 1, dims=3)
        sign = sign[:, :, :, 1] - sign[:, :, :, 0]
        sign = torch.tanh(self.k * sign)                            # (B*N, P, K)

        norm_diff = torch.clamp(
            torch.linalg.vector_norm(diff, dim=3), self.eps, None
        )                                                           # (B*N, P, K)
        norm_roll = torch.clamp(
            torch.linalg.vector_norm(roll_diff, dim=3), self.eps, None
        )
        scalar_product = torch.sum(diff * roll_diff, dim=3)        # (B*N, P, K)
        cos_angle = torch.clamp(
            scalar_product / (norm_diff * norm_roll + self.eps),
            -1 + self.eps,
            1 - self.eps,
        )
        angles = torch.acos(cos_angle)                             # (B*N, P, K)
        mask = torch.abs(
            torch.sum(sign * angles, dim=2) / (2 * math.pi)       # sum over K
        )                                                          # (B*N, P)
        mask = mask.reshape(bn, self.size, self.size)

        return mask, min_diff, b, n


# ---------------------------------------------------------------------------
# Public modules
# ---------------------------------------------------------------------------

class ContourToMask(_ContourBase):
    """Differentiable polygon → binary mask layer.

    Parameters
    ----------
    size : int
        Output image size (square).
    k : float
        Sharpness of the sign approximation (default 1e5).
    eps : float
        Numerical stability term (default 1e-5).
    """

    def __init__(self, size: int, k: float = 1e5, eps: float = 1e-5) -> None:
        super().__init__(size=size, k=k, eps=eps)

    def forward(self, contour: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        contour : torch.Tensor
            Shape (B, N, K, 2), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, N, size, size) — binary mask for each polygon.
        """
        mask, _, b, n = self._compute(contour)
        return torch.clamp(mask, 0, 1).reshape(b, n, self.size, self.size)


class ContourToDistanceMap(_ContourBase):
    """Differentiable polygon → distance map layer.

    Parameters
    ----------
    size : int
        Output image size (square).
    k : float
        Sharpness of the sign approximation (default 1e5).
    eps : float
        Numerical stability term (default 1e-5).
    """

    def __init__(self, size: int, k: float = 1e5, eps: float = 1e-5) -> None:
        super().__init__(size=size, k=k, eps=eps)

    def forward(
        self, contour: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        contour : torch.Tensor
            Shape (B, N, K, 2), values in [0, 1].
        return_mask : bool
            If True, also return the intermediate winding-number mask.

        Returns
        -------
        dmap : torch.Tensor
            Shape (B, N, size, size) — distance map for each polygon.
        mask : torch.Tensor
            Shape (B, N, size, size) — only returned when *return_mask* is True.
        """
        mask, min_diff, b, n = self._compute(contour)
        raw = mask * min_diff
        dmap = (raw / torch.max(raw)).reshape(b, n, self.size, self.size)
        if return_mask:
            return dmap, mask.reshape(b, n, self.size, self.size)
        return dmap


class ContourToIsolines(_ContourBase):
    """Differentiable polygon → Gaussian-weighted isoline layer.

    Parameters
    ----------
    size : int
        Output image size (square).
    isolines : list of float
        Isoline levels, each in [0, 1].
    k : float
        Sharpness of the sign approximation (default 1e5).
    eps : float
        Numerical stability term (default 1e-5).
    """

    def __init__(
        self,
        size: int,
        isolines: list[float],
        k: float = 1e5,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(size=size, k=k, eps=eps)

        if any(element < 0 or element > 1 for element in isolines):
            raise ValueError("all isolines must be in the range [0, 1].")

        # Register in order: mesh (done by super), isolines, then vars (depends on isolines)
        self.register_buffer("isolines", torch.tensor(isolines, dtype=torch.float32))
        self.register_buffer("vars", self.mean_to_var(self.isolines))

    def mean_to_var(self, isolines: torch.Tensor) -> torch.Tensor:
        """Compute per-isoline Gaussian variance from pairwise distances.

        Parameters
        ----------
        isolines : torch.Tensor, shape (I,)

        Returns
        -------
        torch.Tensor, shape (I,)
            Variance for each isoline: ``-min_sq_dist / (8 * log(0.5))``.
        """
        col = isolines[:, None]
        mat = torch.cdist(col, col) ** 2
        mat = mat.clone()
        mat[mat == 0.0] = float("inf")
        min_sq_dist = torch.min(mat, dim=0).values
        return -min_sq_dist / (8 * math.log(0.5))

    def forward(self, contour: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        contour : torch.Tensor
            Shape (B, N, K, 2), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, N, I, size, size) where I = number of isolines.
        """
        mask, min_diff, b, n = self._compute(contour)
        raw = mask * min_diff
        dmap = (raw / torch.max(raw)).reshape(b, n, self.size, self.size)
        mask = torch.clamp(mask, 0.0, 1.0).reshape(b, n, self.size, self.size)
        return mask[:, :, None, ...] * torch.exp(
            -((self.isolines[None, None, :, None, None] - dmap[:, :, None, ...]) ** 2)
            / self.vars[None, None, :, None, None]
        )


class ContourToSDF(_ContourBase):
    """Differentiable polygon → signed distance function (SDF) layer.

    Positive inside the contour, negative outside, zero on the boundary.
    Magnitude approximates the distance to the nearest contour vertex in
    normalized [0, 1] coordinate space.

    Parameters
    ----------
    size : int
        Output image size (square).
    k : float
        Sharpness of the sign approximation (default 1e5).
    eps : float
        Numerical stability term (default 1e-5).
    """

    def __init__(self, size: int, k: float = 1e5, eps: float = 1e-5) -> None:
        super().__init__(size=size, k=k, eps=eps)

    def forward(self, contour: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        contour : torch.Tensor
            Shape (B, N, K, 2), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, N, size, size) — SDF: positive inside, negative outside.
        """
        mask, min_diff, b, n = self._compute(contour)
        # (2·mask − 1): +1 deep inside, −1 deep outside, 0 on the boundary
        sdf = (2.0 * mask - 1.0) * min_diff
        return sdf.reshape(b, n, self.size, self.size)


class Sobel(nn.Module):
    """Sobel edge-detection filter (non-trainable)."""

    def __init__(self) -> None:
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0).unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.filter(img)
        x = torch.sqrt(torch.sum(torch.mul(x, x), dim=1, keepdim=True))
        # Zero the border without in-place ops (in-place would break autograd)
        return F.pad(x[:, :, 1:-1, 1:-1], (1, 1, 1, 1), value=0.0)


class DrawContour(nn.Module):
    """Differentiable polygon contour drawing layer.

    Uses a mask + Sobel edge detector to render the contour boundary.

    Parameters
    ----------
    size : int
        Output image size (square).
    thickness : int
        Line thickness in pixels (default 1).
    k : float
        Sharpness of the sign approximation forwarded to ContourToMask (default 1e5).
    """

    def __init__(self, size: int, thickness: int = 1, k: float = 1e5) -> None:
        super().__init__()
        self.k = k
        self.size = size
        self.thickness = thickness
        self.max_ = nn.MaxPool2d(
            kernel_size=(self.thickness, self.thickness),
            stride=1,
            padding=(self.thickness // 2, self.thickness // 2),
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )
        self.sobel = Sobel()
        self.ctm = ContourToMask(size=self.size, k=self.k)

    def forward(self, contour: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        contour : torch.Tensor
            Shape (B, N, K, 2), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, N, size, size) — drawn contour for each polygon.
        """
        b, n = contour.shape[0], contour.shape[1]

        mask = self.ctm(contour)                                     # (B, N, size, size)

        # Sobel is Conv2d(in_channels=1): process each polygon separately
        mask_flat = mask.reshape(b * n, 1, self.size, self.size)     # (B*N, 1, H, W)
        drawn = self.sobel(mask_flat)                                # (B*N, 1, H, W)
        drawn = self.max_(drawn)                                     # (B*N, 1, H, W)
        drawn = drawn.reshape(b, n, self.size * self.size)           # (B, N, H*W)

        # Normalize per polygon — keepdim via unsqueeze for correct broadcasting
        drawn_min = drawn.min(dim=-1)[0].unsqueeze(-1)               # (B, N, 1)
        drawn_max = drawn.max(dim=-1)[0].unsqueeze(-1)               # (B, N, 1)
        drawn = (drawn - drawn_min) / (drawn_max - drawn_min + 1e-9)

        return drawn.reshape(b, n, self.size, self.size)


class Smoothing(nn.Module):
    """Circular Gaussian smoothing for closed contours.

    The smoothing is applied along the node dimension (K) and correctly
    handles the periodic boundary of a closed contour — no need to satisfy
    ``contour[0] == contour[-1]``.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel.
    """

    def __init__(self, sigma: float) -> None:
        super(Smoothing, self).__init__()
        self.sigma = sigma
        self.register_buffer("kernel", self._build_kernel())

    def _build_kernel(self) -> torch.Tensor:
        """Build a normalised Gaussian kernel of half-width 5σ."""
        half = int(self.sigma * 5)
        kernel_range = np.arange(-half, half + 1, dtype=float)
        x = np.exp(-(kernel_range**2) / (2 * self.sigma**2))
        tmp = torch.tensor(x / np.sum(x), dtype=torch.float32)[None, None]
        return torch.cat([tmp, tmp])

    def forward(self, contours: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        contours : torch.Tensor
            Shape (B, N, K, 2).

        Returns
        -------
        torch.Tensor
            Smoothed contours, shape (B, N, K, 2).
        """
        b, n, k, _ = contours.shape
        contours = contours.reshape(b * n, k, -1)
        margin = k // 2

        # Circular padding: append end of contour before start, and vice-versa
        out = torch.cat([contours[:, -margin:], contours, contours[:, :margin]], dim=1)
        out = torch.moveaxis(out, -1, 1)                             # (B*N, 2, k+2*margin)
        smoothed = F.conv1d(out, self.kernel, padding="same", groups=2)
        smoothed = torch.moveaxis(smoothed, -1, 1)                  # (B*N, k+2*margin, 2)
        return smoothed[:, margin:-margin, :].reshape(b, n, k, 2)


# ---------------------------------------------------------------------------
# Differentiable feature sampling
# ---------------------------------------------------------------------------

def sample_features_on_contour(
    feature_map: torch.Tensor,
    contour: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Sample image features at contour node positions (differentiable).

    Uses bilinear interpolation so that gradients flow back to both the
    feature map and the contour coordinates.

    Parameters
    ----------
    feature_map : torch.Tensor
        Shape (B, C, H, W) — feature map to sample from (e.g. CNN output).
    contour : torch.Tensor
        Shape (B, N, K, 2) — contour node coordinates in [0, 1].
    mode : str
        Interpolation mode passed to ``grid_sample`` (default ``"bilinear"``).
    padding_mode : str
        Behaviour for coordinates outside [0, 1]: ``"border"`` (default),
        ``"zeros"``, or ``"reflection"``.

    Returns
    -------
    torch.Tensor
        Shape (B, N, K, C) — feature vector at each contour node.

    Example
    -------
    >>> fm = torch.rand(2, 64, 128, 128)   # batch of 2, 64-channel feature maps
    >>> c  = torch.rand(2, 3, 100, 2)      # 3 polygons of 100 nodes each
    >>> sample_features_on_contour(fm, c).shape
    torch.Size([2, 3, 100, 64])
    """
    _validate_contour(contour)
    b, c_dim, h, w = feature_map.shape
    _, n, k, _ = contour.shape

    # grid_sample expects coordinates in [-1, 1]; contour is in [0, 1]
    grid = contour.reshape(b, n * k, 1, 2) * 2 - 1          # (B, N*K, 1, 2)

    # grid_sample: (B, C, H, W) × (B, N*K, 1, 2) → (B, C, N*K, 1)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )                                                         # (B, C, N*K, 1)

    # Reshape to (B, N, K, C)
    return sampled[:, :, :, 0].reshape(b, c_dim, n, k).permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# Standalone geometric functions
# ---------------------------------------------------------------------------

def area(contours: torch.Tensor) -> torch.Tensor:
    """Shoelace area of each polygon.

    Parameters
    ----------
    contours : torch.Tensor
        Shape (B, N, K, 2).

    Returns
    -------
    torch.Tensor
        Shape (B, N) — area of each polygon.
    """
    b, n, k, _ = contours.shape
    c = contours.reshape(b * n, k, 2)
    x, y = c[..., 0], c[..., 1]
    areas = torch.abs(
        torch.sum(x * torch.roll(y, 1, dims=1) - torch.roll(x, 1, dims=1) * y, dim=1)
    ) / 2.0
    return areas.reshape(b, n)


def perimeter(contours: torch.Tensor) -> torch.Tensor:
    """Perimeter of each polygon.

    Parameters
    ----------
    contours : torch.Tensor
        Shape (B, N, K, 2).

    Returns
    -------
    torch.Tensor
        Shape (B, N) — perimeter of each polygon.
    """
    b, n, k, _ = contours.shape
    c = contours.reshape(b * n, k, 2)
    distances = torch.sqrt(torch.sum((c - torch.roll(c, shifts=-1, dims=1)) ** 2, dim=2))
    return distances.sum(dim=-1).reshape(b, n)


def hausdorff_distance(contours1: torch.Tensor, contours2: torch.Tensor) -> torch.Tensor:
    """Symmetric Hausdorff distance between corresponding polygon pairs.

    Parameters
    ----------
    contours1, contours2 : torch.Tensor
        Both shape (B, N, K, 2).

    Returns
    -------
    torch.Tensor
        Shape (B, N) — Hausdorff distance for each polygon pair.
    """
    if contours1.shape != contours2.shape:
        raise ValueError(
            f"contours1 and contours2 must have the same shape, "
            f"got {tuple(contours1.shape)} vs {tuple(contours2.shape)}"
        )
    b, n, k, _ = contours1.shape
    c1 = contours1.reshape(b * n, k, 2)
    c2 = contours2.reshape(b * n, k, 2)

    # donot_use_mm avoids numerical errors on the diagonal when c1 == c2
    dists = torch.cdist(c1, c2, compute_mode="donot_use_mm_for_euclid_dist")  # (B*N, K, K)
    min_1_to_2 = torch.min(dists, dim=2)[0]                        # (B*N, K)
    min_2_to_1 = torch.min(dists, dim=1)[0]                        # (B*N, K)
    h_1_to_2 = torch.max(min_1_to_2, dim=1).values                 # (B*N,)
    h_2_to_1 = torch.max(min_2_to_1, dim=1).values                 # (B*N,)
    return torch.max(h_1_to_2, h_2_to_1).reshape(b, n)


def curvature(contour: torch.Tensor) -> torch.Tensor:
    """Curvature at each node of a closed contour.

    The contour is padded by 3 points on each side to handle boundary
    conditions; the 6 padding points are removed from the output.

    Parameters
    ----------
    contour : torch.Tensor
        Shape (B, N, K, 2).

    Returns
    -------
    torch.Tensor
        Shape (B, N, K).

    Example
    -------
    >>> contour = torch.rand(1, 1, 10, 2)
    >>> curvature(contour).shape
    torch.Size([1, 1, 10])
    """
    contour = torch.cat([contour[:, :, -3:, :], contour, contour[:, :, :3, :]], dim=-2)
    b, n, k, _ = contour.shape
    contour = contour.reshape(b * n, k, -1)
    velocity = torch.gradient(contour, dim=1)[0]
    accel = torch.gradient(velocity, dim=1)[0]
    curv = (
        torch.abs(accel[:, :, 0] * velocity[:, :, 1] - velocity[:, :, 0] * accel[:, :, 1])
        / torch.sum(velocity**2, dim=-1) ** 1.5
    )
    return curv.reshape(b, n, k)[:, :, 3:-3]


def normals(contour: torch.Tensor) -> torch.Tensor:
    """Unit outward normal vectors at each contour node.

    Computed via central finite differences and a 90° rotation of the tangent.
    Convention: outward normals for clockwise oriented contours (standard image
    space with y-axis pointing downward).

    Parameters
    ----------
    contour : torch.Tensor
        Shape (B, N, K, 2).

    Returns
    -------
    torch.Tensor
        Shape (B, N, K, 2) — unit normal vector at each node.
    """
    # Central-difference tangent (handles closed contour via roll)
    tangent = torch.roll(contour, -1, dims=2) - torch.roll(contour, 1, dims=2)
    # Rotate −90°: (tx, ty) → (ty, −tx) — outward for CW contours (image convention, y down)
    normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
    return normal / (torch.linalg.vector_norm(normal, dim=-1, keepdim=True) + 1e-8)


def compactness(contours: torch.Tensor) -> torch.Tensor:
    """Isoperimetric compactness of each polygon.

    Defined as ``4π · area / perimeter²``. Equals 1 for a perfect circle
    and decreases towards 0 for more elongated or irregular shapes.

    Parameters
    ----------
    contours : torch.Tensor
        Shape (B, N, K, 2).

    Returns
    -------
    torch.Tensor
        Shape (B, N) — compactness in (0, 1].
    """
    a = area(contours)
    p = perimeter(contours)
    return (4.0 * math.pi * a) / (p ** 2 + 1e-8)


def iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable Intersection over Union between two soft masks.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted mask, shape (B, N, H, W), values in [0, 1].
    target : torch.Tensor
        Target mask, shape (B, N, H, W), values in [0, 1].
    eps : float
        Numerical stability term.

    Returns
    -------
    torch.Tensor
        Shape (B, N).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, "
            f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    intersection = torch.sum(pred * target, dim=(-2, -1))
    union = torch.sum(pred + target - pred * target, dim=(-2, -1))
    return intersection / (union + eps)


def dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable Dice coefficient between two soft masks.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted mask, shape (B, N, H, W), values in [0, 1].
    target : torch.Tensor
        Target mask, shape (B, N, H, W), values in [0, 1].
    eps : float
        Numerical stability term.

    Returns
    -------
    torch.Tensor
        Shape (B, N).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, "
            f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    intersection = torch.sum(pred * target, dim=(-2, -1))
    return 2.0 * intersection / (torch.sum(pred + target, dim=(-2, -1)) + eps)


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------

def spacing_loss(contour: torch.Tensor) -> torch.Tensor:
    """Uniform-spacing regularization loss.

    Penalizes variance in inter-node arc lengths. Minimize to push nodes
    towards equal spacing along the contour.

    Parameters
    ----------
    contour : torch.Tensor
        Shape (B, N, K, 2).

    Returns
    -------
    torch.Tensor
        Shape (B, N).
    """
    diffs = torch.roll(contour, -1, dims=2) - contour          # (B, N, K, 2)
    distances = torch.linalg.vector_norm(diffs, dim=-1)         # (B, N, K)
    return torch.var(distances, dim=-1)                         # (B, N)


def smoothness_loss(contour: torch.Tensor) -> torch.Tensor:
    """Smoothness regularization loss (second-order finite differences).

    Penalizes high curvature. Minimize to obtain a smoother contour.
    Unlike :class:`Smoothing`, this is a *differentiable loss term* and
    does not modify the contour coordinates directly.

    Parameters
    ----------
    contour : torch.Tensor
        Shape (B, N, K, 2).

    Returns
    -------
    torch.Tensor
        Shape (B, N).
    """
    second_diff = (
        torch.roll(contour, -1, dims=2)
        - 2.0 * contour
        + torch.roll(contour, 1, dims=2)
    )                                                           # (B, N, K, 2)
    return torch.mean(torch.sum(second_diff ** 2, dim=-1), dim=-1)


def balloon_loss(contour: torch.Tensor) -> torch.Tensor:
    """Balloon force loss (signed area).

    Minimize to shrink the contour; negate and minimize to expand it.

    Parameters
    ----------
    contour : torch.Tensor
        Shape (B, N, K, 2).

    Returns
    -------
    torch.Tensor
        Shape (B, N).
    """
    return area(contour)


# ---------------------------------------------------------------------------
# Numba JIT helpers (CPU, used by CleanContours)
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def cross_product_numba(a, b):
    return a[0] * b[1] - a[1] * b[0]


@jit(nopython=True, cache=True)
def is_intersecting_numba(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    dp = p3 - p1
    cp1 = cross_product_numba(d1, dp)
    cp2 = cross_product_numba(d1, p4 - p1)
    # Early exit: skip the second pair of cross products if the first test fails
    if np.sign(cp1) == np.sign(cp2):
        return False
    cp3 = cross_product_numba(d2, -dp)
    cp4 = cross_product_numba(d2, p2 - p3)
    return np.sign(cp3) != np.sign(cp4)


@jit(nopython=True, cache=True)
def contour_length_numba(contour):
    total_length = 0.0
    num_points = contour.shape[0]
    for i in range(num_points):
        diff = contour[i] - contour[(i + 1) % num_points]
        total_length += np.sqrt(np.sum(diff**2))
    return total_length


@jit(nopython=True, cache=True)
def erase_first_encounter_loop_numba(contour, threshold_length):
    num_points = contour.shape[0]
    for start_idx in range(num_points - 2):
        p1 = contour[start_idx]
        p2 = contour[start_idx + 1]
        # Precompute AABB of segment (p1, p2) — reused for every inner iteration
        min_x1 = min(p1[0], p2[0])
        max_x1 = max(p1[0], p2[0])
        min_y1 = min(p1[1], p2[1])
        max_y1 = max(p1[1], p2[1])
        for end_idx in range(start_idx + 2, num_points - 1):
            p3 = contour[end_idx]
            p4 = contour[end_idx + 1]
            # AABB rejection: skip the cross-product test for clearly disjoint segments
            if (max_x1 < min(p3[0], p4[0]) or min_x1 > max(p3[0], p4[0]) or
                    max_y1 < min(p3[1], p4[1]) or min_y1 > max(p3[1], p4[1])):
                continue
            if is_intersecting_numba(p1, p2, p3, p4):
                loop = contour[start_idx : end_idx + 1]
                loop_len = contour_length_numba(loop)
                if loop_len < threshold_length / 3 or len(loop) < 2:
                    contour = np.concatenate((contour[:start_idx], contour[end_idx:]))
                    return contour
                elif loop_len >= threshold_length / 2:
                    return contour[start_idx : end_idx + 1]
    return contour


@jit(nopython=True, cache=True)
def erase_first_loop_sweep_numba(contour, threshold_length):
    """Sweep-line segment-intersection scan — O(K log K + K·ε) vs O(K²).

    Segments are sorted by min_x; the inner loop breaks as soon as
    ``min_x[j] > max_x[i]``, so only spatially overlapping pairs are tested.
    A second AABB rejection on the y-axis further prunes the candidate set,
    reducing the number of pair checks by ~50–100× compared to the O(K²) scan.

    The loop-removal logic is identical to ``erase_first_encounter_loop_numba``;
    only the order in which intersections are discovered differs.
    """
    n = contour.shape[0]
    if n < 4:
        return contour

    seg_min_x = np.empty(n - 1, dtype=np.float64)
    seg_max_x = np.empty(n - 1, dtype=np.float64)
    seg_min_y = np.empty(n - 1, dtype=np.float64)
    seg_max_y = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        seg_min_x[i] = min(contour[i, 0], contour[i + 1, 0])
        seg_max_x[i] = max(contour[i, 0], contour[i + 1, 0])
        seg_min_y[i] = min(contour[i, 1], contour[i + 1, 1])
        seg_max_y[i] = max(contour[i, 1], contour[i + 1, 1])

    order = np.argsort(seg_min_x)           # sort segment indices by min_x

    for ki in range(len(order) - 1):
        i = order[ki]
        p1 = contour[i]; p2 = contour[i + 1]

        for kj in range(ki + 1, len(order)):
            j = order[kj]
            if seg_min_x[j] > seg_max_x[i]:  # x-ranges disjoint → stop inner loop
                break
            if abs(i - j) <= 1:               # adjacent segments share an endpoint
                continue
            if seg_min_y[j] > seg_max_y[i] or seg_max_y[j] < seg_min_y[i]:
                continue                       # y-ranges disjoint → skip
            p3 = contour[j]; p4 = contour[j + 1]
            if is_intersecting_numba(p1, p2, p3, p4):
                start_idx = min(i, j); end_idx = max(i, j)
                loop = contour[start_idx : end_idx + 1]
                loop_len = contour_length_numba(loop)
                if loop_len < threshold_length / 3 or len(loop) < 2:
                    return np.concatenate((contour[:start_idx], contour[end_idx:]))
                elif loop_len >= threshold_length / 2:
                    return contour[start_idx : end_idx + 1]
    return contour


@jit(nopython=True, cache=True)
def make_strictly_increasing_numba(sequence, epsilon=1e-3):
    modified = sequence.copy()
    for i in range(1, len(modified)):
        if modified[i] <= modified[i - 1]:
            modified[i] = modified[i - 1] + epsilon
    return modified


# ---------------------------------------------------------------------------
# CleanContours — polygon loop removal and interpolation utilities
# ---------------------------------------------------------------------------

class CleanContours:
    """Utilities for cleaning and re-interpolating closed contours.

    All methods are static — the class is used as a namespace.
    """

    @staticmethod
    def contour_length(contour: np.ndarray) -> float:
        """Total arc length of a polygon.

        Parameters
        ----------
        contour : np.ndarray, shape (N, 2)
        """
        return contour_length_numba(contour)

    @staticmethod
    def is_intersecting(
        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
    ) -> bool:
        """Return True if segment (p1, p2) intersects segment (p3, p4)."""
        return is_intersecting_numba(p1, p2, p3, p4)

    @staticmethod
    def erase_first_encounter_loop(
        contour: np.ndarray, threshold_length: float
    ) -> np.ndarray:
        """Remove or extract the first loop found in *contour*.

        Parameters
        ----------
        contour : np.ndarray, shape (N, 2)
        threshold_length : float
            Loops shorter than ``threshold_length / 3`` are removed;
            loops longer than ``threshold_length / 2`` are returned as-is.
        """
        return erase_first_encounter_loop_numba(contour, threshold_length)

    @staticmethod
    def make_strictly_increasing(x: np.ndarray) -> np.ndarray:
        """Nudge a sequence so that every element is strictly greater than the previous."""
        return make_strictly_increasing_numba(x)

    @staticmethod
    def remove_small_loops(contour: np.ndarray, length: float) -> np.ndarray:
        """Iteratively remove all small loops from *contour*.

        Uses the sweep-line algorithm (O(K log K + K·ε)) which is 10–160×
        faster than the O(K²) AABB scan for typical contour sizes.

        Parameters
        ----------
        contour : np.ndarray, shape (N, 2)
        length : float
            Reference length (typically the full contour perimeter).
        """
        cleaned = erase_first_loop_sweep_numba(contour, length)
        while contour.shape[0] != cleaned.shape[0]:
            contour = cleaned
            cleaned = erase_first_loop_sweep_numba(contour, contour_length_numba(contour))
        return cleaned

    @staticmethod
    def interpolate(contour: np.ndarray, n: int) -> np.ndarray:
        """Cubic-spline re-interpolation of *contour* to *n* evenly-spaced points.

        Parameters
        ----------
        contour : np.ndarray, shape (M, 2)
        n : int
            Number of output points.
        """
        margin = contour.shape[0] // 2
        contour_concat = np.concatenate([contour[-margin:-1], contour, contour[:margin]])
        distance = np.cumsum(
            np.sqrt(np.sum(np.diff(contour_concat, axis=0) ** 2, axis=1))
        )
        distance = np.insert(distance, 0, 0)
        distance = CleanContours.make_strictly_increasing(distance)
        cub = CubicSpline(distance, contour_concat)
        return cub(np.linspace(distance[margin], distance[-margin], n))

    @staticmethod
    def clean_contours(contours: np.ndarray) -> list[np.ndarray]:
        """Remove small loops from every polygon in a batch.

        Parameters
        ----------
        contours : np.ndarray, shape (B, N, K, 2)

        Returns
        -------
        list of np.ndarray
            Cleaned contours (variable length per polygon).
        """
        b, n, k, _ = contours.shape
        contours = contours.reshape(b * n, k, 2)
        results = []
        for contour in contours:
            length = contour_length_numba(contour)
            results.append(CleanContours.remove_small_loops(contour, length))
        return results

    @staticmethod
    def clean_contours_and_interpolate(contours: np.ndarray) -> np.ndarray:
        """Remove small loops and re-interpolate to the original node count.

        Parameters
        ----------
        contours : np.ndarray, shape (B, N, K, 2)

        Returns
        -------
        np.ndarray, shape (B, N, K, 2)
        """
        b, n, k, _ = contours.shape
        contours = contours.reshape(b * n, k, 2)
        results = []
        for contour in contours:
            length = contour_length_numba(contour)
            contour = CleanContours.remove_small_loops(contour, length)
            results.append(CleanContours.interpolate(contour, k))
        return np.array(results).reshape(b, n, k, 2)


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (old underscore names)
# ---------------------------------------------------------------------------

Contour_to_mask = ContourToMask
Contour_to_distance_map = ContourToDistanceMap
Contour_to_isolines = ContourToIsolines
Draw_contour = DrawContour
