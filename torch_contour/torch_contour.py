import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
from torch import cdist


class Contour_to_mask(nn.Module):
    """This layer transform a polygon into a mask

    ...

    Attributes
    ----------
    size: int
        the size of the output image
    k: float
        the control parameter to approximate the sign function
    eps: float
        a parameter to smooth the function and avoid division by 0

    Methods
    -------
    forward(contour)
        forward function that turns the contour into a mask
    """

    def __init__(self, size, k=1e5, eps=1e-5):
        """
        Parameters
        ----------
        size: int
            the size of the output image
        k: float
            the control parameter to approximate the sign function
        eps: float
            a parameter to smooth the function and avoid division by 0

        """
        super().__init__()
        self.k = k
        self.eps = eps
        self.size = size
        self.mesh = (
            torch.unsqueeze(
                torch.stack(
                    torch.meshgrid(torch.arange(self.size), torch.arange(self.size)),
                    dim=-1,
                ).reshape(-1, 2),
                dim=1,
            )
            / self.size
        )

    def forward(self, contour):
        """Return the distance map of the given size given the contour.

        Raises
        ------
        ValueError
            If the values of the contour or not between 0 and 1.
        """

        b, n, k, _ = contour.shape
        device = contour.device
        contour = contour.reshape(b * n, k, -1)
        mesh = self.mesh.unsqueeze(0).repeat(b * n, 1, 1, 1)
        mesh = mesh.to(device)

        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        if (contour < 0).any() or (contour > 1).any():
            raise ValueError("Tensor values should be in the range [0, 1]")

        contour = torch.unsqueeze(contour, dim=1)
        diff = -mesh + contour
        roll_diff = torch.roll(diff, -1, dims=2)
        sign = diff * torch.roll(roll_diff, 1, dims=3)
        sign = sign[:, :, :, 1] - sign[:, :, :, 0]
        sign = torch.tanh(self.k * sign)
        norm_diff = torch.linalg.vector_norm(diff, dim=3)
        norm_roll = torch.linalg.vector_norm(roll_diff, dim=3)
        scalar_product = torch.sum(diff * roll_diff, dim=3)

        clip = torch.clamp(scalar_product / (norm_diff * norm_roll), -1 + self.eps, 1 - self.eps)
        angles = torch.acos(clip)
        sum_angles = torch.clamp(torch.abs(torch.sum(sign * angles, dim=2) / (2 * torch.pi)), 0, 1)
        out0 = sum_angles.reshape(b * n, self.size, self.size)
        mask = out0.reshape(b, n, self.size, self.size)

        return mask


class Contour_to_distance_map(nn.Module):
    """This layer transform a polygon into a distance map

    ...

    Attributes
    ----------
    size: int
        the size of the output image
    k: float
        the control parameter to approximate the sign function
    eps: float
        a parameter to smooth the function and avoid division by 0

    Methods
    -------
    forward(contour)
        forward function that turns the contour into a distance map
    """

    def __init__(self, size, k=1e5, eps=1e-5):
        """
        Parameters
        ----------
        size: int
            the size of the output image
        k: float
            the control parameter to approximate the sign function
        eps: float
            a parameter to smooth the function and avoid division by 0

        """
        super().__init__()
        self.k = k
        self.eps = eps
        self.size = size
        self.mesh = (
            torch.unsqueeze(
                torch.stack(
                    torch.meshgrid(torch.arange(self.size), torch.arange(self.size)),
                    dim=-1,
                ).reshape(-1, 2),
                dim=1,
            )
            / self.size
        )

    def forward(self, contour):
        """Return the distance map on an image of the given size

        ...

        Parameters
        ----------

        contour (torch.Tensor): A 4D tensor of shape (B, N, K, 2) where B is the batch size,
                        N is the number of polygons to draw for each image,
                        K is the number of nodes per polygon,
                        2 represents the coordinates (x, y) of each point,

        Returns:
        torch.Tensor: A 4D tensor of shape (B, N, self.size, self.size) containing a distance map for each polygon of each image of each batch.


        Raises
        ------
        ValueError
            If the values of the contour or not between 0 and 1.
        """

        b, n, k, _ = contour.shape
        device = contour.device
        contour = contour.reshape(b * n, k, -1)
        mesh = self.mesh.unsqueeze(0).repeat(b * n, 1, 1, 1)
        mesh = mesh.to(device)

        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        if (contour < 0).any() or (contour > 1).any():
            raise ValueError("Tensor values should be in the range [0, 1]")

        contour = torch.unsqueeze(contour, dim=1)
        diff = -mesh + contour
        min_diff = torch.min(torch.norm(diff, dim=-1), dim=2)[0]
        min_diff = min_diff.reshape((b * n, self.size, self.size))
        roll_diff = torch.roll(diff, -1, dims=2)
        sign = diff * torch.roll(roll_diff, 1, dims=3)
        sign = sign[:, :, :, 1] - sign[:, :, :, 0]
        sign = torch.tanh(self.k * sign)
        norm_diff = torch.clip(torch.norm(diff, dim=3), self.eps, None)
        norm_roll = torch.clip(torch.norm(roll_diff, dim=3), self.eps, None)
        scalar_product = torch.sum(diff * roll_diff, dim=3)
        clip = torch.clip(scalar_product / (norm_diff * norm_roll), -1 + self.eps, 1 - self.eps)
        angles = torch.arccos(clip)
        sum_angles = torch.abs(torch.sum(sign * angles, dim=2) / (2 * torch.pi))
        resize = sum_angles.reshape(b * n, self.size, self.size)
        dmap = torch.unsqueeze((resize * min_diff) / torch.max(resize * min_diff), 0)
        dmap = dmap.reshape(b, n, self.size, self.size)
        return dmap


class Contour_to_isolines(nn.Module):
    """This layer transform a polygon into a distance map

    ...

    Attributes
    ----------
    size: int
        the size of the output image
    isolines: List(float)
        contains the values of the isolines to extract for each image
    k: float
        the control parameter to approximate the sign function
    eps: float
        a parameter to smooth the function and avoid division by 0

    Methods
    -------
    forward(contour)
        forward function that turns the contour into a series of isolines centered on the given values
    """

    def __init__(self, size, isolines, k=1e5, eps=1e-5):
        """
        Parameters
        ----------
        size: int
            the size of the output image
        k: float
            the control parameter to approximate the sign function
        eps: float
            a parameter to smooth the function and avoid division by 0
        isolines: List(float)
            contains the values of the isolines to extract for each image

        """
        super().__init__()

        if any(element < 0 or element > 1 for element in isolines):
            raise ValueError("all isolines must be in the range [0, 1].")

        self.k = k
        self.eps = eps
        self.size = size
        self.mesh = (
            torch.unsqueeze(
                torch.stack(
                    torch.meshgrid(torch.arange(self.size), torch.arange(self.size)),
                    dim=-1,
                ).reshape(-1, 2),
                dim=1,
            )
            / self.size
        )
        self.isolines = torch.tensor(isolines)
        self.vars = self.mean_to_var(self.isolines)

    def mean_to_var(self, isolines):
        """
        Computes the variance of the distance between isolines.

        Parameters:
        self : object
            The instance of the class where this method is defined.
        isolines : numpy.ndarray
            A 1D array of isoline values.

        Returns:
        numpy.ndarray
            A 1D array containing the variances corresponding to each isoline value.

        The method works as follows:
        1. It reshapes the input isolines array to a column vector.
        2. It computes the squared pairwise distances between all isolines using the cdist function.
        3. It masks out the zero distances (which are the distances from each point to itself).
        4. It finds the minimum non-zero distance for each isoline.
        5. It computes the variance using the formula -min_distance / (8 * np.log(0.5)).
        """
        isolines = isolines[:, None]
        mat = cdist(isolines, isolines) ** 2
        vars = -np.min(np.ma.masked_equal(mat, 0.0, copy=False), 0) / (8 * np.log(0.5))
        return vars

    def forward(self, contour):
        """Return the isolines centered on self.isolines on an image of the given size

        ...

        Parameters
        ----------

        contour (torch.Tensor): A 4D tensor of shape (B, N, K, 2) where B is the batch size,
                        N is the number of polygons to draw for each image,
                        K is the number of nodes per polygon,
                        2 represents the coordinates (x, y) of each point,

        Returns:
        torch.Tensor: A 5D tensor of shape (B, N, I, self.size, self.size) containing a distance map for each polygon of each image of each batch.
                        where I is the number of isolines to extract for each image (I = isolines.shape[0])


        Raises
        ------
        ValueError
            If the values of the contour or not between 0 and 1.
        """

        b, n, k, _ = contour.shape
        device = contour.device
        contour = contour.reshape(b * n, k, -1)
        mesh = self.mesh.unsqueeze(0).repeat(b * n, 1, 1, 1)
        mesh = mesh.to(device)
        self.vars = torch.tensor(self.vars, device=device, dtype=torch.float32).to(device)
        self.isolines = self.isolines.to(device)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        if (contour < 0).any() or (contour > 1).any():
            raise ValueError("Tensor values should be in the range [0, 1]")

        contour = torch.unsqueeze(contour, dim=1)
        diff = -mesh + contour
        min_diff = torch.min(torch.norm(diff, dim=-1), dim=2)[0]
        min_diff = min_diff.reshape((b * n, self.size, self.size))
        roll_diff = torch.roll(diff, -1, dims=2)
        sign = diff * torch.roll(roll_diff, 1, dims=3)
        sign = sign[:, :, :, 1] - sign[:, :, :, 0]
        sign = torch.tanh(self.k * sign)
        norm_diff = torch.clip(torch.norm(diff, dim=3), self.eps, None)
        norm_roll = torch.clip(torch.norm(roll_diff, dim=3), self.eps, None)
        scalar_product = torch.sum(diff * roll_diff, dim=3)
        clip = torch.clip(scalar_product / (norm_diff * norm_roll), -1 + self.eps, 1 - self.eps)
        angles = torch.arccos(clip)
        mask = torch.abs(torch.sum(sign * angles, dim=2) / (2 * torch.pi))
        mask = mask.reshape(b * n, self.size, self.size)
        dmap = torch.unsqueeze((mask * min_diff) / torch.max(mask * min_diff), 0)
        dmap = dmap.reshape(b, n, self.size, self.size)
        mask = mask.reshape(b, n, self.size, self.size)
        isolines = mask[:, :, None, ...] * torch.exp(
            -((self.isolines[None, None, :, None, None] - dmap[:, :, None, ...]) ** (2))
            / (self.vars[None, None, :, None, None])
        )
        return isolines


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=(1, 1), bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        x[:, :, -1, :] = 0
        x[:, :, :, -1] = 0
        x[:, :, 0, :] = 0
        x[:, :, :, 0] = 0

        return x


class Draw_contour(nn.Module):
    """This layer draws a contour

    ...

    Attributes
    ----------
    size: int
        the size of the output image
    k: float
        the control parameter to approximate the sign function
    thickness: int
        the thickness of the contour to be drawn

    Methods
    -------
    forward(contour)
        forward function that draws the contour
    """

    def __init__(self, size, thickness=1, k=1e5):
        """
        Parameters
        ----------
        size: int
            the size of the output image
        k: float
            the control parameter to approximate the sign function
        thickness: int
            the thickness of the contour to be drawn

        """
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
        self.ctm = Contour_to_mask(size=self.size, k=self.k)

    def forward(self, contour):
        """Return the contour drawn on an image of the given size

        ...

        Parameters
        ----------

        contour (torch.Tensor): A 4D tensor of shape (B, N, K, 2) where B is the batch size,
                        N is the number of polygons to draw for each image,
                        K is the number of nodes per polygon,
                        2 represents the coordinates (x, y) of each point,

        Returns:
        torch.Tensor: A 4D tensor of shape (B, N, self.size, self.size) containing contour drawn for each polygons of each image of each batch.


        Raises
        ------
        ValueError
            If the values of the contour or not between 0 and 1.
        """

        if (contour < 0).any() or (contour > 1).any():
            raise ValueError("Tensor values should be in the range [0, 1]")

        b, n, k, _ = contour.shape

        mask = self.ctm(contour)
        drawn = self.sobel(mask)
        drawn = self.max_(drawn)
        drawn = drawn.reshape(b, n, self.size * self.size)
        drawn = (drawn - torch.min(drawn, dim=-1)[0]) / (
            torch.max(drawn, dim=-1)[0] - torch.min(drawn, dim=-1)[0] + 1e-9
        )
        return drawn.reshape(b, n, self.size, self.size)


class Smoothing(nn.Module):
    """
    A PyTorch module for applying Gaussian smoothing to contour data.

    /!\ The smoothing operation takes into account the fact that the contour is a closed contour.
    No need to satisfy contour[0] == contour[k-1]

    The sigma smoothing operate on the dimension of the number of nodes k

    Args:
        sigma (float): The standard deviation of the Gaussian kernel.
    """

    def __init__(self, sigma):
        """
        Initializes the Smoothing module with the specified sigma.

        Args:
            sigma (float): The standard deviation of the Gaussian kernel.
        """
        super(Smoothing, self).__init__()
        self.sigma = sigma
        self.kernel = self.define_kernel()

    def define_kernel(self):
        """
        Defines the Gaussian kernel based on the specified sigma.

        Returns:
            torch.Tensor: The Gaussian kernel tensor.
        """
        mil = self.sigma * 2 * 5 * 1 // 2  # The filter extends to 5 sigma
        filter = np.arange(self.sigma * 2 * 5) - mil
        x = np.exp((-1 / 2) * (filter**2) / (2 * (self.sigma) ** 2))
        tmp = torch.tensor(x / np.sum(x), dtype=torch.float32)[None, None]
        return torch.cat([tmp, tmp])

    def forward(self, contours):
        """
        Applies Gaussian smoothing to the input contour data.

        Args:
            contours (torch.Tensor): The input contour tensor of shape (batch_size, num_contours, num_points, 2).

        Returns:
            torch.Tensor: The smoothed contour tensor.
        """
        b, n, k, _ = contours.shape
        contours = contours.reshape(b * n, k, -1)
        device = contours.device
        self.kernel = self.kernel.to(device)
        margin = k // 2
        top = contours[:, :margin]
        bot = contours[:, -margin:]

        out = torch.cat([bot, contours, top], dim=1)
        out_moved_axis = torch.moveaxis(out, -1, 1)

        smoothed_tensor = F.conv1d(out_moved_axis, self.kernel, padding="same", groups=2)
        return smoothed_tensor[:, :, margin:-margin].reshape(b, n, k, 2)


def area(contours):
    """Return the area of each polygon of each batch

    ...

    Parameters
    ----------

    contour (torch.Tensor): A 4D tensor of shape (B, N, K, 2) where B is the batch size,
                    N is the number of polygons to draw for each image,
                    K is the number of nodes per polygon,
                    2 represents the coordinates (x, y) of each point,

    Returns:
    torch.Tensor: A 2D tensor of shape (B, N) containing the area of each polygon in the batch.

    """
    b, n, k, _ = contours.shape
    contours = contours.reshape((b * n, k, -1))
    y = torch.roll(contours[:, :, 1], shifts=-1, dims=1)
    z = torch.roll(contours[:, :, 0], shifts=-1, dims=1)
    return (torch.abs(torch.sum(contours[:, :, 1] * y + contours[:, :, 0] * z, dim=-1)) / 2.0).reshape((b, n))


def perimeter(contours):
    """Return the perimeter of each polygon of each batch

    ...

    Parameters
    ----------

    contour (torch.Tensor): A 4D tensor of shape (B, N, K, 2) where B is the batch size,
                    N is the number of polygons to draw for each image,
                    K is the number of nodes per polygon,
                    2 represents the coordinates (x, y) of each point,

    Returns:
    torch.Tensor: A 2D tensor of shape (B, N) containing the perimeter of each polygon within each batch.

    """
    # Calculate the distance between consecutive points

    b, n, k, _ = contours.shape
    contours = contours.reshape((b * n, k, -1))

    distances = torch.sqrt(torch.sum((contours - torch.roll(contours, shifts=-1, dims=1)) ** 2, dim=2))
    # Sum the distances for each polygon to get the perimeter
    perimeters = torch.sum(distances, dim=-1).reshape((b, n))

    return perimeters


def hausdorff_distance(contours1, contours2):
    """Return the haussdorf distances of each polygon within each batch

    ...

    Parameters
    ----------

    contour (torch.Tensor): A 4D tensor of shape (B, N, K, 2) where B is the batch size,
                    N is the number of polygons to draw for each image,
                    K is the number of nodes per polygon,
                    2 represents the coordinates (x, y) of each point,

    Returns:
    torch.Tensor: A 2D tensor of shape (B, N) containing the hausdorff distances between each polygon within each batch.
    """
    # Compute pairwise distances

    b, n, k, _ = contours1.shape
    contours1 = contours1.reshape((b * n, k, -1))
    contours2 = contours2.reshape((b * n, k, -1))

    dists = torch.cdist(contours1, contours2)  # (B, N, N)
    # Compute the directed Hausdorff distances
    min_dist_contours1_to_contours2, _ = torch.min(dists, dim=2)  # (B, N)
    min_dist_contours2_to_contours1, _ = torch.min(dists, dim=1)  # (B, N)

    # Max of the minimum distances
    hausdorff_dist_contours1_to_contours2 = torch.max(min_dist_contours1_to_contours2, dim=1).values  # (B,)
    hausdorff_dist_contours2_to_contours1 = torch.max(min_dist_contours2_to_contours1, dim=1).values  # (B,)

    # Final Hausdorff distance is the maximum of both directed distances
    hausdorff_dist = torch.max(hausdorff_dist_contours1_to_contours2, hausdorff_dist_contours2_to_contours1)

    return hausdorff_dist.reshape((b, n))


def curvature(contour):
    """
    Computes the curvature of a given contour.

    This function calculates the curvature of a 2D contour represented as a tensor.
    The input contour is extended at the beginning and end to handle boundary conditions
    for gradient computation. The function then computes the velocity and acceleration
    of the contour points, and finally calculates the curvature using these values.

    Parameters:
    -----------
    contour : torch.Tensor
        A tensor of shape (B, N, K, 2), where:
        - B is the batch size
        - N is the number of contours in a batch
        - K is the number of points in each contour
        - 2 represents the x and y coordinates of each point

    Returns:
    --------
    torch.Tensor
        A tensor of shape (B, N, K-6) representing the curvature of each point
        in the contour, excluding the boundary points used for padding.

    Example:
    --------
    >>> import torch
    >>> contour = torch.rand(1, 1, 10, 2)  # Example contour with random points
    >>> curv = curvature(contour)
    >>> print(curv.shape)
    torch.Size([1, 1, 4])
    """

    contour = torch.cat([contour[:, :, -3:, :], contour, contour[:, :, :3, :]], dim=-2)
    b, n, k, _ = contour.shape
    contour = contour.reshape(b * n, k, -1)
    velocity = torch.gradient(contour, dim=1)[0]
    ds_dt = torch.norm(velocity, dim=-1)
    accel = torch.gradient(velocity, dim=1)[0]
    curvature = (
        torch.abs(accel[:, :, 0] * velocity[:, :, 1] - velocity[:, :, 0] * accel[:, :, 1])
        / torch.sum(velocity**2, dim=-1) ** 1.5
    )
    curvature = curvature.reshape(b, n, k, -1)
    curvature = curvature[:, :, 3:-3, 0]

    return curvature


class CleanContours:
    def __init__(self):
        """

        Initialize the CleanContours class.
        This class can remove loops inside a batch of several contours.

        The methods are only available for numpy arrays and not torch tensors.
        To clean the contours it is necessary to first apply .cpu().detach().numpy() to turn torch tensor into a numpy array.
        """
        pass

    def contour_length(self, contour):
        """
        Calculate the perimeter (total length) of a single polygon contour.

        Parameters:
        - contour (ndarray): An array of shape (N, 2) representing the vertices of the polygon.

        Returns:
        - float: Perimeter of the polygon.
        """
        diff = np.diff(contour, axis=0, append=[contour[0]])
        lengths = np.sqrt((diff**2).sum(axis=1))
        return np.sum(lengths)

    def cross_product(self, a, b):
        """
        Compute the cross product of two 2D vectors.

        Parameters:
        - a (ndarray): First vector of shape (2,).
        - b (ndarray): Second vector of shape (2,).

        Returns:
        - float: Cross product of vectors a and b.
        """
        return a[0] * b[1] - a[1] * b[0]

    def is_intersecting(self, p1, p2, p3, p4):
        """
        Check if line segment p1p2 intersects with line segment p3p4.

        Parameters:
        - p1 (ndarray): Start point of segment p1p2, shape (2,).
        - p2 (ndarray): End point of segment p1p2, shape (2,).
        - p3 (ndarray): Start point of segment p3p4, shape (2,).
        - p4 (ndarray): End point of segment p3p4, shape (2,).

        Returns:
        - bool: True if segments intersect, False otherwise.
        """
        d1 = p2 - p1
        d2 = p4 - p3
        dp = p3 - p1
        cp1 = self.cross_product(d1, dp)
        cp2 = self.cross_product(d1, p4 - p1)
        cp3 = self.cross_product(d2, -dp)
        cp4 = self.cross_product(d2, p2 - p3)
        return (np.sign(cp1) != np.sign(cp2)) & (np.sign(cp3) != np.sign(cp4))

    def find_loops(self, contour):
        """
        Find loops in a single polygon and return their start and end indices along with the loop length.

        Parameters:
        - contour (ndarray): An array of shape (N, 2) representing the vertices of the polygon.

        Returns:
        - list: List of tuples, each tuple containing:
            - ndarray: Indices of the contour forming the loop.
            - float: Length of the loop.
        """
        n = len(contour)
        loops = []

        # Create all combinations of segment pairs
        segments = np.arange(n)
        p1 = contour[segments]
        p2 = contour[(segments + 1) % n]

        for i in range(n):
            for j in range(i + 2, n):
                if j == (i + 1) % n:  # Skip adjacent edges
                    continue
                if self.is_intersecting(p1[i], p2[i], p1[j], p2[j]):
                    loop = (
                        np.concatenate((contour[i : j + 1], contour[: i + 1]), axis=0) if i > j else contour[i : j + 1]
                    )
                    loop_length = self.contour_length(loop)
                    loops.append((np.arange(i, j + 1) % n, loop_length))

        return loops

    def remove_small_loops(self, contour, threshold_length):
        """
        Remove loops smaller than the threshold length from a single polygon.

        Parameters:
        - contour (ndarray): An array of shape (N, 2) representing the vertices of the polygon.
        - threshold_length (float): Threshold length below which loops should be removed.

        Returns:
        - ndarray: Cleaned contour after removing small loops.
        """
        while True:
            loops = self.find_loops(contour)
            loops_to_remove = [loop[0] for loop in loops if loop[1] < threshold_length]
            loops_to_remove = np.concatenate(loops_to_remove)
            if not np.any(loops_to_remove):
                break
            mask = np.ones(len(contour), dtype=bool)
            mask[np.array(loops_to_remove).flatten()] = False
            contour = contour[mask]
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.append(contour, [contour[0]], axis=0)
        return contour

    def clean_contours(self, contours):
        """
        Clean multiple polygons, removing small loops within each polygon.

        Parameters:
        - contours (ndarray): array of shape (B,N,K,2) where B is the batch size, N is the number of polygon per batch and  K is the number of vertices per polygon

        Returns:
        - list of ndarrays: List of cleaned contours with B*N elements, where each contour i is an array of shape (M_i, 2).
        """
        b, n, k, _ = contours.shape
        contours = contours.reshape(b * n, k, 2)
        cleaned_contours = []
        for contour in contours:
            original_length = self.contour_length(contour)
            threshold_length = original_length / 2
            cleaned_contour = self.remove_small_loops(contour, threshold_length)
            cleaned_contours.append(cleaned_contour)

        return cleaned_contours

    def make_strictly_increasing(self, sequence, epsilon=1e-5):
        """
        Modify a sequence to ensure it is strictly increasing by adjusting values up to a small epsilon.

        Parameters:
        - sequence (list): List of numbers representing the sequence to be modified.
        - epsilon (float): Threshold value to consider two numbers as 'equal'. Default is 1e-10.

        Returns:
        - list: Modified sequence where all values are strictly increasing.
        """
        modified_sequence = sequence[:]

        for i in range(1, len(modified_sequence)):
            if modified_sequence[i] == modified_sequence[i - 1]:
                modified_sequence[i] = modified_sequence[i - 1] + epsilon
        return modified_sequence

    def interpolate(self, contour, n):

        margin = n // 10

        top = contour[:margin]
        bot = contour[-margin:-1]

        contour_init_new = np.concatenate([bot, contour, top])
        distance = np.cumsum(np.sqrt(np.sum(np.diff(contour_init_new, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]
        distance = self.make_strictly_increasing(distance)
        indices = np.linspace(0, contour_init_new.shape[0] - 1, 100).astype(int)
        indices = np.unique(indices)
        Cub = CubicSpline(distance[indices], contour_init_new[indices])
        interp_contour = Cub(np.linspace(distance[margin], distance[-margin], n))

        return interp_contour

    def clean_contours_and_interpolate(self, contours):
        """
        Clean multiple polygons, removing small loops within each polygon.

        Parameters:
        - contours (ndarray): array of shape (B,N,K,2) where B is the batch size, N is the number of polygon per batch and  K is the number of vertices per polygon

        Returns:

        - ndarray: Array of cleaned contours, of shape (B, N, K, 2).

        """
        b, n, k, _ = contours.shape
        contours = contours.reshape(b * n, k, 2)
        cleaned_contours = np.zeros((b * n, k, 2))
        for i, contour in enumerate(contours):
            original_length = self.contour_length(contour)
            threshold_length = original_length / 2
            cleaned_contour = self.remove_small_loops(contour, threshold_length)
            interpolated_contour = self.interpolate(cleaned_contour, k)
            cleaned_contours[i] = interpolated_contour
        return cleaned_contours.reshape(b, n, k, 2)