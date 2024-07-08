import torch
import torch.nn as nn


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

        b,n,k,_ = contour.shape
        contour = contour.reshape(b*n,k,-1)
        mesh = self.mesh.unsqueeze(0).repeat(b*n, 1, 1, 1)
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
        out0 = sum_angles.reshape(b*n, self.size, self.size)
        mask = out0.reshape(b,n,self.size, self.size)

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
      
        b,n,k,_ = contour.shape
        contour = contour.reshape(b*n,k,_)
        mesh = self.mesh.unsqueeze(0).repeat(b*n, 1, 1, 1)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        if (contour < 0).any() or (contour > 1).any():
            raise ValueError("Tensor values should be in the range [0, 1]")

        contour = torch.unsqueeze(contour, dim=1)
        diff = -mesh + contour
        min_diff = torch.min(torch.norm(diff, dim=-1), dim=2)[0]
        min_diff = min_diff.reshape((b*n, self.size, self.size))
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
        resize = sum_angles.reshape(b*n, self.size, self.size)
        dmap = torch.unsqueeze((resize * min_diff) / torch.max(resize * min_diff), 0)
        dmap  = dmap.reshape(b,n,self.size, self.size)
        return dmap

class Draw_contour(nn.Module):
    """This layer draws a contour

    ...

    Attributes
    ----------
    size: int
        the size of the output image
    k: float
        the control parameter to approximate the sign function

    Methods
    -------
    forward(contour)
        forward function that draws the contour
    """

    def __init__(self, size, k=1e3):
        """
        Parameters
        ----------
        size: int
            the size of the output image
        k: float
            the control parameter to approximate the dirac function

        """
        super().__init__()
        self.k = k
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

        b, n,k, _ = contour.shape
        contour  = contour.reshape(b*n,k,-1)
        mesh = self.mesh.unsqueeze(0).repeat(b*n, 1, 1, 1)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        if (contour < 0).any() or (contour > 1).any():
            raise ValueError("Tensor values should be in the range [0, 1]")

        contour = torch.unsqueeze(contour, dim=1)
        diff = -mesh + contour
        roll_diff = torch.roll(diff, -1, dims=2)
        x = torch.max(torch.exp(-self.k*10 * torch.sum(diff**2, -1).reshape(b*n, -1, k)), dim=-1)[0].reshape(
            b*n, self.size, self.size
        )
        roll_diff = torch.roll(diff, -1, dims=2)
        norm_diff = torch.linalg.vector_norm(diff, dim=3)
        norm_roll = torch.linalg.vector_norm(roll_diff, dim=3)
        scalar_product = torch.sum(diff * roll_diff, dim=3)
        clip = scalar_product / (norm_diff * norm_roll)
        angles = torch.max(torch.exp(-self.k * (-1 - clip) ** 2), axis=-1)[0]
        contour_drawn = angles.reshape(b*n, self.size, self.size) + x

        return contour_drawn.reshape(b,n,self.size, self.size)


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
    b, n,k, _ = contours.shape
    contours  = contours.reshape((b*n,k,-1))
    y = torch.roll(contours[:,:,1], shifts=-1, dims=1)
    z = torch.roll(contours[:,:,0], shifts=-1, dims=1)
    return (torch.abs(torch.sum(contours[:,:,1] * y + contours[:,:,0] * z,dim=-1))/2.).reshape((b,n))




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

    b, n,k, _ = contours.shape
    contours  = contours.reshape((b*n,k,-1))

    distances = torch.sqrt(torch.sum((contours - torch.roll(contours, shifts=-1, dims=1)) ** 2, dim=2))
    # Sum the distances for each polygon to get the perimeter
    perimeters = torch.sum(distances, dim=-1).reshape((b,n))

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

    b, n,k, _ = contours1.shape
    contours1  = contours1.reshape((b*n,k,-1))
    contours2  = contours2.reshape((b*n,k,-1))


    dists = torch.cdist(contours1, contours2)  # (B, N, N)
    # Compute the directed Hausdorff distances
    min_dist_contours1_to_contours2, _ = torch.min(dists, dim=2)  # (B, N)
    min_dist_contours2_to_contours1, _ = torch.min(dists, dim=1)  # (B, N)

    # Max of the minimum distances
    hausdorff_dist_contours1_to_contours2 = torch.max(min_dist_contours1_to_contours2, dim=1).values  # (B,)
    hausdorff_dist_contours2_to_contours1 = torch.max(min_dist_contours2_to_contours1, dim=1).values  # (B,)

    # Final Hausdorff distance is the maximum of both directed distances
    hausdorff_dist = torch.max(hausdorff_dist_contours1_to_contours2, hausdorff_dist_contours2_to_contours1)

    return hausdorff_dist.reshape((b,n))
