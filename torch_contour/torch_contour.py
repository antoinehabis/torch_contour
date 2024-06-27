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


    def forward(self, contour):
        """Return the distance map of the given size given the contour.

        Raises
        ------
        ValueError
            If the values of the contour or not between 0 and 1.
        """

        b = contour.shape[0]
        mesh = (
            torch.unsqueeze(
                torch.stack(
                    torch.meshgrid(torch.arange(self.size), torch.arange(self.size)),
                    dim=-1,
                ).reshape(-1, 2),
                dim=1,
            )
            / self.size
        )
        mesh = mesh.unsqueeze(0).repeat(b,1,1,1)
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
        out0 = sum_angles.reshape(b, self.size, self.size)
        mask = torch.unsqueeze(out0, dim=0)
        
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


    def forward(self, contour):
        """Return the distance map of the given size given the contour.

        Raises
        ------
        ValueError
            If the values of the contour or not between 0 and 1.
        """
        b = contour.shape[0]
        mesh = (
            torch.unsqueeze(
                torch.stack(
                    torch.meshgrid(torch.arange(self.size), torch.arange(self.size)),
                    dim=-1,
                ).reshape(-1, 2),
                dim=1,
            )
            / self.size
        )
        mesh = mesh.unsqueeze(0).repeat(b,1,1,1)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        if (contour < 0).any() or (contour > 1).any():
            raise ValueError("Tensor values should be in the range [0, 1]")
        


        contour = torch.unsqueeze(contour, dim=1)
        diff = - mesh + contour
        min_diff = torch.min(torch.norm(diff, dim=-1), dim=2)[0]
        min_diff = min_diff.reshape((b,self.size, self.size))
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
        resize = sum_angles.reshape(b, self.size, self.size)
        dmap = torch.unsqueeze((resize * min_diff) / torch.max(resize * min_diff), 0)

        return dmap
    


    
    

def area(polygons):
    """
    Computes the area using the shoelace formula (also known as Gauss's area formula) for polygons.

    Parameters:
    x (torch.Tensor): A 3D tensor of shape (B, 2, N) where B is the batch size, 
                    2 represents the coordinates (x, y) of each point, 
                    and N is the number of points in each polygon.

    Returns:
    torch.Tensor: A 1D tensor of shape (B,N,) containing the area of each polygon in the batch.

    """
    
    y = torch.prod(torch.roll(polygons[:,1,:],shifts=-1,dims=1), dim = 1)
    z = torch.prod(torch.roll(polygons[:,1,:],shifts=1,dims=1), dim = 1)

    return (torch.abs(y-z)/2.)[None]



def perimeter(polygons):
    """
    Computes the perimeter of each polygon in a batch of 2D polygons.

    Parameters:
    polygons (torch.Tensor): A 3D tensor of shape (B, 2, N) where B is the batch size,
                             N is the number of points in each polygon, and 2 represents
                             the coordinates (x, y) of each point.

    Returns:
    torch.Tensor: A 1D tensor of shape (B,) containing the perimeter of each polygon in the batch.
    """
    # Calculate the distance between consecutive points
    distances = torch.sqrt(torch.sum((polygons - torch.roll(polygons, shifts=-1, dims=2)) ** 2, dim=1))
    # Sum the distances for each polygon to get the perimeter
    perimeters = torch.sum(distances, dim=-1)
    
    return perimeters



def hausdorff_distance(polygons1, polygons2):
    """
    Computes the Hausdorff distance between two batches of 2D polygons.

    Parameters:
    polygons1 (torch.Tensor): A tensor of shape (B, 2, N) representing the first batch of polygons,
                        where B is the batch size and N is the number of points in each polygon.
    polygons2 (torch.Tensor): A tensor of shape (B, 2, N) representing the second batch of polygons,
                        where B is the batch size and N is the number of points in each polygon.

    Returns:
    torch.Tensor: A 1D tensor of shape (B,) containing the Hausdorff distance for each pair of polygons.
    """
    # Compute pairwise distances
    polygons1 = polygons1.permute(0, 2, 1)  # (B, N, 1, 2)
    polygons2 = polygons2.permute(0, 2, 1)  # (B, 1, N, 2)

    dists = torch.cdist(polygons1, polygons2)  # (B, N, N)
    # Compute the directed Hausdorff distances
    min_dist_polygons1_to_polygons2, _ = torch.min(dists, dim=2)  # (B, N)
    min_dist_polygons2_to_polygons1, _ = torch.min(dists, dim=1)  # (B, N)

    # Max of the minimum distances
    hausdorff_dist_polygons1_to_polygons2 = torch.max(min_dist_polygons1_to_polygons2, dim=1).values  # (B,)
    hausdorff_dist_polygons2_to_polygons1 = torch.max(min_dist_polygons2_to_polygons1, dim=1).values  # (B,)

    # Final Hausdorff distance is the maximum of both directed distances
    hausdorff_dist = torch.max(hausdorff_dist_polygons1_to_polygons2, hausdorff_dist_polygons2_to_polygons1)

    return hausdorff_dist

