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
