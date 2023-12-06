import torch
import torch.nn as nn


class Contour_to_mask(nn.Module):
    def __init__(self, size, k=1e5, eps=1e-5):
        super().__init__()
        self.k = k
        self.eps = eps
        self.size = size
        self.mesh = torch.unsqueeze(
            torch.stack(
                torch.meshgrid(torch.arange(self.size), torch.arange(self.size)),
                dim=-1,
            ).reshape(-1, 2),
            dim=1,
        )

    def forward(self, contour):

        
        contour = torch.unsqueeze(contour, dim=0)
        diff = -self.mesh + contour
        roll_diff = torch.roll(diff, -1, dims=1)
        sign = diff * torch.roll(roll_diff, 1, dims=2)
        sign = sign[:, :, 1] - sign[:, :, 0]
        sign = torch.tanh(self.k * sign)
        norm_diff = torch.norm(diff, dim=2)
        norm_roll = torch.norm(roll_diff, dim=2)
        scalar_product = torch.sum(diff * roll_diff, dim=2)
        clip = scalar_product / (norm_diff * norm_roll + self.eps)
        angles = torch.arccos(torch.clip(clip, eps, 1 - self.eps))
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        sum_angles = torch.clip(-torch.sum(sign * angles, dim=1) / (2 * torch.pi), 0, 1)
        out0 = sum_angles.reshape(1, self.size, self.size)
        mask = torch.unsqueeze(out0, dim=0)

        return mask
