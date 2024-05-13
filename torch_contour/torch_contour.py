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
        )/self.size

    def forward(self, contour):

        contours = torch.unsqueeze(contour, dim=0)
        diff = -self.mesh + contours
        roll_diff = torch.roll(diff, -1, dims=1)
        sign = diff * torch.roll(roll_diff, 1, dims=2)
        sign = sign[:, :, 1] - sign[:, :, 0]
        sign = torch.tanh(self.k * sign)
        norm_diff = torch.linalg.vector_norm(diff, dim=2)
        norm_roll = torch.linalg.vector_norm(roll_diff, dim=2)
        scalar_product = torch.sum(diff * roll_diff, dim=2)
        clip = torch.clamp(scalar_product / (norm_diff * norm_roll ),-1 + self.eps, 1 - self.eps)
        angles = torch.acos(clip)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        sum_angles = torch.clamp(torch.sum(sign * angles, dim=1) / (2 * torch.pi), 0, 1)
        out0 = sum_angles.reshape(1, self.size, self.size)
        mask = torch.unsqueeze(out0, dim=0)

        return mask



class Contour_to_distance_map(nn.Module):
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
        )/self.size

    def forward(self, contour):

        k = 1e5
        contour = torch.unsqueeze(contour, dim=0)
        diff = -self.mesh + contour
        min_diff = torch.min(torch.norm(diff, dim=-1), dim=1)[0]
        min_diff = min_diff.reshape((self.size, self.size))
        roll_diff = torch.roll(diff, -1, dims=1)
        sign = diff * torch.roll(roll_diff, 1, dims=2)
        sign = sign[:, :, 1] - sign[:, :, 0]
        sign = torch.tanh(k * sign)
        norm_diff = torch.clip(torch.norm(diff, dim=2), self.eps, None)
        norm_roll = torch.clip(torch.norm(roll_diff, dim=2), self.eps, None)
        scalar_product = torch.sum(diff * roll_diff, dim=2)
        clip = torch.clip(scalar_product / (norm_diff * norm_roll), -1 + self.eps, 1 - self.eps)
        angles = torch.arccos(clip)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        sum_angles = torch.sum(sign * angles, dim=1) / (2 * torch.pi)
        resize = sum_angles.reshape(1, self.size, self.size)
        dmap = torch.unsqueeze((resize * min_diff) / torch.max(resize * min_diff),0)
        return dmap

