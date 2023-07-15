import numpy as np
import torch

def scale_frames(p, scaling_factor, inverse=False):
    """Scales the frames by the given scaling factor.
    Args:
        p: A tensor of shape [batch_size, 3] representing the points.
        scaling_factor: A float representing the scaling factor.
        inverse: A boolean representing whether to scale the frames in the inverse direction.
    Returns:
        A tensor of shape [batch_size, 3] representing the scaled points.
    """
    if inverse:
        scaling_factor = 1.0 / scaling_factor
    
def box_pts(rays, pose, theta_y, dim=None):
    """Returns a list of points on the box defined by the rays and pose.
    Args:
        rays: A tensor of shape [batch_size, 3] representing the rays.
        pose: A tensor of shape [batch_size, 4, 4] representing the pose.
        theta_y: A float representing the rotation of the box around the y-axis.
        dim: An integer representing which dimension of the box to return.
    Returns:
        A tensor of shape [batch_size, 8, 3] representing the points on the box.
    """
    # Get the box corners in the local frame.
    box_corners_local = box_pts_local(theta_y, dim)
    # Transform the box corners into the global frame.
    box_corners_global = torch.matmul(pose[:, :3, :3], box_corners_local.unsqueeze(-1)).squeeze(-1)
    box_corners_global += pose[:, :3, 3]
    # Compute the intersection of the rays with the box.
    box_pts = rays.unsqueeze(1) * box_corners_global.unsqueeze(2)
    return box_pts