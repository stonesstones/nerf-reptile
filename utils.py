import torch


def img2mse(rgb, target):
    return torch.mean((rgb - target)**2)


def mse2psnr(x):
    return -10 * torch.log10(x) / torch.log10(torch.tensor(10.))


def to8b(x):
    return (255 * torch.clip(x, 0, 1)).to(torch.uint8)


def box_pts(pts, dirs, dim=None):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in world frame, [N_pts, 3]
        dirs: Corresponding 3D directions given in world frame, [N_pts, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3] 幅、高さ、奥行き
        inverse: if true pts and dirs should be given in object frame and are transofmed back into world frame, bool
            For inverse: pts, [N_pts, N_obj, 3]; dirs, [N_pts, N_obj, 3]

    Returns:
        pts_w: 3d points transformed into object frame (world frame for inverse task)
        dir_w: unit - 3d directions transformed into object frame (world frame for inverse task)
    """
    pts_scaled = scale_frames(pts, dim)
    dirs_scaled = scale_frames(dirs, dim)
    # Normalize direction
    dirs_scaled = dirs_scaled / torch.linalg.vector_norm(dirs_scaled, dim=-1, keepdim=True)

    # Get the intersection with each Bounding Box
    z_vals_in_o, z_vals_out_o, intersection_map = ray_box_intersection(pts_scaled, dirs_scaled)

    if z_vals_in_o is not None:
        # Calculate the intersection points for each box in each object frame
        pts_box_in_o = pts_scaled[intersection_map] + \
            z_vals_in_o[:, None] * dirs_scaled[intersection_map]  # ある物体に交差する最初の点(object frame)
        pts_box_in_w = scale_frames(pts_box_in_o, dim, inverse=True)  # ある物体に交差する最初の点(world frame)
        rays_o_in_w = pts[intersection_map]
        rays_d_in_w = dirs[intersection_map]
        z_vals_in_w = torch.linalg.vector_norm(pts_box_in_w - rays_o_in_w, dim=-1) / torch.linalg.vector_norm(rays_d_in_w, dim=-1)

        # Get the far intersection points and integration steps for each ray-box intersection in world and object frames
        pts_box_out_o = pts_scaled[intersection_map] + \
            z_vals_out_o[:, None] * dirs_scaled[intersection_map]
        pts_box_out_w = scale_frames(pts_box_out_o, dim, inverse=True)
        z_vals_out_w = torch.linalg.vector_norm(pts_box_out_w - rays_o_in_w, dim=-1) / torch.linalg.vector_norm(rays_d_in_w, dim=-1)

        # get viewing direction
        viewdirs_box_o = dirs_scaled[intersection_map]
        viewdirs_box_w = rays_d_in_w / torch.linalg.vector_norm(rays_d_in_w, dim=1)[:, None]
    else:
        # In case no ray intersects with any object return empty lists
        z_vals_in_w = z_vals_out_w = []
        pts_box_in_w = pts_box_in_o = []
        viewdirs_box_w = viewdirs_box_o = []
        z_vals_out_o = z_vals_in_o = []
    pts_o_o = pts_scaled[intersection_map]
    return pts_o_o, pts_box_in_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w, \
        pts_box_in_o, viewdirs_box_o, z_vals_in_o, z_vals_out_o, \
        intersection_map


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected

    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary

    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified

    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = torch.ones_like(ray_o) * -1.  # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = torch.ones_like(ray_o)  # tf.constant([1., 1., 1.])
    inv_d = torch.reciprocal(ray_d)
    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d
    t0 = torch.minimum(t_min, t_max)
    t1 = torch.maximum(t_min, t_max)
    t_near = torch.maximum(torch.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = torch.minimum(torch.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])
    # Check if rays are inside boxes
    intersection_map = torch.where(t_far > t_near)
    # Check that boxes are in front of the ray origin
    positive_far = torch.where(t_far[intersection_map] > 0)
    intersection_map = (intersection_map[0][positive_far[0]],)

    if not intersection_map[0].shape[0] == 0:
        z_ray_in = t_near[intersection_map]
        z_ray_out = t_far[intersection_map]
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def scale_frames(p, sc_factor, inverse=False):
    device = p.device
    dim = torch.tensor([1., 1., 1.], dtype=torch.float32).to(device) * sc_factor
    half_dim = dim / 2
    scaling_factor = (1 / (half_dim + 1e-9))
    if not inverse:
        p_scaled = scaling_factor * p
    else:
        p_scaled = (1 / scaling_factor) * p
    return p_scaled
