import os
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils import colors
from torch import nn

from args_parser import config_parser
from data_loader import Dataloader
from logger import Logger
from nerf_field import NeRFField
from nerf_renders import AccumulationRenderer, RGBRenderer, UniformSampler
from utils import *


class pipeline():

    def __init__(self, img_id: int, dataset, logger: Optional[Logger], device, args):
        self.img_id = img_id
        self.device = device
        self.args = args
        self.data = dataset[img_id]
        self.near = dataset.z_near
        self.far = dataset.z_far
        self.num_coarse_samples = args.N_samples
        self.logger = logger
        self._build_models()

    def _build_models(self, restore=False):
        self.field = NeRFField(4, 128, 4, 128, (4,)).to(self.device)
        self.params = list(self.field.parameters()) + list(self.field.position_encoding.parameters()) + list(self.field.direction_encoding.parameters())
        self.optimizer = torch.optim.Adam(lr=self.args.lrate, params=self.params)
        self.models = {"field": self.field, "optimizer": self.optimizer}

        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        if self.args.lrate_decay > 0:
            def lr_lambda(step):
                return 0.1**(step / (self.args.lrate_decay * 1000))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        if restore:
            self._restore(self.restore_path)

    def _restore(self):
        pass

    def _save_models(self):
        pass

    def get_field_params(self):
        return self.field.mlp_base.state_dict()

    @staticmethod
    def get_initial_field_params():
        field = NeRFField(4, 128, 4, 128, (4,))
        field_state_dict = field.mlp_base.state_dict()
        for key in field_state_dict:
            nn.init.zeros_(field_state_dict[key])
        return field_state_dict

    def load_field_params(self, state_dict):
        self.field.mlp_base.load_state_dict(state_dict)

    def get_rays(self, H, W, focal, c2w):
        """Get ray origins, directions from a pinhole camera.
        Returns: [rays_o + rays_d, H, W, 3]
        """
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d

    def train(self):
        images = self.data['images']
        dim = self.data["wdh3dbb"]
        H, W = images.shape[1:3]
        focal = self.data['focal']
        self._hwf = [int(H), int(W), focal]
        i_eval = np.arange(images.shape[0])
        poses = self.data['poses'][:, :3, :4]
        rays = [self.get_rays(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [batch, rays_o + rays_d, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], axis=1)
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [N*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)

        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        images = torch.tensor(images, dtype=torch.float32).to(self.args.device)
        poses = torch.tensor(poses, dtype=torch.float32).to(self.args.device)
        dim = torch.tensor(dim, dtype=torch.float32).to(self.args.device)
        rays_rgb = torch.tensor(rays_rgb, dtype=torch.float32).to(self.args.device)

        use_batching = not self.args.no_batching
        if use_batching:
            i_batch = 0

        print("Begin training")

        inner_epoch = 0
        while inner_epoch < self.args.n_inner_epochs:
            time0 = time.time()

            if use_batching:
                # Random over all images
                batch = rays_rgb[i_batch:i_batch + self.args.N_rand]  # [B, 2+1, 3]
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:2], batch[2],  # [ray_o + ray_d, B, 3], [1, N*H*W, 3] [1, N*H*W, 1]
                i_batch += self.args.N_rand

                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0
                    inner_epoch += 1
            else:
                raise NotImplementedError
            rgb, disp = self.render(H, W, focal, chunk=self.args.chunk, dim=dim, rays=batch_rays)

            img_loss = img2mse(rgb, target_s)
            psnr = mse2psnr(img_loss)
            loss = img_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            dt = time.time() - time0

            # Rest is logging
            scalar_logs = OrderedDict()
            histogram_logs = OrderedDict()
            image_logs = OrderedDict()
            debug = False
            if debug:
                # if self.logger.should_record(self.args.i_weights):
                #     self.logger.save_weights(self.models, self.logger.global_step)
                if self.logger.global_step < 10 or self.logger.should_record(self.args.i_print):
                    print("\n##############################################")
                    print(f"object id: {self.img_id}, global step: {self.logger.global_step}")
                    print('iter time {:.05f}'.format(dt))
                    if self.logger.should_record(self.args.i_print):
                        scalar_logs.update({
                            "train_loss": loss.item(),
                            "train_psnr": psnr.item(),
                        })
                    self.logger.add_scalars(scalar_logs)

                # if self.logger.should_record(self.args.i_img) and not self.logger.global_step == 0:
            if inner_epoch == self.args.n_inner_epochs:
                with torch.no_grad():
                    # Log a rendered validation view to Tensorboard
                    img_i = np.random.choice(i_eval)
                    target_i = images[img_i]
                    pose_i = poses[img_i, :3, :4].cpu().numpy()
                    rgb, disp = self.render(H, W, focal, chunk=self.args.chunk, c2w=pose_i, dim=dim)
                    psnr = mse2psnr(img2mse(rgb, target_i))
                    image_logs.update({
                        'eval_rgb': to8b(rgb)[None],
                        'eval_disp': to8b(disp)[None],
                        'eval_rgb_holdout': target_i[None],
                    })
                    scalar_logs.update({'psnr_eval': psnr.item()})
                self.logger.add_scalars(scalar_logs, self.logger.outer_epoch)
                self.logger.add_histograms(histogram_logs, self.logger.outer_epoch)
                self.logger.add_images(image_logs, self.logger.outer_epoch)
                self.logger.flush()
            self.logger.add_global_step()

    def render(self, H, W, focal, chunk, dim, rays=None, c2w=None, use_viewdirs=False):
        if c2w is not None:
            rays_o, rays_d = self.get_rays(H, W, focal, c2w)
            rays_o, rays_d = torch.tensor(rays_o).to(self.device), torch.tensor(rays_d).to(self.device)
        else:
            rays_o, rays_d = rays

        if use_viewdirs:
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        shape = rays_d.shape

        # create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        pts_o_o, pts_box_in_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w,\
            pts_box_in_o, viewdirs_box_o, z_vals_in_o, z_vals_out_o, \
            intersection_map = box_pts(rays_o, rays_d, dim=dim)
        nears, fars = z_vals_in_o[..., None] * torch.ones_like(rays_d[..., :1][intersection_map]), z_vals_out_o[..., None] * torch.ones_like(rays_d[..., :1][intersection_map])
        pixel_area = torch.ones_like(rays_d[..., :1][intersection_map])  # for mip-nerf

        # sampling
        ray_bundle = RayBundle(origins=pts_o_o, directions=viewdirs_box_o, pixel_area=pixel_area, nears=nears, fars=fars)
        sampler_uniform = UniformSampler(num_samples=self.num_coarse_samples)
        ray_samples_uniform = sampler_uniform(ray_bundle)

        # get field representation
        field_outputs = self.field(ray_samples_uniform)

        # rendering
        weights = ray_samples_uniform.get_weights(field_outputs[FieldHeadNames.DENSITY])

        num_rays = rays_o.shape[0]
        ray_indices = intersection_map[0].unsqueeze(-1).repeat(1, weights.shape[1]).reshape(-1)
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB].reshape(-1, 3),
            weights=weights.reshape(-1, 1),
            ray_indices=ray_indices,
            num_rays=num_rays
        )
        disp = self.renderer_accumulation(
            weights=weights.reshape(-1, 1),
            ray_indices=ray_indices,
            num_rays=num_rays
        )
        rgb = torch.reshape(rgb, list(shape[:-1]) + [3])
        disp = torch.reshape(disp, list(shape[:-1]) + [1])
        return rgb, disp


def main():
    args = config_parser().parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    args.device = device
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    if args.dataset_type == "srn":
        train_dataset = Dataloader(args).get_data("train")
        # test_dataset = Dataloader(args).get_data("test")

    base_dir = args.basedir + time.strftime("%H%M%S-%m-%d-%Y")
    os.makedirs(base_dir, exist_ok=True)

    clients = args.n_classes if len(train_dataset) >= args.n_classes else len(train_dataset)

    # for logger
    loggers = []
    for k in range(clients):
        obj_dir = os.path.join(base_dir, "{}_obj".format(k))
        # obj_model_dir = os.path.join(obj_dir, "models")
        os.makedirs(obj_dir, exist_ok=True)
        # os.makedirs(obj_model_dir, exist_ok=True)
        loggers.append(Logger(obj_dir, global_step=0))

    server_field_params = pipeline.get_initial_field_params()
    for outer_epoch in range(args.n_outer_epochs):
        field_params = pipeline.get_initial_field_params()
        for k in range(clients):
            print("\n##############################################")
            print("outer epoch: {}, client {}".format(outer_epoch, k))
            trainer = pipeline(k, train_dataset, loggers[k], device, args)
            if outer_epoch > 0:
                trainer.load_field_params(server_field_params)
            trainer.train()
            loggers[k].add_outer_epoch()
            tmp_field_params = trainer.get_field_params()
            for key in field_params:
                field_params[key] += tmp_field_params[key].cpu() / clients
        server_field_params = field_params

        torch.save(server_field_params, os.path.join(base_dir, f"server_field_params{outer_epoch}.pth"))


if __name__ == '__main__':
    main()
