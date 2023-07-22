import glob
import os

import imageio.v3 as iio
import numpy as np


class Dataloader():
    def __init__(self, args):
        self.args = args
        self.dataset_type = args.dataset_type
        self.data_dir = "/home/atsuya/tier4/nerf-reptile/data/cars"
        self.stage = "train"
        if self.dataset_type == "srn":
            self.data = SRNDataset
        else:
            raise NotImplementedError("Unsupported dataset type", self.dataset_type)

    def get_data(self, stage="train"):
        if self.stage == "train":
            train_data = self.data(self.data_dir, stage=stage)
            return train_data
        elif self.stage == "test":
            test_data = self.data(self.data_dir, stage=stage)
            return test_data
        else:
            raise NotImplementedError("Unsupported stage", self.stage)


class SRNDataset():
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self, path, stage="train", image_size=(128, 128), world_scale=1.0
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        self.base_path = path + "_" + stage
        self.dataset_name = os.path.basename(path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        is_car = "car" in self.dataset_name

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        # self.image_to_tensor = get_image_to_tensor_balanced()
        # self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = np.diag(
            np.array([1, -1, -1, 1], dtype=np.float32)
        )

        if is_car:
            self.z_near = 0.8
            self.z_far = 1.8
        else:
            NotImplementedError("Unsupported dataset", self.dataset_name)
        self.lindisp = False

    def __len__(self):
        return len(self.intrins)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())
            original_image_size = (height, width)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        index_3dbb = [None, None, None]
        max_xyz = [0, 0, 0]
        index = 0
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = iio.imread(rgb_path)[..., :3]
            img = transform_img_uint8_to_float(img)
            mask = (img != 1.).all(axis=-1)[..., None].astype(np.float32)

            pose = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)

            pose = pose @ self._coord_trans  # 何をしている
            new_x, new_y, new_z = np.abs(pose[:3, 3])
            if new_x > max_xyz[0]:
                index_3dbb[0] = index
                max_xyz[0] = new_x
            elif new_y > max_xyz[1]:
                index_3dbb[1] = index
                max_xyz[1] = new_y
            elif new_z > max_xyz[2]:
                index_3dbb[2] = index
                max_xyz[2] = new_z
            index += 1
            rows = np.any(mask, axis=1)  # 白の背景を除く
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]  # マスクされていない最初の行
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

            mask_indices = np.where(mask == 1.)
            for col_indice in np.unique(mask_indices[0]):
                row_indices = mask_indices[1][np.where(mask_indices[0] == col_indice)]
                mask[col_indice, row_indices[0]:row_indices[-1]] = 1.
            all_imgs.append(img)
            all_masks.append(mask)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = np.stack(all_imgs)
        all_poses = np.stack(all_poses)
        all_masks = np.stack(all_masks)
        all_bboxes = np.stack(all_bboxes)

        if all_imgs.shape[-3:-1] != original_image_size:
            scale = all_imgs.shape[-2] / original_image_size[0]
            focal *= scale
            cx *= scale
            cy *= scale

        # index_3dbb [x, y, z]の軸上にある画像のインデックス
        # radius = np.max(np.linalg.norm(all_poses[index_3dbb][:, :3, 3], ord=2, axis=1))
        for index, image_index in enumerate(index_3dbb):
            two_d_bb = all_bboxes[image_index]
            if index == 0:
                dx = two_d_bb[2] - two_d_bb[0]
                hx = two_d_bb[3] - two_d_bb[1]
            elif index == 1:
                wy = two_d_bb[2] - two_d_bb[0]
                hy = two_d_bb[3] - two_d_bb[1]
            elif index == 2:
                pass
        wdh = np.array([wy * (hx / hy), dx, hx], dtype=np.float32) / focal

        # if all_imgs.shape[-3:-1] != self.image_size:
        #     scale = self.image_size[0] / all_imgs.shape[-2]
        #     focal *= scale
        #     cx *= scale
        #     cy *= scale
        #     all_bboxes *= scale

        #     all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
        #     all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = np.array(focal, dtype=np.float32)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": np.array([cx, cy], dtype=np.float32),
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
            "wdh3dbb": wdh * 1.2,
        }
        return result


def transform_img_uint8_to_float(img):
    img = img.astype(np.float32) / 255.0
    return img


if "__main__" == __name__:
    from args_parser import config_parser
    args = config_parser().parse_args()
    dataset = Dataloader(args).get_data("test")
    print(dataset[0])
