import torchvision, pathlib
import torch
from ..nuscenes.camera.augmentations import StrongAug, GeometricAug
from ..registry import PIPELINES
from PIL import Image
import numpy as np

@PIPELINES.register_module
class LoadDataTransform(torchvision.transforms.ToTensor):
    def __init__(self, dataset_dir, image_config, augment='none'):
        super().__init__()

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.image_config = image_config

        xform = {
            'none': [],
            'strong': [StrongAug()],
            'geometric': [StrongAug(), GeometricAug()],
        }[augment] + [torchvision.transforms.ToTensor()]

        self.img_transform = torchvision.transforms.Compose(xform)
        self.to_tensor = super().__call__

    def get_cameras(self, sample: dict, h, w, top_crop):
        """
        Note: we invert I and E here for convenience.
        """
        images = list()
        intrinsics = list()
        for image_path, I_original in zip(sample['images'], sample['intrinsics']):
            h_resize = h + top_crop
            w_resize = w

            image = Image.open(self.dataset_dir / image_path)

            image_new = image.resize((w_resize, h_resize), resample=Image.BILINEAR)
            image_new = image_new.crop((0, top_crop, image_new.width, image_new.height))

            I = np.float32(I_original)
            I[0, 0] *= w_resize / image.width
            I[0, 2] *= w_resize / image.width
            I[1, 1] *= h_resize / image.height
            I[1, 2] *= h_resize / image.height
            I[1, 2] -= top_crop

            images.append(self.img_transform(image_new))
            intrinsics.append(torch.tensor(I))

        return {
            'cam_idx': torch.LongTensor(sample.cam_ids),
            'image': torch.stack(images, 0),
            'intrinsics': torch.stack(intrinsics, 0),
            'extrinsics': torch.tensor(np.float32(sample.extrinsics)),
        }


    def __call__(self, res, info):
        result = dict()
        result.update(self.get_cameras(res, **self.image_config))

        return result, info
