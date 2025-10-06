import numpy as np
from unet_utils import extract_patches
import gryds
# from gryds.transformers import BSplineTransformation2D 

def augmentation_brightness(images, segmentations, patch_size, patches_per_im, batch_size, brightness_range=(-0.2, 0.2)):

    total_patches = len(images) * patches_per_im
    nr_batches = int(np.ceil(total_patches / batch_size))

    while True:
        x, y = extract_patches(images, segmentations, patch_size, patches_per_im, seed=np.random.randint(0, 500))

        #random brightness per patch
        for i in range(len(x)):
            offset = np.random.uniform(brightness_range[0], brightness_range[1])
            x[i] = np.clip(x[i] + offset, 0.0, 1.0)  

        
        for idx in range(nr_batches):
            x_batch = x[idx * batch_size:(idx + 1) * batch_size]
            y_batch = y[idx * batch_size:(idx + 1) * batch_size]
            yield x_batch, y_batch



def augmentation_gryds(images, segmentations, patch_size, patches_per_im, batch_size,
                       brightness_range=(-0.2, 0.2), seed=None):

    rng = np.random.default_rng(seed)
    

    while True:  
        
        x, y = extract_patches(images, segmentations, patch_size, patches_per_im, seed=seed)
        
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb, yb = x[start:end], y[start:end]

            # Augmentation part
            x_aug, y_aug = [], []
            for i in range(len(xb)):
                xi, yi = xb[i], yb[i]

                #B-spline grid
                random_grid = np.random.rand(2, 3, 3)
                random_grid -= 0.5
                random_grid /= 5
                bspline = gryds.BSplineTransformation(random_grid)

                # Warp image (RGB channels)
                warped_channels = []
                for c in range(xi.shape[-1]):
                    interp = gryds.Interpolator(xi[..., c])
                    warped_channels.append(interp.transform(bspline))
                warped_image = np.stack(warped_channels, axis=-1)

                # Warp mask
                interp_mask = gryds.Interpolator(yi[..., 0])
                warped_mask = interp_mask.transform(bspline)[..., None]

                # Brightness adjustment
                factor = rng.uniform(*brightness_range)
                warped_image = np.clip(warped_image * factor, 0, 255).astype(xi.dtype)

                x_aug.append(warped_image)
                y_aug.append(warped_mask.astype(yi.dtype))

            yield np.array(x_aug), np.array(y_aug)


