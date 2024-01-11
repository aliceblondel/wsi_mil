import random
import numpy as np
import torch
from scipy import linalg
import skimage.color as skimage_color

HEX2RGB = np.array([
    [0.650, 0.704, 0.286], # H
    [0.216, 0.801, 0.558], # E
    [0.0, 0.0, 0.0]
])
HEX2RGB[2, :] = np.cross(HEX2RGB[0, :], HEX2RGB[1, :])
RGB2HEX = linalg.inv(HEX2RGB)

def sample_augmentation(gaussian_mean, gaussian_std, uniform_min, uniform_max, as_tensor=True):
    if uniform_min is not None and uniform_max is not None:
        h_factor = random.uniform(uniform_min, uniform_max)
        e_factor = random.uniform(uniform_min, uniform_max)
    elif gaussian_mean is not None and gaussian_std is not None:
        h_factor = random.gauss(gaussian_mean, gaussian_std)
        e_factor = random.gauss(gaussian_mean, gaussian_std)
    else:
        raise ValueError('should not happen')
    if as_tensor:
        return torch.tensor([h_factor, e_factor, 1]).unsqueeze(1)
    return h_factor, e_factor

class HEStainAugmentation:
    """ 
    Deconvolve input tile into H, E, and Residual channels. 
    Then sample one gaussian-random factor for H and for E,
    and scale these stains accordingly.
    Reapply scaled stains to produced stain-augmented tiles. 
    """

    def __init__(self, gaussian_mean=None, gaussian_std=None, uniform_min=None, uniform_max=None):
        assert (gaussian_mean is None and gaussian_std is None) != (uniform_min is None and uniform_max is None)
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max

    def select_factors(self):
        h_factor = random.gauss(self.gaussian_mean, self.gaussian_std)
        e_factor = random.gauss(self.gaussian_mean, self.gaussian_std)
        return h_factor, e_factor

    def __call__(self, img, h_factor=None, e_factor=None, **kwargs):
        
        # Rescale if needed
        if img.dtype == np.uint8:
            img = img / 255.
            rescale = True
        else:
            rescale = False

        # Get h_factor, e_factor
        if h_factor is None or e_factor is None:
            h_factor, e_factor = sample_augmentation(
                self.gaussian_mean, 
                self.gaussian_std, 
                self.uniform_min,
                self.uniform_max, 
                as_tensor=False
            )
        
        augmented_RGB2HEX = RGB2HEX * [[h_factor], [e_factor], [1]]
        separated_augmented_image = skimage_color.separate_stains(img, augmented_RGB2HEX)
        augmented_image = skimage_color.combine_stains(separated_augmented_image, HEX2RGB)

        if rescale:
            augmented_image = (255 * augmented_image).astype(np.uint8)

        return augmented_image

    def __repr__(self):
        format_string = self.__class__.__name__ + '(gaussian_mean={}, gaussian_std={}'.format(
            self.gaussian_mean,
            self.gaussian_std
        )
        return format_string
