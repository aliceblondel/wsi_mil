import os
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as dataset
from pkg.wsi_mil.tile_wsi.dataset.augmentation import HEStainAugmentation


class Dataset(dataset):
    def __init__(
            self, 
            dataframe, 
            data_folder, 
            transform = None,
            is_train = False,
            mean = None,
            std = None,
            pad_mode = False, 
            height = None,
            width = None,
        ) -> None:
        self.dataframe = dataframe
        self.data_folder = data_folder
        self.slides_folder = os.path.join(data_folder, "Slides")
        self.annotation_folder = os.path.join(data_folder, "Annotations")
        self.is_train = is_train
        self.transform = transform
        self.pad_mode = pad_mode
        self.height = height
        self.width = width
        if (mean is not None) and (std is not None) and not transform:
            self.set_transform(mean, std)
    
    def _load_img(self, index):
        pass

    def set_transform(self, mean=0, std=1):
        self.mean = mean
        self.std = std
        transform_list=[]
        if self.is_train:
            transform_list.extend([
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=360, 
                    border_mode=0,  
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.GridDistortion(
                    num_steps=5, 
                    distort_limit=0.05, 
                    border_mode=0, 
                    p=1
                ),
                A.Lambda(
                    name="HEStainAugmentation",
                    image=HEStainAugmentation(gaussian_mean=1, gaussian_std=0.3), 
                    mask=None,
                ),
            ])
        if self.pad_mode:
            transform_list.insert(0, A.PadIfNeeded(
                min_height=self.height, min_width=self.width, border_mode=0, value=0, p=1))
            transform_list.append(A.CenterCrop(height=self.height, width=self.width, p=1))

        transform_list.append(
            A.Normalize(
                mean=self.mean,
                std=self.std,
                max_pixel_value=1,
            )
        )

        self.transform = A.Compose(transform_list)


    def plot_transforms(self, index):
        n = len(self.transform)
        img = self._load_img(index)

        plt.figure(figsize=(5*n, 5))
        plt.subplot(1, n+1, 1)
        plt.imshow(img)
        plt.title("Image originale")

        for i, t in enumerate(self.transform):
            etape_name = t.__class__.__name__ if t.__class__.__name__!="Lambda" else t.name
            img = t(image=np.array(img))["image"]

            plt.subplot(1, n+1, i + 2)
            plt.imshow(img)
            plt.title(f"Étape {i + 1}: {etape_name}")

        plt.show()
    