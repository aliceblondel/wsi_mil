
"""
Inspired by  https://github.com/trislaz/Democratizing_WSI.
"""
import os
import json
import cv2
import random
import itertools
import math
from pathlib import Path
import torch
import pandas as pd
import pickle
from PIL import Image

import numpy as np
import openslide
from PIL import ImageDraw
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, square
from torchvision import transforms

from pkg.wsi_mil.tile_wsi.dataset.dataset import Dataset
from ..utils import visualise_cut



class SlideTileDataset(Dataset):

    mag_level0 = 40 
    ds_per_level = 2
    final_tile_size = 224

    def __init__(
            self,
            id,
            dataframe,
            data_folder, 
            save_folder,
            magnification_tile=10, 
            max_tiles_per_slide=None,
            mask_tolerance = 0.9,
            transform=None,
            is_train=False,
            mean=None,
            std=None,
            make_info = True,
            save_tiles_img = True,
        ):

        super().__init__(
            dataframe=dataframe, 
            data_folder=data_folder, 
            transform=transform, 
            is_train = is_train, 
            mean = mean, 
            std = std,
        )

        self.id = id
        self.max_tiles_per_slide = max_tiles_per_slide
        self.magnification_tile = magnification_tile
        self.level_tile = self._get_level()
        self.mask_level = -1
        self.mask_tolerance = mask_tolerance
        self._set_out_path(save_folder)
        
        self.make_info = make_info
        self.save_tiles_img = save_tiles_img
        self.dataframe = dataframe[dataframe["zone_id"]==id]
        contours_dict = self.dataframe[["slide_id", "id"]].groupby('slide_id')['id'].apply(list).to_dict()
        self.tile_coords = self._prepare_tile_coords(contours_dict)
        

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, index):
        row, col = self.tile_coords[index]["coord"][:2]
        slide_path = self.tile_coords[index]["slide_path"]
        tile = self._load_img(index)

        if self.save_tiles_img: # TO DO
            tile_img = Image.fromarray(tile, 'RGB')
            slide_id = Path(slide_path).stem
            tiles_folder = self.visu_folder / f'{self.id}__{slide_id}'
            tiles_folder.mkdir(exist_ok=True, parents=True)
            tile_img.save( tiles_folder / f'tile_{index}.png')

        if self.transform:
            tile = self.transform(image=tile)["image"]

        return tile, torch.Tensor(np.array([col, row]))
    
    def _load_img(self, index):
        slide = self.tile_coords[index]["slide"]
        coord = self.tile_coords[index]["coord"]
        row, col, h, w = coord[:4]
    
        img = slide.read_region(
            location=(col, row), 
            level=self.level_tile, 
            size=(min(self.final_tile_size, slide.dimensions[0] - col), 
                  min(self.final_tile_size, slide.dimensions[1] - row))
        )
        img = np.asarray(img.convert("RGB"))

        return img

    def _get_thumbnail(self, slide):
        thumbnail = slide.get_thumbnail(
            slide.level_dimensions[self.mask_level]
        )
        return thumbnail
 
    def _make_masked_thumbnail(self, tile_coords, slide, thumbnail, slide_id):
        thumbnail_file_name = f'{self.zone_id}__{slide_id}_masked_thumbnail.png'
        save_masked_thumbnail = self.visu_folder / thumbnail_file_name
        draw = ImageDraw.Draw(thumbnail)
        ds = slide.level_downsamples[self.mask_level]
        for index, tile_info in enumerate(tile_coords):
            row, col, h, w = tile_info["coord"][:4]
            scaled_row, scaled_col = row // ds, col // ds
            scaled_h, scaled_w = h // ds, w // ds
            draw.rectangle(
                (scaled_col, scaled_row, scaled_col + scaled_w, scaled_row + scaled_h),
                outline='red',
            )
            draw.text((scaled_col, scaled_row), str(index), fill='red')
        thumbnail.save(save_masked_thumbnail)


    def _prepare_tile_coords(self, contours_dict):
        tile_coords = []
        for slide_id, contour_ids in contours_dict.items():
            
            slide_path = os.path.join(self.data_folder, "Slides", slide_id + ".ndpi")
            geojson_filepath = os.path.join(self.data_folder, "Annotations", slide_id + ".geojson")
            
            slide_tile_coords = self._prepare_slide_tile_coord(slide_id, slide_path, geojson_filepath, contour_ids)
            
            tile_coords.extend(slide_tile_coords)

        if isinstance(self.max_tiles_per_slide, int):
            indices = np.random.choice(
                len(tile_coords), 
                min(self.max_tiles_per_slide, len(tile_coords)), 
                replace=False,
            )
            tile_coords = [tile_coords[i] for i in indices]
        else:
            random.shuffle(tile_coords)
            
        return tile_coords

    def _set_out_path(self, save_folder):
        """_set_out_path. 
        Sets the path to store the outputs of the tiling.
        Creates them if they do not exist yet.
        """        
        level_folder_name = "level_" + str(self.level_tile)
        self.save_folder = Path(save_folder) / level_folder_name
        self.save_folder.mkdir(exist_ok=True, parents=True)

        self.info_folder = self.save_folder / 'info'
        self.info_folder.mkdir(exist_ok=True, parents=True)

        self.visu_folder = self.save_folder / 'visu'
        self.visu_folder.mkdir(exist_ok=True, parents=True) 

    
    def _prepare_slide_tile_coord(self, slide_id, slide_path, geojson_filepath, contour_ids):

        slide = openslide.open_slide(slide_path)
        thumbnail = self._get_thumbnail(slide)
        zone_coordinates = self._get_contour_coord(geojson_filepath, contour_ids)
        dico = self._get_clean_grid(slide, thumbnail, zone_coordinates)

        slide_tile_coords = [
            {
                "slide_path": slide_path, 
                "slide": slide,
                "coord": coord,
            } 
            for coord in dico['tile_coords']
        ]

        if self.make_info:            
            
            self._make_masked_thumbnail(slide_tile_coords, slide, thumbnail, slide_id)
            self._make_infodocs(slide, slide_tile_coords, slide_id)
            self._make_visualisations(slide, slide_tile_coords, slide_id)

        return slide_tile_coords

    def _get_level(self):
        level = math.log(self.mag_level0 / self.magnification_tile, self.ds_per_level)
        return int(level)

    def _get_clean_grid(self, slide, thumbnail, zone_coordinates=None):
        slide_height, slide_width = slide.dimensions[1], slide.dimensions[0]
        mask_ds = int(slide.level_downsamples[self.mask_level])
        mask = self._make_auto_mask(thumbnail)
        if zone_coordinates is not None:
            zone_mask = self._make_zone_mask(zone_coordinates, mask.shape, mask_ds)    
            mask = mask * zone_mask
        size_at_0 = self.final_tile_size * (2 ** self.level_tile)
        grid = self._grid_blob((0, 0), (slide_height, slide_width), (size_at_0, size_at_0))
        grid = [
            (x[0], x[1], size_at_0, size_at_0, 0)
            for x in grid 
            if self._check_coordinates(
                x[0], x[1], 
                (size_at_0, size_at_0), 
                mask, 
                mask_ds, 
            )
        ]
        dico = {'tile_coords': grid, 'mask': mask}
        return dico 

    def _make_auto_mask(self, thumbnail):
        """
        Create a binary mask from a downsampled version of a WSI. 
        Uses the Otsu algorithm and a morphological opening.
        """
        im = np.array(thumbnail)[:, :, :3]
        im_gray = rgb2gray(im)
        #im_gray = self._clear_border(im_gray, prop=10)
        size = im_gray.shape
        im_gray = im_gray.flatten()
        pixels_int = im_gray[np.logical_and(im_gray > 0.02, im_gray < 0.98)]
        t = threshold_otsu(pixels_int)
        mask = opening(closing((im_gray < t).reshape(size), square(2)), square(2))
        return mask
    
    def _make_zone_mask(self, zone_coordinates, mask_shape, mask_downsample):
        
        mask = np.zeros(mask_shape)
        for coord in zone_coordinates:
            ds_coord = coord//mask_downsample 
            cv2.drawContours(mask, [ds_coord.astype(np.int32)], 0, (1), -1)

        return mask.astype(bool)

    @staticmethod
    def _clear_border(mask, prop):
        r, c = mask.shape
        pr, pc = r // prop, c // prop
        mask[:pr, :] = 0
        mask[r - pr :, :] = 0
        mask[:, :pc] = 0
        mask[:, c - pc :] = 0
        return mask

    def _grid_blob(self, point_start, point_end, patch_size):
        """
        Forms a uniform grid starting from the top left point point_start
        and finishes at point point_end of size patch_size for the given slide.
        Args:
            point_start : Tuple like object of integers of size 2.
            point_end : Tuple like object of integers of size 2. (x,y) = (row, col) = (height, width)
            patch_size : Tuple like object of integers of size 2.
        Returns:
            List of coordinates of grid.
        """
        patch_size_0 = patch_size
        size_x, size_y = patch_size_0
        list_col = range(point_start[1], point_end[1], size_x)
        list_row = range(point_start[0], point_end[0], size_y)
        return list(itertools.product(list_row, list_col))

    def _check_coordinates(self, row, col, patch_size, mask, mask_downsample):
        """
        Checks if the patch at coordinates x, y in res 0 is valid.
        Args:
            row : Integer. row coordinate of the patch.
            col : Integer. col coordinate of the patch.
            patch_size : Tuple of integers. Size of the patch.
            mask : Numpy array. Mask of the slide.
            mask_downsample : Integer. Resolution of the mask.
        Returns:
            Boolean. True if the patch is valid, False otherwise.
        """
        col_0, row_0 = col, row
        col_1, row_1 = col + patch_size[0], row + patch_size[1]
        ## Convert coordinates to mask_downsample resolution
        col_0, row_0 = col_0 // mask_downsample, row_0 // mask_downsample
        col_1, row_1 = col_1 // mask_downsample, row_1 // mask_downsample
        if col_0 < 0 or row_0 < 0 or row_1 > mask.shape[0] or col_1 > mask.shape[1]:
            return False
        mask_patch = mask[row_0:row_1, col_0:col_1]
        if mask_patch.sum() <= self.mask_tolerance * np.ones(mask_patch.shape).sum():
            return False
        return True


    def _get_contour_coord(self, geojson_filepath, contour_ids):

        with open(geojson_filepath) as f:
            dict_annotations = json.load(f)

        contour_coordinates= []
        for dict_contour in dict_annotations["features"]:
            coord = dict_contour['geometry']['coordinates']
            assert len(coord) == 1
            if dict_contour['id'] in contour_ids:
                contour_coordinates.append(np.array(coord[0]))
        
        assert len(contour_ids) == len(contour_coordinates)

        return contour_coordinates
        
    def _make_infodocs(self, slide, slide_tile_coords, slide_id):
        """_make_infodocs.
        Creates the files containing the information relative 
        to the extracted tiles.
        infos are saved in the outpath 'info'.
        *   infodict : same as param_tiles, but dictionnary version (needs pickle 
            load them). Stores the ID of the tile, the x and y position in the 
            original WSI, the size in pixels at the desired level of extraction and 
            finally the level of extraction.
        *   infodf : same as infodict, stored as a pandas DataFrame.
        *   infomat : relates the tiles to their position on the WSI.
            matrix of size (n_tiles_H, n_tiles_W), each cell correspond to a tile
            and is fill with -1 if the tile is background else with the tile ID.

        :param param_tiles: list: output of the patch_sampling.
        """
        infodict = {}
        infos=[]
        patch_size_0 = self.final_tile_size * (2 ** self.level_tile)
        dim_info_mat = (
            slide.level_dimensions[0][0] // patch_size_0, 
            slide.level_dimensions[0][1] // patch_size_0
        )
        infomat = np.zeros(dim_info_mat) - 1 

        for o, para in enumerate(slide_tile_coords):
            row, col, _, _, _ = para["coord"]
            size = self.final_tile_size
            level = self.level_tile

            infos.append({'ID': o, 'x':col, 'y':row, 'xsize':size, 'ysize':size, 'level':level})
            infodict[o] = {'x':col, 'y':row, 'xsize':size, 'ysize':size, 'level':level} 
            infomat[col//(patch_size_0+1), row//(patch_size_0+1)] = o
        
        df = pd.DataFrame(infos)
        df.to_csv(os.path.join(self.info_folder, self.id + "__" + slide_id + '_infos.csv'), index=False)
        np.save(os.path.join(self.info_folder, self.id + "__" + slide_id + '_infomat.npy'), infomat)
        with open(os.path.join(self.info_folder,  self.id + "__" + slide_id + '_infodict.pickle'), 'wb') as f:
            pickle.dump(infodict, f)

    def _make_visualisations(self, slide, slide_tile_coords, slide_id):
        """_make_visualisations.
        Creates and save an image showing the locations of the extracted tiles.

        :param param_tiles: list, output of usi.patch_sampling.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.use('Agg')

        size = self.final_tile_size
        level = self.level_tile
        param_tiles = []
        for o, para in enumerate(slide_tile_coords):
            row, col, _, _, _ = para["coord"]
            param_tiles.append([col, row, size, size,level])

        res_to_view = slide.level_count + self.mask_level
        PLOT_ARGS = {'color': 'red', 'size': (12, 12),  'with_show': False,
                     'title': "n_tiles={}".format(len(slide_tile_coords))}
        
        visualise_cut(slide, param_tiles, res_to_view=res_to_view, plot_args=PLOT_ARGS)
        plt.savefig("{}_visu.png".format(os.path.join(self.visu_folder, self.id+"__"+slide_id)))

def st_collate_fn(samples):
    imgs = torch.concat([torch.Tensor(sample[0]).unsqueeze(0) for sample in samples])
    ids = torch.concat([torch.Tensor(sample[1]).unsqueeze(0) for sample in samples])
    return imgs.permute(0, 3, 1, 2), ids
