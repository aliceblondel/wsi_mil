import pandas as pd
import pickle
import os
import numpy as np
import openslide
from pkg.wsi_mil.tile_wsi.encoders import *

from .utils import make_auto_mask, patch_sampling, get_size, visualise_cut, get_image

#TODO Ajouter name_slide dans les infos

class ImageTiler:
    """
    Class implementing several possible tiling.
    initialized with the namespace args:
        args    .path_wsi: path to the WSI to tile.
                .path_mask: path to the annotation in xml format. Not necessary 
                    if auto_mask= 1.
                .level: pyramidal level of tile extraction.
                .size: dimension in pixel of the extracted tiles.
                .auto_mask: 0 or 1. If 1, automatically extracts relevant
                    portions of the WSI
                .tiler: type of tiling. available: simple | imagenet | moco
                .path_outputs: str, root of the paths where are stored the outputs.
                .model_path: str, when using moco tiler.
                .mask_tolerance: minimum percentage of mask on a tile for selection.
    """
    def __init__(self, args, make_info=True):
        self.level = args.level # Level to which sample patch.
        self.nf = args.nf # Useful for managing the outputs
        self.device = args.device
        self.size = (args.size, args.size)
        self.path_wsi = args.path_wsi 
        self.max_nb_tiles = args.max_nb_tiles
        self.path_outputs = os.path.join(args.path_outputs, args.tiler, f'level_{args.level}')
        self.auto_mask = args.auto_mask
        self.path_mask = args.path_mask
        self.model_path = args.model_path 
        self.infomat = None
        self.tiler = args.tiler
        self.name_wsi, self.ext_wsi = os.path.splitext(os.path.basename(self.path_wsi))
        # If nf is used, it manages the output paths.
        self.outpath = self._set_out_path()
        self.slide = openslide.open_slide(self.path_wsi)
        self.make_info = make_info
        if args.mask_level < 0:
            self.mask_level = self.slide.level_count + args.mask_level
        else:
            self.mask_level = args.mask_level    
        self.rgb_img = self.slide.get_thumbnail(self.slide.level_dimensions[self.mask_level]) 
        self.rgb_img = np.array(self.rgb_img)[:,:,:3]
        self.mask_tolerance = args.mask_tolerance

    def _set_out_path(self):
        """_set_out_path. Sets the path to store the outputs of the tiling.
        Creates them if they do not exist yet.
        """
        outpath = dict()
        nf = self.nf
        tiler = self.tiler
        outpath['info'] = '.' if nf else os.path.join(self.path_outputs, 'info') 
        outpath['visu'] = '.' if nf else os.path.join(self.path_outputs, 'visu')
        if tiler == 'simple':
            outpath['tiles'] = '.' if nf else os.path.join(self.path_outputs, self.name_wsi)
        else:
            outpath['tiles'] = '.' if nf else os.path.join(self.path_outputs, 'mat')
        #Creates the dirs.
        [os.makedirs(v, exist_ok=True) for k,v in outpath.items()]
        return outpath
 
    def _get_mask_function(self):
        """
        the patch sampling functions need as argument a function that takes a WSI a returns its 
        binary mask, used to tile it. here it is.
        """
        if self.auto_mask:
            mask_function = lambda x: make_auto_mask(x, mask_level=-1)
        else:
            path_mask = os.path.join(self.path_mask, self.name_wsi + ".xml")
            assert os.path.exists(path_mask), "No annotation at the given path_mask"
            mask_function = lambda x: get_polygon(image=self.rgb_img, path_xml=path_mask, label='t')
        return mask_function

    def tile_image(self):
        """tile_image.
        Main function of the class. Tiles the WSI and writes the outputs.
        WSI of origin is specified when initializing TileImage.
        """
        self.mask_function = self._get_mask_function()
        tiler = self.get_tile_encoder()
        param_tiles = patch_sampling(slide=self.slide, mask_level=self.mask_level, mask_function=self.mask_function, 
            analyse_level=self.level, patch_size=self.size, mask_tolerance = self.mask_tolerance)
        if self.make_info:
            self._make_infodocs(param_tiles)
            self._make_visualisations(param_tiles)
        tiler(
            self.slide, 
            self.path_wsi, 
            self.name_wsi, 
            self.outpath, 
            param_tiles, 
            device="mps", 
            model_path=None,
        )

    def _make_visualisations(self, param_tiles):
        """_make_visualisations.
        Creates and save an image showing the locations of the extracted tiles.

        :param param_tiles: list, output of usi.patch_sampling.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.use('Agg')
        PLOT_ARGS = {'color': 'red', 'size': (12, 12),  'with_show': False,
                     'title': "n_tiles={}".format(len(param_tiles))}
        visualise_cut(self.slide, param_tiles, res_to_view=self.mask_level, plot_args=PLOT_ARGS)
        plt.savefig("{}_visu.png".format(os.path.join(self.outpath['visu'], self.name_wsi)))

    def _get_infomat(self):
        """Returns a zero-matrix, such that each entry correspond to a tile in the WSI.
        ID of each tiles will be stored here. 

        Returns
        -------
        tuple
            (mat -ndarray- , size_patch_0 -int, size of a patch in level 0- )
        """
        size_patch_0 = get_size(self.slide, size_from=self.size, level_from=self.level, level_to=0)[0]
        dim_info_mat = (self.slide.level_dimensions[0][0] // size_patch_0, self.slide.level_dimensions[0][1] // size_patch_0)
        info_mat = np.zeros(dim_info_mat)
        return info_mat, size_patch_0 

    def _make_infodocs(self, param_tiles):
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
        infomat , patch_size_0 = self._get_infomat() 
        if self.tiler == 'classifier':
            self.infomat_classif = np.zeros(infomat.shape)
        for o, para in enumerate(param_tiles):
            infos.append({'ID': o, 'x':para[0], 'y':para[1], 'xsize':self.size[0], 'ysize':self.size[0], 'level':para[4]})
            infodict[o] = {'x':para[0], 'y':para[1], 'xsize':self.size[0], 'ysize':self.size[0], 'level':para[4]} 
            infomat[para[0]//(patch_size_0+1), para[1]//(patch_size_0+1)] = o + 1 
            #+1 car 3 lignes plus loin je sauve infomat-1 (background à -1) 
        df = pd.DataFrame(infos)
        self.infomat = infomat - 1 

        # Saving
        df.to_csv(os.path.join(self.outpath['info'], self.name_wsi + '_infos.csv'), index=False)
        np.save(os.path.join(self.outpath['info'], self.name_wsi + '_infomat.npy'), infomat-1)
        with open(os.path.join(self.outpath['info'],  self.name_wsi + '_infodict.pickle'), 'wb') as f:
            pickle.dump(infodict, f)
   
    def get_tile_encoder(self):
        if self.tiler == "ctranspath":
            return ctranspath_tiler
        elif self.tiler == "simple": 
            return simple_tiler
        elif self.tiler == "imagenet": 
            return imagenet_tiler
        elif self.tiler == "ciga": 
            return ciga_tiler
        elif self.tiler == "moco": 
            return moco_tiler
        elif self.tiler == "simclr": 
            return simclr_tiler
        else:
            raise NotImplementedError