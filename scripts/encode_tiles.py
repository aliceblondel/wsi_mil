import os 
os.environ["USE_TRANSPATH"] = "True"
print(os.environ["USE_TRANSPATH"])

import pandas as pd
import numpy as np
import torch
from pkg.wsi_mil.tile_wsi.dataset.slide_tile_dataset import SlideTileDataset, st_collate_fn
from torch.utils.data import DataLoader
from pkg.wsi_mil.tile_wsi.encoders import CTranspathModel

def encode_tiles(
        model, 
        zone_id,
        dataframe,
        data_folder,
        save_folder,
        magnification_tile=10, 
        max_tiles_per_slide=None,
        device='cpu', 
        mask_tolerance=0.9,
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
        save_tiles_img=False,
    ):

    data = SlideTileDataset(
        zone_id, 
        dataframe=dataframe, 
        data_folder=data_folder,
        magnification_tile=magnification_tile, 
        max_tiles_per_slide=max_tiles_per_slide, 
        mask_tolerance=mask_tolerance,
        save_tiles_img=save_tiles_img,
        save_folder=save_folder,
        transform = None,  
        mean = mean,
        std = std, 
    )

    if len(data) == 0:
        return None, None
    dataloader = DataLoader(
        data, 
        batch_size= min(100, len(data)), 
        shuffle=False, 
        collate_fn = st_collate_fn,
    )

    embedding_path = data.save_folder / "mat" 
    pca_embedding_path = data.save_folder / "mat_pca"
    embedding_path.mkdir(exist_ok=True)
    pca_embedding_path.mkdir(exist_ok=True)

    embeddings, pca_embeddings = [], []
    xys = []
    for batch in dataloader:
        with torch.no_grad():
            im, xy = batch
            im = im.to(device)
            xys.append(xy)
            emb = model(im)
            pca_emb = model.apply_pca(emb)
            embeddings.append(emb)
            pca_embeddings.append(pca_emb)
    embeddings = np.concatenate(embeddings)
    pca_embeddings = np.concatenate(pca_embeddings)
    xys = torch.concatenate(xys)
    embedding_name = zone_id + ".npy"
    np.save(embedding_path/embedding_name, embeddings)
    np.save(pca_embedding_path/embedding_name, pca_embeddings)
    
    del dataloader, data
    del batch
    return embeddings, xys, data.save_folder 


if __name__=="__main__":

    import argparse
    from pkg.wsi_mil.tile_wsi.pca_partial import pca_savings

    parser = argparse.ArgumentParser(description='Script Description')
    parser.add_argument('--device', type=str, default = "cpu", help='Device')
    parser.add_argument('--gt_filepath', type=str, help='GT file path')
    parser.add_argument('--data_folder', type=str, help='Data folder')
    parser.add_argument('--embedding_folder', type=str, help='Tile Embedding folder')
    parser.add_argument('--save_tiles_img', type=bool, default = False, help='Save Tile Images during Tiling')
    args = parser.parse_args()

    df = pd.read_excel(args.gt_filepath, index_col=0)
    model = CTranspathModel().to(args.device)

    model.eval()
    with torch.no_grad():
        for zone_id in df["zone_id"].unique():
            print(zone_id)
            tile_embeddings, tile_xys, save_folder = encode_tiles(
                model, 
                zone_id,
                df,
                args.data_folder,
                args.embedding_folder,
                magnification_tile=10, 
                max_tiles_per_slide=None,
                device=args.device, 
                mask_tolerance=0.9,
                save_tiles_img=args.save_tiles_img,
            )
    
    pca_path = save_folder / "pca" 
    pca_path.mkdir(exist_ok=True)
    pca_savings(model.pca, pca_path)