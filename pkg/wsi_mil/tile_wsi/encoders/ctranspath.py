import os
import requests
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
if os.environ['USE_TRANSPATH'] == 'True':
    from timm.models.layers.helpers import to_2tuple
    import timm
from pkg.wsi_mil.tile_wsi.utils import get_image
from pkg.wsi_mil.utils import get_device

def download_ctranspath_weights():
    model_root = Path('./model_weights/')
    model_root.mkdir(parents=True, exist_ok=True)
    model_path = model_root / 'ctranspath.pth'
    if model_path.exists():
        print(f'{str(model_path)} already exists')
        return str(model_path.resolve())
    else:
        print(f'Downloading {str(model_path)}')
        os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX" -O {str(model_path.resolve())} && rm -rf /tmp/cookies.txtOD""")
        return str(model_path.resolve())

def download_pca_ctranspath_weights():
    model_root = Path('../model_weights/')
    model_root.mkdir(parents=True, exist_ok=True)
    model_path = model_root / 'pca-ctranspath.npy'
    if model_path.exists():
        print(f'{str(model_path)} already exists')
        return str(model_path.resolve())
    else:
        print(f'Downloading {str(model_path)}')
        response = requests.get('https://data.mendeley.com/public-files/datasets/d573xfd9fg/files/b2301b46-433e-4028-aba1-853c71739638/file_downloaded')
        response.raise_for_status()  # Will raise an error for a bad status code
        with open(model_path, 'wb') as file:
            file.write(response.content)
        return str(model_path.resolve())
    
def load_transpath_model(transpath_weights_path):
    """
    Code taken from https://github.com/Xiyue-Wang/TransPath
    """
    model = timm.create_model(
        'swin_tiny_patch4_window7_224', 
        embed_layer=ConvStem, 
        pretrained=False
    )
    model.head = nn.Identity()
    state_dict = torch.load(transpath_weights_path)['model']
    model.load_state_dict(state_dict, strict=True)
    return model


class ConvStem(nn.Module):
    """
    Code taken from https://github.com/Xiyue-Wang/TransPath
    """
    def __init__(self, 
            img_size=224, 
            patch_size=4, 
            in_chans=3, 
            embed_dim=768, 
            norm_layer=None, 
            flatten=True, 
            output_fmt='NHWC',
        ):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class CTranspathModel(nn.Module):
    """
    Wraps a model so that forward it outputs the right embeddings,
    depending on the type of model
    """
    
    def __init__(self,):
        
        super(CTranspathModel, self).__init__()
        
        model_weights_path = download_ctranspath_weights()
        model = load_transpath_model(model_weights_path)
        self.model = model
        
        pca = download_pca_ctranspath_weights()
        pca = np.load(pca, allow_pickle=True).item()
        self.pca = pca

    def apply_pca(self, embeddings, n_dims=256):
        return self.pca.transform(embeddings)[:,:n_dims]

    def forward(self, x):
        embeddings = self.model(x)
        embeddings = embeddings.squeeze().detach().cpu().numpy()
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1,-1)
        return embeddings

def ctranspath_tiler(slide, path_wsi, name_wsi, outpath, param_tiles, device="cpu", model_path=None, apply_pca=True):
    device = get_device()
    model = CTranspathModel(apply_pca=apply_pca).to(device)
    tiles = []
    for o, para in enumerate(param_tiles):
        image = get_image(slide=slide, para=para, numpy=False)
        image = image.convert("RGB")
        # image = preprocess(image).unsqueeze(0)
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            t = model(image).squeeze()
        tiles.append(t.cpu().numpy())
    mat = np.vstack(tiles)    
    np.save(os.path.join(outpath['tiles'], f'{name_wsi}_embedded.npy'), mat)
