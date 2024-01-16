from glob import glob
from argparse import ArgumentParser
from torchvision.models import resnet50, resnet18, resnext50_32x4d
from torchvision import transforms
from torch.nn import Identity
import torch
import os
import numpy as np
from pkg.wsi_mil.tile_wsi.utils import get_image
from pkg.wsi_mil.utils import get_device

def simple_tiler(slide, path_wsi, name_wsi, outpath, param_tiles, device="cpu", model_path=None):
    """simple_tiler.
    Simply writes tiles as .png

    :param param_tiles: list: output of the patch_sampling.
    """
    for o, para in enumerate(param_tiles):
        patch = get_image(slide=path_wsi, para=para, numpy=False)
        path_tile = os.path.join(outpath['tiles'], f"tile_{o}.png")
        patch.save(path_tile)
        del patch

def _forward_pass_WSI(model, slide, param_tiles, preprocess):
    """_forward_pass_WSI. Feeds a pre-trained model, already loaded, 
    with the extracted tiles.

    :param model: Pytorch loaded module.
    :param param_tiles: list, output of the patch_sampling.
    :param preprocess: torchvision.transforms.Compose.
    """
    device = get_device()
    tiles = []
    for o, para in enumerate(param_tiles):
        image = get_image(slide=slide, para=para, numpy=False)
        image = image.convert("RGB")
        image = preprocess(image).unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            t = model(image).squeeze()
        tiles.append(t.cpu().numpy())
    mat = np.vstack(tiles)
    return mat

def imagenet_tiler(slide, path_wsi, name_wsi, outpath, param_tiles, device="cpu", model_path=None):
    """imagenet_tiler.
    Encodes each tiles thanks to a resnet18 pretrained on Imagenet.
    Embeddings are 512-dimensionnal.

    :param param_tiles: list, output of the patch_sampling.
    """
    model = resnet18(pretrained=True)
    model.fc = Identity()
    model = model.to(device)
    model.eval()
    preprocess = _get_transforms(imagenet=True)
    mat = _forward_pass_WSI(model, slide, param_tiles, preprocess)
    np.save(os.path.join(outpath['tiles'], f'{name_wsi}_embedded.npy'), mat)

def _get_transforms(imagenet=True):
    """_get_transforms.
    For tiling encoding, normalize the input with moments of the
    imagenet distribution or of the training set of the MoCo model.

    :param imagenet: bool: use imagenet pretraining.
    """
    if not imagenet:
        trans = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize([0.747, 0.515,  0.70], [0.145, 0.209, 0.154])])
    else: 
        trans = transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return trans

def ciga_tiler(slide, path_wsi, name_wsi, outpath, param_tiles, device="cpu", model_path=None):
    def load_model_weights(model, weights):
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        return model
    model = resnet18()
    state = torch.load(model_path, map_location='cpu')
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    model = load_model_weights(model, state_dict)
    model.fc = Identity()
    model = model.to(device)
    model.eval()
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    tiles = []
    for o, para in enumerate(param_tiles):
        image = usi.get_image(slide=slide, para=para, numpy=False)
        image = image.convert("RGB")
        # if self.from_0:
        #     image = image.resize(self.size)
        image = preprocess(image).unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            t = model(image).squeeze()
        tiles.append(t.cpu().numpy())
    mat = np.vstack(tiles)
    np.save(os.path.join(outpath, '{}_embedded.npy'.format(name_wsi)), mat)


def moco_tiler(slide, path_wsi, name_wsi, outpath, param_tiles, device="cpu", model_path=None):
    """moco_tiler.
    Encodes each tiles thanks to a resnet18 pretrained with MoCo.

    Code for loading the model taken from the MoCo official package:
    https://github.com/facebookresearch/moco

    :param param_tiles: list, output of the patch_sampling.
    """
    model = resnet18()
    checkpoint = torch.load(model_path, map_location='cpu')
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.fc = Identity()
    model = model.to(device)
    model.eval()
    preprocess = _get_transforms(imagenet=False)
    mat = _forward_pass_WSI(model, param_tiles, preprocess)
    np.save(os.path.join(outpath['tiles'], 'mat', f'{name_wsi}_embedded.npy'), mat)

def simclr_tiler(slide, path_wsi, name_wsi, outpath, param_tiles, device="cpu", model_path=None):
    raise NotImplementedError
