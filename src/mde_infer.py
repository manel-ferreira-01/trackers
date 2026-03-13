import sys 
import torch
import numpy as np
from tqdm import tqdm

_MODEL_CACHE = globals().get("_MODEL_CACHE", {})

def infer_mde(video_tensor, mde_model="moge"):

    match mde_model:
        case "vggt":
            from auxiliar.call_vggt import call_vggt
            from aux import wrap_value, unwrap_value
            import io

            response = call_vggt(video_tensor.cpu(), device="cuda:0")
            depth_infered = torch.tensor(np.load(io.BytesIO(unwrap_value(response.data["depth"]))))
            instrisics_infered = torch.tensor(np.load(io.BytesIO(unwrap_value(response.data["intrinsic"]))))

        case "unidepth":
            if "unidepth" not in _MODEL_CACHE:
                sys.path.append("/home/manuelf/UniDepth/")
                from unidepth.models import UniDepthV2
                _MODEL_CACHE["unidepth"] = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to("cuda:0")
            
            model = _MODEL_CACHE["unidepth"]
            depths, uni_conf, uni_focal = [], [], []
            for frame in tqdm(range(video_tensor.shape[1])):
                pred = model.infer(video_tensor[0, frame].permute(2, 0, 1).to("cuda:0"))
                depths.append(pred["depth"])
                uni_conf.append(1 / pred["confidence"])
                uni_focal.append(pred["intrinsics"])

            instrisics_infered = torch.stack(uni_focal).cpu().squeeze()
            depth_infered = torch.stack(depths).unsqueeze(-1).cpu().squeeze()

        case "d-any-v2":
            if "d-any-v2" not in _MODEL_CACHE:
                sys.path.append("/home/manuelf/Depth-Anything-V2")
                from depth_anything_v2.dpt import DepthAnythingV2
                encoder = 'vitl'
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                m = DepthAnythingV2(**model_configs[encoder])
                m.load_state_dict(torch.load(f'/home/manuelf/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
                _MODEL_CACHE["d-any-v2"] = m.to("cuda:0").eval()

            model = _MODEL_CACHE["d-any-v2"]
            depths = []
            for frame in tqdm(range(video_tensor.shape[1])):
                depth = model.infer_image(video_tensor[0, frame].to("cuda:0").cpu().numpy())
                depths.append(torch.tensor(depth))

            depth_infered = 1 / torch.clip(torch.stack(depths).unsqueeze(-1).cpu().squeeze(), 100)

        case "moge":
            if "moge" not in _MODEL_CACHE:
                sys.path.append("/home/manuelf/MoGe")
                from moge.model.v2 import MoGeModel
                _MODEL_CACHE["moge"] = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda:0")

            model = _MODEL_CACHE["moge"]
            depths, instrisics = [], []
            for frame in (range(video_tensor.shape[1])):
                pred = model.infer((video_tensor[0, frame].permute(2, 0, 1).to("cuda:0") + 1) / 2.0)
                depths.append(pred["depth"])
                instrisics.append(pred["intrinsics"])

            depth_infered = torch.stack(depths).unsqueeze(-1).cpu().squeeze()
            instrisics_infered = torch.stack(instrisics).cpu().squeeze()
            instrisics_infered[:, 1, :] *= video_tensor.shape[2]
            instrisics_infered[:, 0, :] *= video_tensor.shape[3]

    torch.cuda.empty_cache()
    return depth_infered, instrisics_infered


def clear_model_cache(mde_model=None):
    """Free cached models. Pass a model name to clear just one, or None to clear all."""
    if mde_model:
        _MODEL_CACHE.pop(mde_model, None)
    else:
        _MODEL_CACHE.clear()