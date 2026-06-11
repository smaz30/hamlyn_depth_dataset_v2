import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import numpy as np

from .optical_flow import OpticalFlow
def preprocess(batch, W_pred = 640, H_pred= 512):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  
            T.Resize(size=(int(H_pred), int(W_pred)), interpolation=T.InterpolationMode.BILINEAR)
        ]
    )
    batch = transforms(batch)
    return batch

device = "cuda" if torch.cuda.is_available() else "cpu"

optical_flow = OpticalFlow(mask_mode="flow").cuda()


def compute_depth_map_raft(im_left, im_right, K ,baseline, W, H):
    tensor_im_1 = torch.from_numpy(im_left).permute(2, 0, 1).unsqueeze(0) 
    tensor_im_2 = torch.from_numpy(im_right).permute(2, 0, 1).unsqueeze(0) 

    img1_batch = preprocess(tensor_im_1, W, H).to(device)
    img2_batch = preprocess(tensor_im_2, W, H).to(device)
    with torch.no_grad():
        flow, mask = optical_flow.get_flows(img1_batch, img2_batch, get_mask=True)
    DISPARITY = flow[0,0:1,:,:].squeeze().detach().cpu().numpy()

    depth = (K[0,0] * baseline) / (-DISPARITY)
    DISPARITY = -DISPARITY

    return depth, mask[0][0].cpu().numpy().astype(np.uint8), DISPARITY