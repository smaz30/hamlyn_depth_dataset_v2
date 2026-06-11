import argparse
from pathlib import Path
from natsort import natsorted
import numpy as np
import cv2
from PIL import Image

from src import load_intrinsics, load_extrinsics, compute_depth_map_raft

def main(
        data_folder : str,
        calibration_folder : str,
        vis : bool,
         ):
    hamlyn_path = Path(data_folder)
    calibraion_path = Path(calibration_folder)

    list_of_data = natsorted(hamlyn_path.glob('*/'))
    
    for folder in list_of_data:
        id = folder.name[-2:]
        intrinsics_p = str(calibraion_path / id ) + '/' + 'intrinsics.txt'
        extrinsics_p = str(calibraion_path / id ) + '/' + 'extrinsics.txt'
        folder_mask = folder / 'depth_mask'
        folder_mask.mkdir(exist_ok=True, parents=True)
        folder_depth_psudo = folder / 'depth_psudo'
        folder_depth_psudo.mkdir(exist_ok=True, parents=True)
        K = load_intrinsics(intrinsics_p)
        R, T = load_extrinsics(extrinsics_p)

        img_left_path = folder / 'image01'
        img_right_path = folder / 'image02'

        print(f"Processing Sequence {folder.name}")
        for i, (im_l_p, im_r_p) in enumerate(zip(natsorted(img_left_path.glob('*.jpg')), natsorted(img_right_path.glob('*.jpg')))):
            im_left = cv2.imread(str(im_l_p))
            im_right = cv2.imread(str(im_r_p))

            image_left_rgb= cv2.cvtColor(im_left, cv2.COLOR_BGR2RGB)
            image_right_rgb= cv2.cvtColor(im_right, cv2.COLOR_BGR2RGB)
            H,W,C = im_right.shape

            depth, mask, disp = compute_depth_map_raft(image_left_rgb, image_right_rgb, K, np.linalg.norm(T),W, H )
            # Avoid Black borders
            zer_mask = (im_left[:,:,0]>0)*1.0
            mask = mask*zer_mask

            # NOTE: normalization is done to have the same color mapping with a maximum depth of 250mm to compare visually with the original depth map.
            depth_normalized =255 *(depth * mask) / 250 

            depth_normalized = depth_normalized.astype(np.uint8)

            depth_colored = cv2.applyColorMap(
                depth_normalized,
                cv2.COLORMAP_INFERNO
            )
            mask_to_show = cv2.applyColorMap(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),cv2.COLORMAP_INFERNO)

            # At least half of the depth should be available
            if mask.sum()>(H*W*0.5):
                depth_file_name = im_l_p.stem + '.npy'
                mask_file_name = im_l_p.stem + '.npy'
                np.save(str(folder_depth_psudo / depth_file_name), depth.astype(np.float32))
                np.save(str(folder_mask / mask_file_name), mask.astype(np.utint8))
            
            if vis:
                d1 = np.array(Image.open(fr"{im_l_p.parent.parent / 'depth01' }\{im_l_p.stem}.png"))
                d1_n = cv2.applyColorMap((d1 /250 * 255).astype(np.uint8),cv2.COLORMAP_INFERNO)
                cv2.imshow('left', im_left)
                cv2.imshow('right', im_right)
                if int(id) >13:
                    cv2.imshow('depth',depth_colored[:,180:590])
                    cv2.imshow('mask',mask_to_show[:,180:590])
                    cv2.imshow('d1_n',d1_n[:,180:590])
                else:
                    cv2.imshow('depth',depth_colored)
                    cv2.imshow('mask',mask_to_show)
                    cv2.imshow('d1_n',d1_n)
                cv2.waitKey(0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo ground truth depth maps from stereo images.")

    parser.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="Path to the data folder"
    )

    parser.add_argument(
        "--calibration-folder",
        type=str,
        required=True,
        help="Path to the calibration folder"
    )
    parser.add_argument(
        "--visualization",
        action="store_true",
        help="Enable visualization"
    )

    args = parser.parse_args()
    main(
        data_folder=args.data_folder,
        calibration_folder=args.calibration_folder,
        vis = args.visualization
        )
