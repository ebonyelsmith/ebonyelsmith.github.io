import os
import time
import numpy as np
import skimage as sk
import skimage.io as skio
import cv2
import argparse
from utils import log_alignment, align_exhaustive, align_exhaustive_vectorize, align_multiscale, automatic_crop, automatic_contrast_adjustment, automatic_white_balance

def preprocess_image(im):
    # compute height of each channel
    height = int(np.floor(im.shape[0] / 3))

    # separate color channels
    b = im[:height]
    g = im[height:2*height]
    r = im[2*height:3*height]

    # backup original channels for cropping later
    backup_r = r.copy()
    backup_g = g.copy()
    backup_b = b.copy()

    # crop each channel to avoid alignment issues from borders
    crop_percent = 0.15
    crop_h = int(np.floor(crop_percent * height))
    crop_w = int(np.floor(crop_percent * im.shape[1]))
    b = b[crop_h: -crop_h, crop_w: -crop_w]
    g = g[crop_h: -crop_h, crop_w: -crop_w] 
    r = r[crop_h: -crop_h, crop_w: -crop_w]

    return r, g, b, backup_r, backup_g, backup_b



def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Align and crop a color image composed of three vertically stacked rgb channels.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('--method', type=str, default='multiscale', choices=['exhaustive', 'multiscale'], help='Alignment method to use: exhaustive or multiscale (default: multiscale).')
    parser.add_argument('--type', type=str, default='color', choices=['color', 'edge'], help='Type of alignment to perform: color or edge (default: color).')
    args = parser.parse_args()

    # load image
    im = skio.imread(args.image_path)
    im = sk.img_as_float(im)

    # preprocess image
    r, g, b, backup_r, backup_g, backup_b = preprocess_image(im)

    # align channels
    if args.method == 'exhaustive':
        print("Using exhaustive alignment...")
        start_time = time.time()
        dx_ar, dy_ar, _ = align_exhaustive(b, r, search_radius=15, metric='ncc')
        dx_ag, dy_ag, _ = align_exhaustive(b, g, search_radius=15, metric='ncc')
        elapsed_time = time.time() - start_time
    elif args.method == 'multiscale':
        print("Using multiscale alignment...")
        start_time = time.time()
        dx_ar, dy_ar = align_multiscale(b, r, search_radius=15, levels=5, metric='ncc', method= args.type)
        dx_ag, dy_ag = align_multiscale(b, g, search_radius=15, levels=5, metric='ncc', method= args.type)
        elapsed_time = time.time() - start_time

    print(f"Alignment took {elapsed_time:.2f} seconds.")
    ar = np.roll(backup_r, shift=(dy_ar, dx_ar), axis=(0, 1))
    ag = np.roll(backup_g, shift=(dy_ag, dx_ag), axis=(0, 1))
    im_out = np.dstack([ar, ag, backup_b])
    print(f"Red channel shift: dx={dx_ar}, dy={dy_ar}")
    print(f"Green channel shift: dx={dx_ag}, dy={dy_ag}")

    # save image before cropping
    output_folder2 = 'output_images'

    # save output image
    im_out_to_save = sk.img_as_ubyte(im_out)
    filename_without_ext = os.path.splitext(os.path.basename(args.image_path))[0]
    output_folder = os.path.join(output_folder2, filename_without_ext)
    os.makedirs(os.path.join(os.path.dirname(__file__), '../', output_folder), exist_ok=True)
    skio.imsave(os.path.join(os.path.dirname(__file__), '../', output_folder, f'{args.type}_aligned_{filename_without_ext}.png'), im_out_to_save)
    print(f"Saved aligned image to {os.path.join(output_folder, f'{args.type}_aligned_{filename_without_ext}.png')}")
    log_alignment(f"../{output_folder2}",filename_without_ext, dx_ar, dy_ar, dx_ag, dy_ag, elapsed_time)

    # cropping after alignment
    r_cropped, g_cropped, b_cropped = automatic_crop(backup_r, backup_g, backup_b, dx_ar, dy_ar, dx_ag, dy_ag, border_percent=0.1, threshold=0.5, metric='squared_error')
    im_out_cropped = np.dstack([r_cropped, g_cropped, b_cropped])

    # save cropped image
    im_out_cropped_to_save = sk.img_as_ubyte(im_out_cropped)
    skio.imsave(os.path.join(os.path.dirname(__file__), '../', output_folder, f'{args.type}_cropped_{filename_without_ext}.png'), im_out_cropped_to_save)
    print(f"Saved cropped image to {os.path.join(output_folder, f'{args.type}_cropped_{filename_without_ext}.png')}")

    # automatic contrast adjustment
    r_contrast, g_contrast, b_contrast = automatic_contrast_adjustment(r_cropped, g_cropped, b_cropped, method='hist_eq_luminance')
    im_out_contrast = np.dstack([r_contrast, g_contrast, b_contrast])

    # save contrast adjusted image
    im_out_contrast_to_save = sk.img_as_ubyte(im_out_contrast)
    skio.imsave(os.path.join(os.path.dirname(__file__), '../', output_folder, f'{args.type}_contrast_{filename_without_ext}.png'), im_out_contrast_to_save)
    print(f"Saved contrast adjusted image to {os.path.join(output_folder, f'{args.type}_contrast_{filename_without_ext}.png')}")

    # automatic white balance
    r_wb, g_wb, b_wb = automatic_white_balance(r_cropped, g_cropped, b_cropped, method='white_patch_percentile')
    im_out_wb = np.dstack([r_wb, g_wb, b_wb])

    # save white balanced image
    im_out_wb_to_save = sk.img_as_ubyte(im_out_wb)
    skio.imsave(os.path.join(os.path.dirname(__file__), '../', output_folder, f'{args.type}_white_balanced_{filename_without_ext}.png'), im_out_wb_to_save)
    print(f"Saved white balanced image to {os.path.join(output_folder, f'{args.type}_white_balanced_{filename_without_ext}.png')}")

    # color mapping via gray world assumption
    r_gw, g_gw, b_gw = automatic_white_balance(r_cropped, g_cropped, b_cropped, method='gray_world')
    im_out_gw = np.dstack([r_gw, g_gw, b_gw])

    # save gray world balanced image
    im_out_gw_to_save = sk.img_as_ubyte(im_out_gw)
    skio.imsave(os.path.join(os.path.dirname(__file__), '../', output_folder, f'{args.type}_color_mapped_{filename_without_ext}.png'), im_out_gw_to_save)
    print(f"Saved color mapped image to {os.path.join(output_folder, f'{args.type}_color_mapped_{filename_without_ext}.png')}")

if __name__ == '__main__':
    main()
