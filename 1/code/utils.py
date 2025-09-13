import os
import csv
import time
import numpy as np
import skimage as sk
import skimage.io as skio
import cv2

def log_alignment(save_path,image_name, dx_ar, dy_ar, dx_ag, dy_ag, elapsed_time):
    """
    Log alignment and time results to a CSV file.
    input:
        image_name: name of the image
        dx_ar, dy_ar: shifts applied to red channel to align with blue channel
        dx_ag, dy_ag: shifts applied to green channel to align with blue channel
        elapsed_time: time taken for alignment
    """
    # csv_path = os.path.join("output_images", "alignment_log.csv")
    csv_path = os.path.join(save_path, "alignment_log.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Image Name", "dx_ar", "dy_ar", "dx_ag", "dy_ag", "Elapsed Time (s)"])
        writer.writerow([image_name, dx_ar, dy_ar, dx_ag, dy_ag, elapsed_time])


def align_exhaustive(im1, im2, search_radius, start_dx = 0, start_dy = 0, metric='ncc'):
    """
    Align im1 to im2 using exhaustive search within the given search radius.
    Use specified metric as the similarity metric.
    input:
        im1, im2: images to be aligned
        search_radius: radius of the search window (in pixels)
    output:
        best_dx, best_dy: the optimal shifts in x and y directions
    """

    # get dimensions of images
    h, w = im1.shape
    best_ncc = -np.inf
    best_ssd = np.inf

    best_dx, best_dy = 0, 0
    for dy in range(start_dy - search_radius, start_dy + search_radius + 1):
        for dx in range(start_dx - search_radius, start_dx + search_radius + 1):

            if dy >= 0 and dx >= 0:
                shifted_im1_cropped = im1[dy:h, dx:w]
                im2_cropped = im2[:h-dy, :w-dx]
            elif dy >= 0 and dx < 0:
                shifted_im1_cropped = im1[dy:h, :w+dx]
                im2_cropped = im2[:h-dy, -dx:w]
            elif dy < 0 and dx >= 0:
                shifted_im1_cropped = im1[:h+dy, dx:w]
                im2_cropped = im2[-dy:h, :w-dx]
            else:  # dy < 0 and dx < 0
                shifted_im1_cropped = im1[:h+dy, :w+dx]
                im2_cropped = im2[-dy:h, -dx:w]

            # print(f"shifted_im1_cropped shape: {shifted_im1_cropped.shape}, im2_cropped shape: {im2_cropped.shape}  ")
            
            
            if metric == 'ncc':
                # compute normalized cross-correlation
                ncc = np.sum(((shifted_im1_cropped - np.mean(shifted_im1_cropped)) * (im2_cropped - np.mean(im2_cropped))))
                ncc /= np.sqrt(np.sum((shifted_im1_cropped - np.mean(shifted_im1_cropped))**2) * np.sum((im2_cropped - np.mean(im2_cropped))**2))

                # update best shifts if current ncc is better
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_dx, best_dy = dx, dy
            
            if metric == 'ssd':
                # compute sum of squared differences
                ssd = np.sum((shifted_im1_cropped - im2_cropped)**2)

                # update best shifts if current ssd is better
                if ssd < best_ssd:
                    best_ssd = ssd
                    best_dx, best_dy = dx, dy


    # return best_dx, best_dy, best_ncc, best_ssd
    if metric == 'ncc':
        return best_dx, best_dy, best_ncc
    if metric == 'ssd':
        return best_dx, best_dy, best_ssd
    

def align_exhaustive_vectorize(im1, im2, search_radius, start_dx = 0, start_dy = 0, metric='ncc', method='color'):
    """
    Align im1 to im2 using exhaustive search within the given search radius.
    Use specified metric as the similarity metric.
    input:
        im1, im2: images to be aligned
        search_radius: radius of the search window (in pixels)
        start_dx, start_dy: starting shifts in x and y directions
        method: method to use for alignment ('color' or 'edge')
        metric: similarity metric to use ('ncc' or 'ssd')
    output:
        best_dx, best_dy: the optimal shifts in x and y directions
    """
    
    def compute_ssd(shifted_im1_cropped, im2_cropped):
        ssd = np.sum((shifted_im1_cropped - im2_cropped)**2)
        return ssd
    
    def shift_images(im1, im2, dy, dx):
        h, w = im1.shape
        if dy >= 0 and dx >= 0:
            shifted_im1_cropped = im1[dy:h, dx:w]
            im2_cropped = im2[:h-dy, :w-dx]
        elif dy >= 0 and dx < 0:
            shifted_im1_cropped = im1[dy:h, :w+dx]
            im2_cropped = im2[:h-dy, -dx:w]
        elif dy < 0 and dx >= 0:
            shifted_im1_cropped = im1[:h+dy, dx:w]
            im2_cropped = im2[-dy:h, :w-dx]
        else:  # dy < 0 and dx < 0
            shifted_im1_cropped = im1[:h+dy, :w+dx]
            im2_cropped = im2[-dy:h, -dx:w]
        return shifted_im1_cropped, im2_cropped
    

    if method == 'edge':
        im1 = cv2.Sobel(src=im1, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        im2 = cv2.Sobel(src=im2, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    elif method == 'color':
        pass


    
    # get dimensions of images
    h, w = im1.shape
    best_ncc = -np.inf
    best_ssd = np.inf

    best_dx, best_dy = 0, 0

    dys = np.array(np.arange(start_dy - search_radius, start_dy + search_radius + 1))
    dxs = np.array(np.arange(start_dx - search_radius, start_dx + search_radius + 1))

    grid = np.array(np.meshgrid(dys, dxs)).T.reshape(-1, 2)
    num_groups = 3
    group_size = grid.shape[0] // num_groups
    # group_indices = np.array_split(np.arange(grid.shape[0]), num_groups)
    groups = np.array_split(grid, num_groups)

    for group in groups:
        shifted_im1_cropped_list, im2_cropped_list = zip(*[shift_images(im1, im2, group[i, 0], group[i, 1]) for i in range(group.shape[0])])

        shifted_im1_cropped_padded = [np.pad(z, pad_width=(( (h - z.shape[0]) // 2, (h - z.shape[0] + 1) // 2),
                                ( (w - z.shape[1]) // 2, (w - z.shape[1] + 1) // 2)),
                mode='constant', constant_values=0) for z in shifted_im1_cropped_list]
        im2_cropped_padded = [np.pad(z, pad_width=(( (h - z.shape[0]) // 2, (h - z.shape[0] + 1) // 2),
                                ( (w - z.shape[1]) // 2, (w - z.shape[1] + 1) // 2)),
                mode='constant', constant_values=0) for z in im2_cropped_list]

        shifted_im1_cropped_array = np.stack(shifted_im1_cropped_padded)
        shifted_im1_cropped_mask = (shifted_im1_cropped_array != 0)
        im2_cropped_array = np.stack(im2_cropped_padded)
        im2_cropped_mask = (im2_cropped_array != 0)

        if metric == 'ncc':
            # compute means with masking
            mean_im1 = np.sum(shifted_im1_cropped_array * shifted_im1_cropped_mask, axis=(1,2)) / np.sum(shifted_im1_cropped_mask, axis=(1,2))
            mean_im2 = np.sum(im2_cropped_array * im2_cropped_mask, axis=(1,2)) / np.sum(im2_cropped_mask, axis=(1,2))
            # compute normalized cross-correlation
            ncc = np.sum(((shifted_im1_cropped_array * shifted_im1_cropped_mask - mean_im1[:, np.newaxis, np.newaxis])
                        * (im2_cropped_array * im2_cropped_mask - mean_im2[:, np.newaxis, np.newaxis])), axis=(1,2))
            ncc /= np.sqrt(np.sum((shifted_im1_cropped_array * shifted_im1_cropped_mask - mean_im1[:, np.newaxis, np.newaxis])**2, axis=(1,2))
                        * np.sum((im2_cropped_array * im2_cropped_mask - mean_im2[:, np.newaxis, np.newaxis])**2, axis=(1,2)))
            # best_ncc = np.max(np.max(ncc), best_ncc)
            best_ncc_in_group = np.max(ncc)
            if best_ncc_in_group > best_ncc:
                best_ncc = best_ncc_in_group
                # best_dy, best_dx = grid[np.argmax(ncc)]
                best_dy, best_dx = group[np.argmax(ncc)]
        if metric == 'ssd':
            ssds = map(compute_ssd, shifted_im1_cropped_list, im2_cropped_list)
            best_ssd_in_group = min(ssds)
            if best_ssd_in_group < best_ssd:
                best_ssd = best_ssd_in_group
                # best_dy, best_dx = grid[np.argmin(ssds)]
                best_dy, best_dx = group[np.argmin(ssds)]
        

    # return best_dx, best_dy, best_ncc, best_ssd
    if metric == 'ncc':
        return best_dx, best_dy, best_ncc
    if metric == 'ssd':
        return best_dx, best_dy, best_ssd
    

def build_pyramid(im, levels):
    """
    Build an image pyramid with the specified number of levels.
    input:
        im: input image
        levels: number of levels in the pyramid
        
    output:
        pyramid: list of images at each level of the pyramid
    """
    pyramid = [im]
    for i in range(1, levels):
       
        im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2))
        pyramid.append(im)
    return pyramid


def align_multiscale(im1, im2, search_radius, levels, metric='ncc', method='color'):
    """
    Align im1 to im2 using a multiscale approach. Better for images with larger resolution.
    input:
        im1, im2: images to be aligned
        search_radius: radius of the search window (in pixels) at the coarsest level
        levels: number of levels in the pyramid
        metric: similarity metric to use ('ncc' or 'ssd')
    output:
        best_dx, best_dy: the optimal shifts in x and y directions
    """
    if method == 'edge':
        im1 = cv2.Sobel(src=im1, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        im2 = cv2.Sobel(src=im2, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    elif method == 'color':
        pass
    # build pyramids for both of the images (already cropped)
    pyr1 = build_pyramid(im1, levels)
    pyr2 = build_pyramid(im2, levels)

    overall_dx, overall_dy = 0, 0

    # start exhaustive search at the coarsest level
    for level in range(levels-1, -1, -1):
        print(f"Level {level}:")
        im1_level = pyr1[level]
        im2_level = pyr2[level]

        # get dimensions of images at current level
        h, w = im1_level.shape

        
        start_dx = overall_dx // (2**level)
        start_dy = overall_dy // (2**level)

        # perform exhaustive search at the current level
        dx, dy, _ = align_exhaustive(im1_level, im2_level, search_radius, start_dx, start_dy, metric)
        # dx, dy, _ = align_exhaustive_vectorize(im1_level, im2_level, search_radius, start_dx, start_dy, metric, method=method)

        # reduce search radius for next level
        search_radius = max(2, search_radius // 2)
        
        
        overall_dx = dx * (2 ** level)
        overall_dy = dy * (2 ** level)
            
        

    return overall_dx, overall_dy


def automatic_crop(r, g, b, dx_ar, dy_ar, dx_ag, dy_ag, border_percent=0.15, threshold=0.55, metric='squared_error'):
    """
    Automatically crop the aligned images to remove excess borders. Only check the outer specified
    border_percent of the image to consider for cropping.
    input:
        im1, im2: aligned images
        dx, dy: shifts applied to im1 to align with im2
        border_percent: percentage of the image border to consider for cropping
    """

    h, w = r.shape

    crop_h = int(np.floor(border_percent * h))
    crop_w = int(np.floor(border_percent * w))

    # aligned images
    r_aligned = np.roll(r, shift=(dy_ar, dx_ar), axis=(0, 1))
    g_aligned = np.roll(g, shift=(dy_ag, dx_ag), axis=(0, 1))
    b_aligned = b

    if metric == 'squared_error':
        # squared differences between each pixel for each border
        top_diff = (r_aligned[:crop_h, :] - g_aligned[:crop_h, :])**2 + (r_aligned[:crop_h, :] - b_aligned[:crop_h, :])**2 + (g_aligned[:crop_h, :] - b_aligned[:crop_h, :])**2
        bottom_diff = (r_aligned[-crop_h:, :] - g_aligned[-crop_h:, :])**2 + (r_aligned[-crop_h:, :] - b_aligned[-crop_h:, :])**2 + (g_aligned[-crop_h:, :] - b_aligned[-crop_h:, :])**2
        left_diff = (r_aligned[:, :crop_w] - g_aligned[:, :crop_w])**2 + (r_aligned[:, :crop_w] - b_aligned[:, :crop_w])**2 + (g_aligned[:, :crop_w] - b_aligned[:, :crop_w])**2
        right_diff = (r_aligned[:, -crop_w:] - g_aligned[:, -crop_w:])**2 + (r_aligned[:, -crop_w:] - b_aligned[:, -crop_w:])**2 + (g_aligned[:, -crop_w:] - b_aligned[:, -crop_w:])**2

        # error for each pixel in each border
        stacked_top = np.dstack([r_aligned[:crop_h, :], g_aligned[:crop_h, :], b_aligned[:crop_h, :]])
        top_mask = (np.min(stacked_top, axis=2) >  0.1) & (np.max(stacked_top, axis=2) < 0.95)
        top_error = np.where(top_mask, top_diff, 3)
        
        stacked_bottom = np.dstack([r_aligned[-crop_h:, :], g_aligned[-crop_h:, :], b_aligned[-crop_h:, :]])
        bottom_mask = (np.min(stacked_bottom, axis=2) >  0.1) & (np.max(stacked_bottom, axis=2) < 0.95)
        bottom_error = np.where(bottom_mask, bottom_diff, 3)
        stacked_left = np.dstack([r_aligned[:, :crop_w], g_aligned[:, :crop_w], b_aligned[:, :crop_w]])
        left_mask = (np.min(stacked_left, axis=2) >  0.1) & (np.max(stacked_left, axis=2) < 0.95)
        left_error = np.where(left_mask, left_diff, 3)
        stacked_right = np.dstack([r_aligned[:, -crop_w:], g_aligned[:, -crop_w:], b_aligned[:, -crop_w:]])
        right_mask = (np.min(stacked_right, axis=2) >  0.1) & (np.max(stacked_right, axis=2) < 0.95)
        right_error = np.where(right_mask, right_diff, 3)
        
        # extract rows and column with error above a certain threshold and crop accordingly
        top_row_candidates = np.mean(top_error, axis=1)
        bottom_row_candidates = np.mean(bottom_error, axis=1)
        left_col_candidates = np.mean(left_error, axis=0)
        right_col_candidates = np.mean(right_error, axis=0)

        threshold = threshold
        # top_row_val = np.min(top_row_candidates[top_row_candidates > threshold]) if np.any(top_row_candidates > threshold) else np.max(top_row_candidates)
        top_row_val = np.array(top_row_candidates[top_row_candidates > threshold])[-1] if np.any(top_row_candidates > threshold) else np.max(top_row_candidates)
        top_row = np.where(top_row_candidates == top_row_val)[0][0] if np.any(top_row_candidates > threshold) else 0
        # bottom_row_val = np.min(bottom_row_candidates[bottom_row_candidates > threshold]) if np.any(bottom_row_candidates > threshold) else np.max(bottom_row_candidates)
        bottom_row_val = np.array(bottom_row_candidates[bottom_row_candidates > threshold])[0] if np.any(bottom_row_candidates > threshold) else np.max(bottom_row_candidates)
        bottom_row = len(bottom_row_candidates) - np.where(bottom_row_candidates == bottom_row_val)[0][0] if np.any(bottom_row_candidates > threshold) else 0
        # left_col_val = np.min(left_col_candidates[left_col_candidates > threshold]) if np.any(left_col_candidates > threshold) else np.max(left_col_candidates)
        left_col_val = np.array(left_col_candidates[left_col_candidates > threshold])[-1] if np.any(left_col_candidates > threshold) else np.max(left_col_candidates)
        left_col = np.where(left_col_candidates == left_col_val)[0][0] if np.any(left_col_candidates > threshold) else 0
        # right_col_val = np.min(right_col_candidates[right_col_candidates > threshold]) if np.any(right_col_candidates > threshold) else np.max(right_col_candidates)
        right_col_val = np.array(right_col_candidates[right_col_candidates > threshold])[0] if np.any(right_col_candidates > threshold) else np.max(right_col_candidates)
        right_col = len(right_col_candidates) - np.where(right_col_candidates == right_col_val)[0][0] if np.any(right_col_candidates > threshold) else 0
        
        

        # crop the aligned images
        r_cropped = r_aligned[top_row:-bottom_row, left_col:-right_col] if bottom_row !=0 and right_col !=0 else \
            r_aligned[top_row:, left_col:] if bottom_row ==0 and right_col ==0 else \
            r_aligned[top_row:-bottom_row, left_col:] if right_col ==0 else \
            r_aligned[top_row:, left_col:-right_col]
        
        g_cropped = g_aligned[top_row:-bottom_row, left_col:-right_col] if bottom_row !=0 and right_col !=0 else \
            g_aligned[top_row:, left_col:] if bottom_row ==0 and right_col ==0 else \
            g_aligned[top_row:-bottom_row, left_col:] if right_col ==0 else \
            g_aligned[top_row:, left_col:-right_col]
        
        b_cropped = b_aligned[top_row:-bottom_row, left_col:-right_col] if bottom_row !=0 and right_col !=0 else \
            b_aligned[top_row:, left_col:] if bottom_row ==0 and right_col ==0 else \
            b_aligned[top_row:-bottom_row, left_col:] if right_col ==0 else \
            b_aligned[top_row:, left_col:-right_col]

    return r_cropped, g_cropped, b_cropped

def automatic_contrast_adjustment(r, g, b, method='min_max'):
    """
    Automatically adjust the contrast of the aligned images using min-max scaling.
    input:
        r, g, b: aligned images
    output:
        r_adj, g_adj, b_adj: contrast adjusted images
    """
    def adjust_channel_min_max(channel):
        min_val = np.min(channel)
        max_val = np.max(channel)
        channel_adj = (channel - min_val) / (max_val - min_val)
        return channel_adj
    
    def adjust_channel_hist_eq(channel):
        # compute histogram
        hist, bins = np.histogram(channel.flatten(), bins=256, range=[0,1])
        cdf = hist.cumsum()

        cdf_normalized = cdf / cdf[-1]  # normalize to [0,1]
        channel_adj = np.interp(x=channel.flatten(), xp=bins[:-1], fp=cdf_normalized)
        channel_adj = channel_adj.reshape(channel.shape)
        return channel_adj
    
    def adjust_luminance_hist_eq(r,g,b):
        img = np.dstack([r, g, b])
        ycbcr = sk.color.rgb2ycbcr(img)
        y = ycbcr[:,:,0]/255.0
        y_eq = adjust_channel_hist_eq(y)
        ycbcr[:,:,0] = y_eq * 255.0
        img_eq = sk.color.ycbcr2rgb(ycbcr)
        img_eq = np.clip(img_eq, 0, 1)
        red_adj = img_eq[:,:,0]
        green_adj = img_eq[:,:,1]
        blue_adj = img_eq[:,:,2]
        return red_adj, green_adj, blue_adj

    if method == 'min_max':
        r_adj = adjust_channel_min_max(r)
        g_adj = adjust_channel_min_max(g)
        b_adj = adjust_channel_min_max(b)
    elif method == 'hist_eq':
        r_adj = adjust_channel_hist_eq(r)
        g_adj = adjust_channel_hist_eq(g)
        b_adj = adjust_channel_hist_eq(b)
    
    elif method == 'hist_eq_luminance':
        r_adj, g_adj, b_adj = adjust_luminance_hist_eq(r, g, b)

    return r_adj, g_adj, b_adj

def automatic_white_balance(r, g, b, method='gray_world'):
    """
    Automatically adjust the white balance of aligned images using gray world assumption.
    input:
        r, g, b: aligned images
        method: white balance method (default: 'gray_world')
    output:
        r_wb, g_wb, b_wb: white balanced images
    """

    def gray_world(r, g, b):
        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        mean_avg = (r_mean + g_mean + b_mean) / 3.0
        # mean_avg = 1.0
        r_wb = np.clip(r * (mean_avg / r_mean), 0, 1)
        g_wb = np.clip(g * (mean_avg / g_mean), 0, 1)
        b_wb = np.clip(b * (mean_avg / b_mean), 0, 1)
        return r_wb, g_wb, b_wb
    
    def white_patch_naive(r, g, b):
        r_max = np.max(r)
        g_max = np.max(g)
        b_max = np.max(b)
        r_wb = np.clip(r * (1.0 / r_max), 0, 1)
        g_wb = np.clip(g * (1.0 / g_max), 0, 1)
        b_wb = np.clip(b * (1.0 / b_max), 0, 1)
        return r_wb, g_wb, b_wb
    
    def white_patch_percentile(r, g, b, percentile=95):
        r_perc = np.percentile(r, percentile)
        g_perc = np.percentile(g, percentile)
        b_perc = np.percentile(b, percentile)
        r_wb = np.clip(r * (1.0 / r_perc), 0, 1)
        g_wb = np.clip(g * (1.0 / g_perc), 0, 1)
        b_wb = np.clip(b * (1.0 / b_perc), 0, 1)
        return r_wb, g_wb, b_wb
        

    if method == 'gray_world':
        r_wb, g_wb, b_wb = gray_world(r, g, b)
    elif method == 'white_patch_naive':
        r_wb, g_wb, b_wb = white_patch_naive(r, g, b)
    elif method == 'white_patch_percentile':
        r_wb, g_wb, b_wb = white_patch_percentile(r, g, b, percentile=95)

    return r_wb, g_wb, b_wb



