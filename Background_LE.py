import re
from pathlib import Path
import matplotlib as mpl
import matplotlib.pylab as plt
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
import pandas as pd
from tqdm import tqdm
import cell_analysis_tools as cat
from cell_analysis_tools.flim import regionprops_omi
from cell_analysis_tools.io import load_image
from natsort import natsorted
mpl.rcParams['figure.dpi'] = 300
from datetime import date
from sys import platform
import numpy as np
import tifffile
import cv2
import random
from skimage import morphology
from scipy import ndimage
from skimage.measure import label
from skimage.segmentation import clear_border
standard_dictionary = {'mask_cell': '', 'PDCO_line': '', 'cancer': '', 'treatment': '', 'fad_photons': '', 'nadh_photons': ''}
file_suffixes = {'im_photons': '.tif', 'nadh_photons': '_NADH.tif', 'fad_photons': '_FAD.tif', 'mask_cell': '_cellpose.tiff'}

def visualize_dictionary(dict_entry, extra_entries=[], remove_entries=[], channel=None, path_output: str=None):
    pass
    entries = ['nadh_photons', 'fad_photons', 'mask_cell']
    if len(extra_entries) != 0:
        entries += extra_entries
    for entry in remove_entries:
        entries.remove(entry)
    rows = 3
    cols = 5
    fig, ax = plt.subplots(rows, cols, figsize=(20, 12))
    filename_image = Path(dict_entry['nadh_photons']).stem
    fig.suptitle(f'{filename_image}')
    for pos, key in enumerate(entries):
        pass
        dict_entry[key]
        col = pos % cols
        row = pos // cols
        path_image = Path(dict_entry[key])
        if path_image.suffix in ['.tiff', '.tif']:
            image = tifffile.imread(path_image)
        else:
            image = load_image(path_image)
        if image is not None:
            if len(image.shape) > 2:
                if image.shape[2] == 3:
                    image = np.sum(image, axis=2)
                if image.shape[0] == 2:
                    image = image[channel, ...]
            ax[row, col].imshow(image)
            ax[row, col].set_title(f'{path_image.stem} \n min: {np.min(image):.3f}  max: {np.max(image):.3f}')
            ax[row, col].set_axis_off()
    if str(path_output) != 'None':
        plt.savefig(path_output / filename_image)
    plt.show()

def load_data_create_dict(path_dataset, path_output, edge_width=20):
    list_all_files = list(path_dataset.rglob('*'))
    list_str_all_files = [str(b) for b in list_all_files]
    list_str_all_files = natsorted(list_str_all_files)
    list_all_nadh_photons_images = list(filter(re.compile('.*NADH.*\\.tif$').search, list_str_all_files))
    list_all_nadh_photons_images = [image for image in list_all_nadh_photons_images if not image.startswith('.') and 'filtered' not in image]
    print('All files:', list_str_all_files)
    print('After NADH filtering:', list_all_nadh_photons_images)
    list_incomplete_sets = []
    dict_dir = {}
    for path_str_im_photons in tqdm(list_all_nadh_photons_images):
        pass
        path_im_photons_nadh = Path(path_str_im_photons)
        handle_im = path_im_photons_nadh.stem.rsplit('_', 1)[0]
        print('handle_im: ' + handle_im)
        dict_dir[handle_im] = standard_dictionary.copy()
        handle_base = path_im_photons_nadh.stem.rsplit('_', 1)[0]
        print('handle: ' + handle_base)
        print('After second filtering:', list_str_all_files)
        dict_dir[handle_im]['nadh_photons'] = list(filter(re.compile(handle_base + file_suffixes['nadh_photons']).search, list_str_all_files))[0]
        try:
            handle_mask = handle_base
            dict_dir[handle_im]['mask_cell'] = list(filter(re.compile(handle_mask + file_suffixes['mask_cell']).search, list_str_all_files))[0]
        except IndexError:
            print(f'{handle_im} | one or more masks missing skipping, set')
        try:
            path_str_im_photons_fad = list(filter(re.compile(handle_base + file_suffixes['fad_photons']).search, list_str_all_files))[0]
        except IndexError:
            print(f'{handle_im} | one or more fad files missing, skipping set')
            list_incomplete_sets.append(f'{handle_im} | missing: fad files')
            continue
        path_im_photons_fad = Path(path_str_im_photons_fad)
        handle_fad = path_im_photons_fad.stem.rsplit('_', 1)[0]
        dict_dir[handle_im]['fad_photons'] = list(filter(re.compile(handle_fad + file_suffixes['fad_photons']).search, list_str_all_files))[0]
        dict_dir[handle_im]['resolution'] = load_image(Path(dict_dir[handle_im]['nadh_photons'])).shape
    df = pd.DataFrame(dict_dir).transpose()
    df.index.name = 'base_name'
    df.to_csv(path_output / 'regionprops_omi.csv')
    if len(list_incomplete_sets) != 0:
        pass
        df_incomplete = pd.DataFrame(list_incomplete_sets, columns=['incomplete_sets'])
        df_incomplete.to_csv(path_output / 'regionprops_omi_incomplete.csv')
    return (df, list_incomplete_sets)
if __name__ == '__main__':
    pass
    path_dataset = Path('Z:\\Angela Hsu\\Angela Figure 4\\Keyence')
    path_output = path_dataset / 'outputs'
    path_output.mkdir(exist_ok=True)
    path_dictionaries = path_output / 'dictionaries'
    path_dictionaries.mkdir(exist_ok=True)
    path_features = path_output / 'features'
    path_features.mkdir(exist_ok=True)
    path_summary = path_output / 'summary'
    path_summary.mkdir(exist_ok=True)
    df, incomplete = load_data_create_dict(path_dataset=path_dataset, path_output=path_output / 'dictionaries')
    for key, item in list(df.iterrows())[:1]:
        pass
        visualize_dictionary(item)

def create_leading_edge_mask(mask, mask_path, border_size=13):
    mask_suffix = file_suffixes['mask_cell']
    cell_number_list = np.unique(mask)
    cell_number_list = cell_number_list[cell_number_list != 0]
    leading_edge_mask = np.zeros_like(mask)
    for cell_number in cell_number_list:
        sc_mask = (mask == cell_number).astype(np.uint8)
        eroded_sc_mask = cv2.erode(sc_mask, np.ones((border_size, border_size), np.uint8))
        leading_edge_sc_mask = sc_mask - eroded_sc_mask
        leading_edge_mask = leading_edge_mask + leading_edge_sc_mask * cell_number
    leading_edge_mask_path = str(mask_path).replace(mask_suffix, '_leading_edge.tiff')
    success = cv2.imwrite(leading_edge_mask_path, leading_edge_mask)
    if not success:
        print('Failed at saving image at' + leading_edge_mask_path)
    return leading_edge_mask

def largest_connected_object_size(binary_mask):
    labeled_mask, num_features = ndimage.label(binary_mask)
    flat_labels = labeled_mask.flatten()
    sizes = np.bincount(flat_labels)[1:]
    if len(sizes) > 0:
        max_size = sizes.max()
        max_label = sizes.argmax() + 1
        labeled_mask[labeled_mask == max_label] = 0
        return (max_size, max_label)
    else:
        print('No objects found in the binary image.')
        return (0, 0)

def fill_holes_like_imagej(mask):
    """
    Fills holes in a binary mask like ImageJ's 'Process > Binary > Fill Holes'.
    A 'hole' is any background region (False) *not* part of the largest connected background component.
    """
    mask = mask.astype(bool)
    background = ~mask
    labeled_bg, num = label(background, return_num=True, connectivity=1)
    counts = np.bincount(labeled_bg.ravel())
    counts[0] = 0
    largest_label = np.argmax(counts)
    keep_bg = labeled_bg == largest_label
    filled_mask = mask | ~keep_bg
    return filled_mask.astype(np.uint8)

def make_background_mask(im_intensity, mask):
    """
    - im_intensity: 2D numpy array of the image.
    - mask: 2D numpy array of the organoid mask.
    """
    gradient_magnitude = np.sqrt(cv2.Sobel(im_intensity, cv2.CV_64F, 1, 0, ksize=5) ** 2 + cv2.Sobel(im_intensity, cv2.CV_64F, 0, 1, ksize=5) ** 2)
    gradient_magnitude_min = gradient_magnitude.min()
    gradient_magnitude_max = gradient_magnitude.max()
    if gradient_magnitude_max != gradient_magnitude_min:
        normalized_gradient_magnitude = (gradient_magnitude - gradient_magnitude_min) / (gradient_magnitude_max - gradient_magnitude_min)
    else:
        normalized_gradient_magnitude = np.zeros_like(gradient_magnitude)
    normalized_gradient_magnitude_8bit = (normalized_gradient_magnitude * 255).astype(np.uint8)
    _, gradient_otsu_threshold = cv2.threshold(normalized_gradient_magnitude_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(gradient_otsu_threshold > 0) < 0.05 * im_intensity.size:
        gradient_otsu_threshold1 = gradient_otsu_threshold
        thresh = np.max(normalized_gradient_magnitude_8bit[np.logical_not(gradient_otsu_threshold1)])
        normalized_gradient_magnitude_8bit[normalized_gradient_magnitude_8bit > thresh] = 0
        _, gradient_otsu_threshold2 = cv2.threshold(normalized_gradient_magnitude_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gradient_otsu_threshold = gradient_otsu_threshold1 | gradient_otsu_threshold2
    filtered_image = morphology.remove_small_objects(gradient_otsu_threshold, min_size=25)
    gradient_otsu_threshold = filtered_image
    im_intensity_sub = im_intensity[::4, ::4]
    gradient_magnitude = np.sqrt(cv2.Sobel(im_intensity_sub, cv2.CV_64F, 1, 0, ksize=5) ** 2 + cv2.Sobel(im_intensity_sub, cv2.CV_64F, 0, 1, ksize=5) ** 2)
    gradient_magnitude_min = gradient_magnitude.min()
    gradient_magnitude_max = gradient_magnitude.max()
    if gradient_magnitude_max != gradient_magnitude_min:
        normalized_gradient_magnitude = (gradient_magnitude - gradient_magnitude_min) / (gradient_magnitude_max - gradient_magnitude_min)
    else:
        normalized_gradient_magnitude = np.zeros_like(gradient_magnitude)
    normalized_gradient_magnitude_8bit_sub = (normalized_gradient_magnitude * 255).astype(np.uint8)
    _, gradient_otsu_threshold_sub = cv2.threshold(normalized_gradient_magnitude_8bit_sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(gradient_otsu_threshold_sub > 0) < 0.05 * im_intensity_sub.size:
        gradient_otsu_threshold_sub1 = gradient_otsu_threshold_sub
        thresh = np.max(normalized_gradient_magnitude_8bit_sub[np.logical_not(gradient_otsu_threshold_sub1)])
        normalized_gradient_magnitude_8bit_sub[normalized_gradient_magnitude_8bit_sub > thresh] = 0
        _, gradient_otsu_threshold_sub2 = cv2.threshold(normalized_gradient_magnitude_8bit_sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gradient_otsu_threshold_sub = gradient_otsu_threshold_sub1 | gradient_otsu_threshold_sub2
    gradient_otsu_threshold_super = cv2.resize(gradient_otsu_threshold_sub, (im_intensity.shape[1], im_intensity.shape[0]), interpolation=cv2.INTER_LINEAR)
    gradient_otsu_threshold = gradient_otsu_threshold | gradient_otsu_threshold_super
    filtered_image = morphology.remove_small_objects(gradient_otsu_threshold, min_size=25)
    gradient_otsu_threshold = filtered_image
    print('')
    print('gradient mask done')
    selem = morphology.disk(5)
    gradient_mask_closed = morphology.binary_closing(gradient_otsu_threshold, footprint=selem)
    gradient_mask_filled = fill_holes_like_imagej(gradient_mask_closed)
    print('Gradient mask filled')
    labeled_mask, num_features = ndimage.label(gradient_mask_filled)
    flat_labels = labeled_mask.flatten()
    sizes = np.bincount(flat_labels)[1:]
    if len(sizes) > 0:
        max_size1 = sizes.max()
        max_label1 = sizes.argmax() + 1
        labeled_mask[labeled_mask == max_label1] = 0
        max_size2, max_label2 = largest_connected_object_size(labeled_mask > 0)
    else:
        max_size1 = max_label1 = max_size2 = max_label2 = 0
        print('No objects found in the binary image.')
    if (np.sum(gradient_mask_filled) > 0.8 * im_intensity.size) | (max_size1 > 5 * max_size2):
        gradient_mask_filled = gradient_otsu_threshold
    low_gradient_mask = np.logical_not(gradient_mask_filled)
    binary_mask = (mask == 0).astype(int)
    kernel_size = 250
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(im_intensity, cv2.MORPH_OPEN, kernel)
    tophat_result = cv2.subtract(im_intensity, opening)
    _, intensity_otsu_threshold = cv2.threshold(tophat_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_intensity_mask = np.logical_not(intensity_otsu_threshold)
    background_mask1 = binary_mask & low_gradient_mask & low_intensity_mask
    background_mask2 = binary_mask & low_gradient_mask
    max_size, _ = largest_connected_object_size(background_mask1)
    if max_size > 0.15 * im_intensity.size:
        background_mask = background_mask2
        if_state = 'Tophat skipped'
    else:
        background_mask = background_mask1
        if_state = 'Tophat performed'
    not_background_mask = np.logical_not(background_mask)
    print('Not Background mask done')
    flattened = im_intensity.flatten()
    percentile_value = np.percentile(flattened, 98)
    brightest_pixels_mask = im_intensity >= percentile_value
    not_background_mask = not_background_mask | brightest_pixels_mask
    filtered_image = morphology.remove_small_objects(not_background_mask, min_size=25)
    not_background_mask = filtered_image
    selem = morphology.disk(10)
    closed_holes_mask = morphology.binary_closing(not_background_mask, footprint=selem)
    filled_holes_mask = fill_holes_like_imagej(closed_holes_mask)
    final_background_mask = np.logical_not(filled_holes_mask)
    print('Kernel 10 closing filling done')
    labeled_mask, num_features = ndimage.label(filled_holes_mask)
    flat_labels = labeled_mask.flatten()
    sizes = np.bincount(flat_labels)[1:]
    if len(sizes) > 0:
        max_size1 = sizes.max()
        max_label1 = sizes.argmax() + 1
        labeled_mask[labeled_mask == max_label1] = 0
        max_size2, max_label2 = largest_connected_object_size(labeled_mask > 0)
    else:
        max_size1 = max_label1 = max_size2 = max_label2 = 0
        print('No objects found in the binary image.')
    if (np.sum(filled_holes_mask) > 0.7 * im_intensity.size) | (max_size1 > 3 * max_size2):
        not_background_mask = np.logical_not(background_mask)
        filtered_image = morphology.remove_small_objects(not_background_mask, min_size=25)
        not_background_mask = filtered_image
        selem = morphology.disk(5)
        closed_holes_mask = morphology.binary_closing(not_background_mask, footprint=selem)
        filled_holes_mask = fill_holes_like_imagej(closed_holes_mask)
        final_background_mask = np.logical_not(filled_holes_mask)
        print('Kernel 5 closing filling done')
        labeled_mask, num_features = ndimage.label(filled_holes_mask)
        flat_labels = labeled_mask.flatten()
        sizes = np.bincount(flat_labels)[1:]
        if len(sizes) > 0:
            max_size1 = sizes.max()
            max_label1 = sizes.argmax() + 1
            labeled_mask[labeled_mask == max_label1] = 0
            max_size2, max_label2 = largest_connected_object_size(labeled_mask > 0)
        else:
            max_size1 = max_label1 = max_size2 = max_label2 = 0
            print('No objects found in the binary image.')
        if (np.sum(filled_holes_mask) > 0.8 * im_intensity.size) | (max_size1 > 4 * max_size2):
            not_background_mask = np.logical_not(background_mask)
            filtered_image = morphology.remove_small_objects(not_background_mask, min_size=25)
            not_background_mask = filtered_image
            filled_holes_mask = not_background_mask
            final_background_mask = np.logical_not(not_background_mask)
            print('Zero closing was used')
    if np.mean(final_background_mask) < 0.3:
        not_background_mask = np.logical_not(background_mask)
        filtered_image = morphology.remove_small_objects(not_background_mask, min_size=25)
        not_background_mask = filtered_image
        final_background_mask = np.logical_not(not_background_mask)
    final_background_mask_temp = final_background_mask
    final_background_mask_temp[0, :] = True
    final_background_mask_temp[-1, :] = True
    final_background_mask_temp[:, 0] = True
    final_background_mask_temp[:, -1] = True
    filtered_image = morphology.remove_small_objects(final_background_mask_temp > 0, min_size=200000, connectivity=2)
    if np.sum(filtered_image) != np.sum(final_background_mask_temp):
        final_background_mask = filtered_image
        final_background_mask[0, :] = False
        final_background_mask[-1, :] = False
        final_background_mask[:, 0] = False
        final_background_mask[:, -1] = False
    print('')
    rows = 4
    cols = 1
    fig, ax = plt.subplots(rows, cols, figsize=(10, 30))
    ax[0].imshow(im_intensity)
    ax[0].set_axis_off()
    ax[0].set_title('Intensity Image')
    ax[1].imshow(np.logical_not(gradient_otsu_threshold), cmap='gray')
    ax[1].set_axis_off()
    ax[1].set_title('Gradient Mask')
    ax[2].imshow(low_gradient_mask, cmap='gray')
    ax[2].set_axis_off()
    ax[2].set_title('Close & Fill Holes')
    ax[3].imshow(final_background_mask, cmap='gray')
    ax[3].set_axis_off()
    ax[3].set_title('Subtract Fine-tuned Cellpose Mask')
    plt.show()
    return final_background_mask

def combine_masks(nadh_mask, fad_mask):
    """
    Combine the background masks for NADH and FAD images.
    
    - nadh_mask: 2D numpy array of the NADH background mask.
    - fad_mask: 2D numpy array of the FAD background mask.
    
    Returns:
    - Combined final background mask as a 2D numpy array.
    """
    combined_mask = nadh_mask & fad_mask
    total_pixels = combined_mask.size
    true_pixels_combined = np.sum(combined_mask)
    if true_pixels_combined < 0.4 * total_pixels:
        true_pixels_nadh = np.sum(nadh_mask)
        true_pixels_fad = np.sum(fad_mask)
        if true_pixels_nadh > true_pixels_fad:
            final_final_background_mask = nadh_mask
        else:
            final_final_background_mask = fad_mask
    else:
        final_final_background_mask = combined_mask
    return final_final_background_mask

def calculate_average_background(im_intensity, final_final_background_mask):
    background_pixels = im_intensity[final_final_background_mask]
    average_background = np.mean(background_pixels)
    return average_background
list_path_csv_image_sets = list((path_output / 'dictionaries').glob('*_omi.csv'))
analysis_type = 'whole_cell'
leading_edge = True
for dict_path in list_path_csv_image_sets[:1]:
    pass
    print(f'processing: {dict_path.stem}')
    df_image_set = pd.read_csv(dict_path)
    df_image_set = df_image_set.set_index('base_name', drop=True)
    for base_name, row_data in tqdm(list(df_image_set.iterrows())):
        pass
        if analysis_type == 'whole_cell':
            mask_path = Path(row_data.mask_cell)
            mask = load_image(mask_path)
        if leading_edge:
            leading_edge_mask = create_leading_edge_mask(mask, mask_path=mask_path)
        im_nadh_intensity = load_image(Path(row_data.nadh_photons))
        im_fad_intensity = load_image(Path(row_data.fad_photons))
        omi_props = regionprops_omi(image_id=base_name, label_image=mask, im_nadh_intensity=im_nadh_intensity, im_fad_intensity=im_fad_intensity, other_props=['area', 'perimeter', 'solidity', 'eccentricity', 'axis_major_length', 'axis_minor_length'])
        df = pd.DataFrame(omi_props).transpose()
        nadh_final_background_mask = make_background_mask(im_nadh_intensity, mask)
        fad_final_background_mask = make_background_mask(im_fad_intensity, mask)
        final_combined_mask = combine_masks(nadh_final_background_mask, fad_final_background_mask)
        rows = 1
        cols = 3
        fig, ax = plt.subplots(rows, cols, figsize=(20, 12))
        ax[0].imshow(im_nadh_intensity)
        ax[0].set_title('NADH Intensity Image')
        ax[1].imshow(final_combined_mask, cmap='gray')
        ax[1].set_title('Final Combined Background Mask')
        ax[2].imshow(im_fad_intensity)
        ax[2].set_title('FAD Intensity Image')
        plt.show()
        df['nadh_background'] = calculate_average_background(im_nadh_intensity, final_combined_mask)
        df['fad_background'] = calculate_average_background(im_fad_intensity, final_combined_mask)
        try:
            df['nadh_intensity_norm'] = df['nadh_intensity_mean'] / df['nadh_background']
            df['fad_intensity_norm'] = df['fad_intensity_mean'] / df['fad_background']
            df['redox_ratio'] = df['nadh_intensity_mean'] / (df['nadh_intensity_mean'] + df['fad_intensity_mean'])
            df['normalized_redox_ratio'] = df['nadh_intensity_norm'] / (df['nadh_intensity_norm'] + df['fad_intensity_norm'])
            mask_suffix = file_suffixes['mask_cell']
            background_mask_path_nadh = str(mask_path).replace(mask_suffix, '_backgndmask_NADH.tiff')
            success = cv2.imwrite(background_mask_path_nadh, (nadh_final_background_mask * 255).astype(np.uint8))
            background_mask_path_fad = str(mask_path).replace(mask_suffix, '_backgndmask_FAD.tiff')
            success = cv2.imwrite(background_mask_path_fad, (fad_final_background_mask * 255).astype(np.uint8))
            background_mask_path_final = str(mask_path).replace(mask_suffix, '_backgndmask_Final.tiff')
            success = cv2.imwrite(background_mask_path_final, (final_combined_mask * 255).astype(np.uint8))
            df['base_name'] = base_name
            for item_key in row_data.keys():
                df[item_key] = row_data[item_key]
            if leading_edge:
                leading_edge_nadh_intensity_values = im_nadh_intensity[leading_edge_mask == 1]
                leading_edge_fad_intensity_values = im_fad_intensity[leading_edge_mask == 1]
                leading_edge_nadh_intensity_mean = np.mean(leading_edge_nadh_intensity_values)
                leading_edge_fad_intensity_mean = np.mean(leading_edge_fad_intensity_values)
                leading_edge_omi_props = regionprops_omi(image_id=base_name, label_image=leading_edge_mask, im_nadh_intensity=im_nadh_intensity, im_fad_intensity=im_fad_intensity, other_props=['area'])
                leading_edge_df = pd.DataFrame(leading_edge_omi_props).transpose()
                leading_edge_df['base_name'] = base_name
                for item_key in row_data.keys():
                    leading_edge_df[item_key] = row_data[item_key]
                if not leading_edge_df.empty:
                    leading_edge_df = leading_edge_df.rename(columns={'nadh_intensity_mean': 'leading_edge_nadh_intensity_mean', 'fad_intensity_mean': 'leading_edge_fad_intensity_mean', 'area': 'leading_edge_area'})
                    df = pd.merge(df, leading_edge_df[['leading_edge_nadh_intensity_mean', 'leading_edge_fad_intensity_mean', 'leading_edge_area', 'mask_label']], on='mask_label', how='outer')
                    df['leading_edge_nadh_intensity_norm'] = df['leading_edge_nadh_intensity_mean'] / df['nadh_background']
                    df['leading_edge_fad_intensity_norm'] = df['leading_edge_fad_intensity_mean'] / df['fad_background']
                    df['leading_edge_redox_ratio'] = df['leading_edge_nadh_intensity_mean'] / (df['leading_edge_nadh_intensity_mean'] + df['leading_edge_fad_intensity_mean'])
                    df['leading_edge_normalized_redox_ratio'] = df['leading_edge_nadh_intensity_norm'] / (df['leading_edge_nadh_intensity_norm'] + df['leading_edge_fad_intensity_norm'])
                first_column_name = df.columns[0]
                desired_order = ['base_name', 'leading_edge_normalized_redox_ratio', 'nadh_intensity_mean', 'nadh_intensity_stdev', 'fad_intensity_mean', 'fad_intensity_stdev', 'area', 'perimeter', 'solidity', 'eccentricity', 'axis_major_length', 'axis_minor_length', 'nadh_background', 'fad_background', 'nadh_intensity_norm', 'fad_intensity_norm', 'redox_ratio', 'normalized_redox_ratio', 'leading_edge_nadh_intensity_mean', 'leading_edge_fad_intensity_mean', 'leading_edge_area', 'leading_edge_nadh_intensity_norm', 'leading_edge_fad_intensity_norm', 'leading_edge_redox_ratio', 'leading_edge_normalized_redox_ratio', 'mask_cell', 'PDCO_line', 'cancer', 'treatment', 'fad_photons', 'nadh_photons', 'resolution']
                missing_columns = [col for col in desired_order if col not in df.columns]
                if missing_columns:
                    raise ValueError(f'Missing columns in DataFrame: {missing_columns}')
                df_reordered = df[[first_column_name] + desired_order]
                output_file = path_output / 'features' / f'features_{base_name}_{analysis_type}.csv'
                df_reordered.to_csv(output_file)
        except KeyError as e:
            if str(e) == "'nadh_intensity_mean'":
                print(f"Skipping {base_name} due to missing 'nadh_intensity_mean'")
                continue
path_output_props = path_output / 'summary'
list_path_features_csv = list((path_output / 'features').glob('*.csv'))
df_all_props = None
for path_feat_csv in tqdm(list_path_features_csv):
    pass
    df_omi = pd.read_csv(path_feat_csv)
    df_all_props = df_omi if df_all_props is None else pd.concat([df_all_props, df_omi], ignore_index=True)
df_all_props = df_all_props.set_index('base_name', drop=True)
d = date.today().strftime('%Y_%m_%d')
df_all_props.to_csv(path_output_props / f'{d}_all_props.csv')
