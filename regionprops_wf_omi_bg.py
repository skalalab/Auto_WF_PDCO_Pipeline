import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pylab as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import pandas as pd
from tqdm import tqdm

import cell_analysis_tools as cat
from cell_analysis_tools.flim import regionprops_omi
from cell_analysis_tools.io import load_image
from natsort import natsorted

mpl.rcParams["figure.dpi"] = 300
from datetime import date

from sys import platform

import numpy as np
import tifffile

import cv2
import random
from skimage import morphology
from scipy import ndimage
# from scipy.ndimage import label, find_objects
from skimage.measure import label
from skimage.segmentation import clear_border
#%% Placeholder dictionary of header entries

standard_dictionary = {
    # experiment details
    # "date": "",  # YYYYMMDD
    # masks
    # "mask_nuclei": "",  # path_to_file
    # "mask_cytoplasm": "",  # path_to_file
    "mask_cell": "",  # path_to_file
    # metadata
    # "analyzed_by": "",  # initials
    # "reanalyzed_by": "",  # initials
    # "nadh_image_number": "",
    # "fad_image_number": "",
    # "experiment": "",
    # "media": "",
    # "cell_type": "",
    "PDCO_line": "",
    "cancer": "",
    # "dish": "",
    # "experiment": "",  # ["confluency", "glucose" ,"ph","seahorse","duroquinone","tmre"]
    "treatment": "",  # 2DG, Rotenone, IAA, Antimycin
    # spc exports
    # fad
    "fad_photons": "",  # path_to_file,
    # "fad_a1": "",  # path_to_file,
    # "fad_a2": "",  # path_to_file,
    # "fad_t1": "",  # path_to_file,
    # "fad_t2": "",  # path_to_file,
    # nadh
    "nadh_photons": "",  # path_to_file,
    # "nadh_a1": "",  # path_to_file,
    # "nadh_a2": "",  # path_to_file,
    # "nadh_t1": "",  # path_to_file,
    # "nadh_t2": "",  # path_to_file,
    # other parameters
    # "dye": "",
    # "resolution": "",
    # "objective": "",
    # "filter_cube": "",
}

#%% Suffixes to use for matching files

file_suffixes = {
    "im_photons": ".tif",
    # "mask_cell": "_cp_masks.tiff",
    "mask_cell": "_cellpose.tiff",
    # "mask_cell": "_photons_cellpose.tiff",
    # "mask_cell": "_cellpose.tiff",
    # "mask_cell": "_photons_cellmask.tiff",
    # "mask_cell": "_mask.tiff",
    # "mask_cytoplasm": "_mask_cyto.tiff",
    # "mask_nuclei": "_mask_nuclei.tiff",
    # "a1[%]": "_a1\[%\].asc",
    # "a2[%]": "_a2\[%\].asc",
    # "t1": "_t1.asc",
    # "t2": "_t2.asc",
    # "chi": "_chi.asc",
    # "sdt": ".sdt",
}

#%% function for visualizing dictionary


def visualize_dictionary(
    dict_entry,
    extra_entries=[],
    remove_entries=[],
    channel=None,
    path_output: str = None,
):
    pass
    # show grid of all the images it's matching up
    # photons, masks, spc outputs
    entries = [
        "nadh_photons",
        # "nadh_a1",
        # "nadh_a2",
        # "nadh_t1",
        # "nadh_t2",
        #
        "fad_photons",
        # "fad_a1",
        # "fad_a2",
        # "fad_t1",
        # "fad_t2",
        "mask_cell",
        # "mask_cytoplasm",
        # "mask_nuclei",
    ]
    if len(extra_entries) != 0:
        entries += extra_entries
    for entry in remove_entries:
        entries.remove(entry)

    rows = 3
    cols = 5
    fig, ax = plt.subplots(rows, cols, figsize=(20, 12))
    filename_image = Path(dict_entry["nadh_photons"]).stem
    # if platform == "linux":
    #     dataset_dir = str(Path(dict_entry["nadh_photons"]).parent).split('/', 5)[5]
    # else:
    #     dataset_dir = str(Path(dict_entry["nadh_photons"]).parent).split("\\", 5)[5]
        
        
    # fig.suptitle(f"{filename_image} \n {dataset_dir}")
    fig.suptitle(f"{filename_image}")

    for pos, key in enumerate(entries):
        pass
        dict_entry[key]

        col = pos % cols
        row = pos // cols

        path_image = Path(dict_entry[key])
        # load correct format
        if path_image.suffix in [".tiff", ".tif"]:
            image = tifffile.imread(path_image)
        else:
            image = load_image(path_image)

        # load proper channel 
        if image is not None:
            if len(image.shape) > 2:
                if image.shape[2] == 3:  # rgb image of cyto mask
                    image = np.sum(image, axis=2)
                if image.shape[0] == 2:  # multi channel image, pick channel number
                    image = image[channel, ...]        
            ax[row, col].imshow(image)
            ax[row, col].set_title(
                f"{path_image.stem} \n min: {np.min(image):.3f}  max: {np.max(image):.3f}"
                )
            ax[row, col].set_axis_off()
            # for item in ([ax[row, col].title, ax[row, col].xaxis.label, ax[row, col].yaxis.label] +
            #              ax[row, col].get_xticklabels() + ax[row, col].get_yticklabels()):
            #     item.set_fontsize(20)

    if str(path_output) != "None":
        plt.savefig(path_output / filename_image)
    plt.show()

#%% load and aggregate data into a set


def load_data_create_dict(path_dataset, path_output, edge_width=20):
    ##%% Load dataset paths and find all nadh images

    # GET LIST OF ALL FILES FOR REGEX
    list_all_files = list(path_dataset.rglob("*"))
    list_str_all_files = [str(b) for b in list_all_files]
    list_str_all_files = natsorted(list_str_all_files)

    # GET LIST OF ALL PHOTONS IMAGES
    list_all_nadh_photons_images = list(
        # filter(re.compile(r".*Ch2-_photons.asc").search, list_str_all_files)
        # filter(re.compile(r".*_photons.asc").search, list_str_all_files)
        # filter(re.compile(r".*n\..*_photons.asc").search, list_str_all_files)
        filter(re.compile(".*NADH.*\.tif$").search, list_str_all_files)
    )
    list_all_nadh_photons_images = [image for image in list_all_nadh_photons_images if not image.startswith('.') and 'filtered' not in image]

    ##%% POPULATE DICTIONARY
    print("All files:", list_str_all_files)
    print("After NADH filtering:", list_all_nadh_photons_images)

    list_incomplete_sets = []

    dict_dir = {}
    for path_str_im_photons in tqdm(list_all_nadh_photons_images):
        pass
    
        # generate dict name
        path_im_photons_nadh = Path(path_str_im_photons)
        handle_im = path_im_photons_nadh.stem.rsplit("_", 1)[0]
        print("handle_im: " + handle_im)
        # handle_im = path_im_photons_nadh.stem
        dict_dir[handle_im] = standard_dictionary.copy()

        # NADH
        
        handle_nadh = path_im_photons_nadh.stem.rsplit("_", 1)[0]
        print("handle: "+ handle_nadh)
        # nadh image number and treatment
        # image_number_nadh = int(handle_nadh.rsplit("-",1)[1])
        # _, treatment, dish, _, _, = handle_nadh.split("_")

        # dict_dir[handle_im]["nadh_image_number"] = image_number_nadh
        # dict_dir[handle_im]["dish"] = int(dish.replace("d",""))

        # standardize treatment name
        # if treatment == "eto" : treatment = "etomoxir"
        # elif treatment == "IAA" : treatment = "iodoacetic acid"
        # elif treatment == "SA" : treatment = "sodium arsenite"
        # elif treatment == "SF" : treatment = "sodium fluoroacetate"

        # dict_dir[handle_im]["treatment"] = treatment

        # paths to files
        print("After second filtering:", list_str_all_files)
        dict_dir[handle_im]["nadh_photons"] = list(
            filter(
                re.compile(handle_nadh + file_suffixes["im_photons"]).search,
                list_str_all_files,
            )
        )[0]
        
        # dict_dir[handle_im]["nadh_a1"] = list(
        #     filter(
        #         re.compile(handle_nadh + file_suffixes["a1[%]"]).search,
        #         list_str_all_files,
        #     )
        # )[0]
        # dict_dir[handle_im]["nadh_a2"] = list(
        #     filter(
        #         re.compile(handle_nadh + file_suffixes["a2[%]"]).search,
        #         list_str_all_files,
        #     )
        # )[0]
        # dict_dir[handle_im]["nadh_t1"] = list(
        #     filter(
        #         re.compile(handle_nadh + file_suffixes["t1"]).search, list_str_all_files
        #     )
        # )[0]
        # dict_dir[handle_im]["nadh_t2"] = list(
        #     filter(
        #         re.compile(handle_nadh + file_suffixes["t2"]).search, list_str_all_files
        #     )
        # )[0]

        # MASKS
        try:
            handle_mask = handle_nadh
            dict_dir[handle_im]["mask_cell"] = list(
                filter(
                    re.compile(handle_mask + file_suffixes["mask_cell"]).search,
                    list_str_all_files,
                )
            )[0]
            # dict_dir[handle_im]["mask_cytoplasm"] = list(
            #     filter(
            #         re.compile(handle_mask + file_suffixes["mask_cytoplasm"]).search,
            #         list_str_all_files,
            #     )
            # )[0]
            # dict_dir[handle_im]["mask_nuclei"] = list(
            #     filter(
            #         re.compile(handle_mask + file_suffixes["mask_nuclei"]).search,
            #         list_str_all_files,
            #     )
            # )[0]
  
        except IndexError:
            print(f"{handle_im} | one or more masks missing skipping, set")
            # list_incomplete_sets.append(f"{handle_im} | missing: mask files")
            # del dict_dir[handle_im]
            # continue

        # FAD
        # locate corresponding photons image
        handle_im_fad = re.sub(r'NADH', 'FAD', handle_im)
        try:
            path_str_im_photons_fad = list(
                 filter(
                 #     re.compile("FAD" + file_suffixes["im_photons"]).search,
                     re.compile(handle_im_fad + file_suffixes["im_photons"]).search,
                     list_str_all_files,
                 )
            )[0]
        except IndexError:
            print(f"{handle_im} | one or more fad files missing, skipping set")
            list_incomplete_sets.append(f"{handle_im} | missing: fad files")
            # del dict_dir[handle_im]
            continue

        path_im_photons_fad = Path(path_str_im_photons_fad)
        handle_fad = path_im_photons_fad.stem.rsplit("_", 1)[0]

        # image number
        # image_number = handle_fad.rsplit("-",1)[1]
        # dict_dir[handle_im]["fad_image_number"] = int(image_number)

        # paths to images
        dict_dir[handle_im]["fad_photons"] = list(
             filter(
                 re.compile(handle_fad + file_suffixes["im_photons"]).search,
                 list_str_all_files,
             )
         )[0]
        
        # dict_dir[handle_im]["fad_a1"] = list(
        #     filter(
        #         re.compile(handle_fad + file_suffixes["a1[%]"]).search,
        #         list_str_all_files,
        #     )
        # )[0]
        # dict_dir[handle_im]["fad_a2"] = list(
        #     filter(
        #         re.compile(handle_fad + file_suffixes["a2[%]"]).search,
        #         list_str_all_files,
        #     )
        # )[0]
        # dict_dir[handle_im]["fad_t1"] = list(
        #     filter(
        #         re.compile(handle_fad + file_suffixes["t1"]).search, list_str_all_files
        #     )
        # )[0]
        # dict_dir[handle_im]["fad_t2"] = list(
        #     filter(
        #         re.compile(handle_fad + file_suffixes["t2"]).search, list_str_all_files
        #     )
        # )[0]

        # OTHER HEADER CSV INFO
        # dict_dir[handle_im]["date"] = metadata_experiment["date"][0]
        # dict_dir[handle_im]["cell_line"] = metadata_experiment["cell_line"][0]
        # dict_dir[handle_im]["cell_type"] = metadata_experiment["cell_type"][0]
        # dict_dir[handle_im]["tissue"] = metadata_experiment["tissue"][0]
        # dict_dir[handle_im]["cancer"] = metadata_experiment["cancer"][0]
        # dict_dir[handle_im]["media"] = metadata_experiment["media"][0]
        # dict_dir[handle_im]["dye"] = metadata_experiment["dye"][0]
        #dict_dir[handle_im]["resolution"] = metadata_experiment["resolution"][0]
        # dict_dir[handle_im]["objective"] = metadata_experiment["objective"][0]
        # dict_dir[handle_im]["filter_cube"] = metadata_experiment["filter_cube"][0]
        # dict_dir[handle_im]["experiment"] = metadata_experiment["experiment"][0]

        # DIRECTORY FILENAME INFO
        # dict_dir[handle_im]["analyzed_by"] = analyzed_by

        # OTHER VALUES
        dict_dir[handle_im]["resolution"] = load_image(
            Path(dict_dir[handle_im]["nadh_photons"])
        ).shape

    # export
    df = pd.DataFrame(dict_dir).transpose()
    df.index.name = "base_name"
    df.to_csv(path_output / "regionprops_omi.csv")

    if len(list_incomplete_sets) != 0:
        pass
        df_incomplete = pd.DataFrame(list_incomplete_sets, columns=["incomplete_sets"])
        df_incomplete.to_csv(path_output / "regionprops_omi_incomplete.csv")

    return df, list_incomplete_sets

    #%%


if __name__ == "__main__":
    pass
    #%% Load dictionary and compute the regionprops omi parameters
    
    # Comment in if running example
    HERE = Path(__file__).absolute().resolve().parent
    # path_dataset = HERE.parent / r"\\skala-dv1.discovery.wisc.edu\ws\skala\Angela Hsu\Organoid\New folder (2)\New folder"
    path_dataset = HERE.parent / r"Z:\skala\Angela Hsu\060325_keyence redo\062025_13pxl redo\13pxl auto\final 100 Romi auto"
    
    # path_output = HERE.parent / "regionprops_omi/outputs/"
    
    # comment in if running own code
    # path_dataset = Path(r"\\skala-dv1.discovery.wisc.edu\ws\skala\0-Projects and Experiments\ECG - Mohit_CASPI\211027_Panc1_10s-60s\Paired_1\256_60s_n\ROI_summed")
    # path_dataset = Path(r"\\skala-dv1.discovery.wisc.edu\ws\skala\0-Projects and Experiments\KS - regionprops test\ROI_summed_orig_names")
    # path_dataset = Path(r"\\skala-dv1.discovery.wisc.edu\ws\skala\0-Projects and Experiments\KS - regionprops test\ROI_summed")
    # path_dataset = Path(r"E:\Analysis\Darcie Moore\2P fits 2ch")
    # path_dataset = Path(r"E:\Analysis\Dissociated_OV_ibidi_M\Combined2\ROI_summed")
    # path_dataset = Path(r"E:\Analysis\231115_PBMC_1018")

    
    path_output = path_dataset / "outputs"
    path_output.mkdir(exist_ok=True)
    path_dictionaries = path_output / "dictionaries"
    path_dictionaries.mkdir(exist_ok=True)
    path_features = path_output / "features" 
    path_features.mkdir(exist_ok=True)
    path_summary = path_output / "summary" 
    path_summary.mkdir(exist_ok=True)


    df, incomplete = load_data_create_dict(
        path_dataset=path_dataset, path_output=(path_output / "dictionaries")
    )

    for key, item in list(df.iterrows())[:1]:
        pass
        visualize_dictionary(item)
#%% Create a leading edge mask by subtracting a border of given size from the mask.
def create_leading_edge_mask(mask, mask_path, border_size=13):
    mask_suffix = file_suffixes['mask_cell']
    cell_number_list = np.unique(mask)
    cell_number_list = cell_number_list[cell_number_list != 0]
    leading_edge_mask = np.zeros_like(mask)
    for cell_number in cell_number_list:
        # single cell mask 
        sc_mask = (mask == cell_number).astype(np.uint8)
        eroded_sc_mask = cv2.erode(sc_mask, np.ones((border_size, border_size), np.uint8))
        # obtain leading edge mask by subtracting the eroded mask from single cell mask
        leading_edge_sc_mask = sc_mask - eroded_sc_mask
        leading_edge_mask = leading_edge_mask + leading_edge_sc_mask * cell_number
    leading_edge_mask_path =str(mask_path).replace(mask_suffix, "_leading_edge.tiff")
    success = cv2.imwrite(leading_edge_mask_path, leading_edge_mask)
    if not success:
      print("Failed at saving image at" + leading_edge_mask_path)  

    return leading_edge_mask
#%% calculate background intensity 
def largest_connected_object_size(binary_mask):
    # Label connected components
    labeled_mask, num_features = ndimage.label(binary_mask)
    
    # Calculate the size of each connected component
    
    # sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    flat_labels = labeled_mask.flatten()
    sizes = np.bincount(flat_labels)[1:]  # [1:] to exclude the background (label 0)
   
    
    # Return the size of the largest connected component
    if len(sizes) > 0:  # Check if there are any sizes to process
        max_size = sizes.max()
        max_label = sizes.argmax() + 1  # +1 to adjust for background label
        labeled_mask[labeled_mask == max_label] = 0
        return max_size, max_label
       
    else:
        print("No objects found in the binary image.")
        return 0, 0  # Return 0 if no connected components are found

def fill_holes_like_imagej(mask):
    """
    Fills holes in a binary mask like ImageJ's 'Process > Binary > Fill Holes'.
    A 'hole' is any background region (False) *not* part of the largest connected background component.
    """
    mask = mask.astype(bool)
    background = ~mask  # holes and outer background

    # Label all connected background regions
    labeled_bg, num = label(background, return_num=True, connectivity=1)

    # Find the largest background region (usually the true background)
    counts = np.bincount(labeled_bg.ravel())
    counts[0] = 0  # exclude background label 0 (which is for masked-out or unlabeled)

    largest_label = np.argmax(counts)

    # Create a mask for the largest background component
    keep_bg = labeled_bg == largest_label

    # Fill everything else (i.e., enclosed holes)
    filled_mask = mask | ~keep_bg

    return filled_mask.astype(np.uint8)

def make_background_mask(im_intensity, mask):
    """
    - im_intensity: 2D numpy array of the image.
    - mask: 2D numpy array of the organoid mask.
    """
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(cv2.Sobel(im_intensity, cv2.CV_64F, 1, 0, ksize=5)**2 + cv2.Sobel(im_intensity, cv2.CV_64F, 0, 1, ksize=5)**2)

    # Normalize gradient magnitude to [0, 1]
    gradient_magnitude_min = gradient_magnitude.min()
    gradient_magnitude_max = gradient_magnitude.max()
    if gradient_magnitude_max != gradient_magnitude_min:
        normalized_gradient_magnitude = (gradient_magnitude - gradient_magnitude_min) / (gradient_magnitude_max - gradient_magnitude_min)
    else:
        normalized_gradient_magnitude = np.zeros_like(gradient_magnitude)
    # Convert to 8-bit image (required for cv2.threshold)
    normalized_gradient_magnitude_8bit = (normalized_gradient_magnitude * 255).astype(np.uint8)
    # Apply Otsu's method to find the optimal threshold
    _, gradient_otsu_threshold = cv2.threshold(normalized_gradient_magnitude_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
   
    # If high gradient population is smaller than 5% of pixels, do a second thresholding on the low gradient population
    if np.sum(gradient_otsu_threshold > 0) < 0.05 * im_intensity.size:
        gradient_otsu_threshold1 = gradient_otsu_threshold
        thresh = np.max(normalized_gradient_magnitude_8bit[np.logical_not(gradient_otsu_threshold1)])
        normalized_gradient_magnitude_8bit[normalized_gradient_magnitude_8bit > thresh] = 0
        _, gradient_otsu_threshold2 = cv2.threshold(normalized_gradient_magnitude_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gradient_otsu_threshold = gradient_otsu_threshold1 | gradient_otsu_threshold2

    ##### Remove less than five pixel objects
    filtered_image = morphology.remove_small_objects(gradient_otsu_threshold, min_size=25)
    gradient_otsu_threshold = filtered_image
    # labeled_mask, num_features = label(gradient_otsu_threshold)
    # for i in range(1, num_features + 1):
    #     if np.sum(labeled_mask == i)<25:
    #         labeled_mask[labeled_mask == i] = 0
    # gradient_otsu_threshold = labeled_mask>0
    #### Morphological way of removing small objects
    # structuring_element = morphology.disk(15)  # You can adjust the size of the disk
    # opened_image = morphology.opening(gradient_otsu_threshold, structuring_element)
    # gradient_otsu_threshold = opened_image
    
    # Subsample intensity image and perform Sobel steps again.
    im_intensity_sub = im_intensity[::4, ::4]
    gradient_magnitude = np.sqrt(cv2.Sobel(im_intensity_sub, cv2.CV_64F, 1, 0, ksize=5)**2 + cv2.Sobel(im_intensity_sub, cv2.CV_64F, 0, 1, ksize=5)**2)
    # Normalize gradient magnitude to [0, 1]
    gradient_magnitude_min = gradient_magnitude.min()
    gradient_magnitude_max = gradient_magnitude.max()
    if gradient_magnitude_max != gradient_magnitude_min:
        normalized_gradient_magnitude = (gradient_magnitude - gradient_magnitude_min) / (gradient_magnitude_max - gradient_magnitude_min)
    else:
        normalized_gradient_magnitude = np.zeros_like(gradient_magnitude)
    # Convert to 8-bit image (required for cv2.threshold)
    normalized_gradient_magnitude_8bit_sub = (normalized_gradient_magnitude * 255).astype(np.uint8)
    # Apply Otsu's method to find the optimal threshold
    _, gradient_otsu_threshold_sub = cv2.threshold(normalized_gradient_magnitude_8bit_sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # If high gradient population is smaller than 5% of pixels, do a second thresholding on the low gradient population
    if np.sum(gradient_otsu_threshold_sub > 0) < 0.05 * im_intensity_sub.size:
        gradient_otsu_threshold_sub1 = gradient_otsu_threshold_sub
        thresh = np.max(normalized_gradient_magnitude_8bit_sub[np.logical_not(gradient_otsu_threshold_sub1)])
        normalized_gradient_magnitude_8bit_sub[normalized_gradient_magnitude_8bit_sub>thresh] = 0
        _, gradient_otsu_threshold_sub2 = cv2.threshold(normalized_gradient_magnitude_8bit_sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gradient_otsu_threshold_sub = gradient_otsu_threshold_sub1 | gradient_otsu_threshold_sub2
    gradient_otsu_threshold_super = cv2.resize(gradient_otsu_threshold_sub, (im_intensity.shape[1], im_intensity.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Add gradient mask from sub-sampled image to gradient mask from original image
    gradient_otsu_threshold = gradient_otsu_threshold | gradient_otsu_threshold_super
    
    ##### Remove less than five pixel objects
    filtered_image = morphology.remove_small_objects(gradient_otsu_threshold, min_size=25)
    gradient_otsu_threshold = filtered_image
    # labeled_mask, num_features = label(gradient_otsu_threshold)
    # for i in range(1, num_features + 1):
    #     if np.sum(labeled_mask == i)<25:
    #         labeled_mask[labeled_mask == i] = 0
    # gradient_otsu_threshold = labeled_mask>0
    #### Morphological way of removing small objects
    # structuring_element = morphology.disk(15)  # You can adjust the size of the disk
    # opened_image = morphology.opening(gradient_otsu_threshold, structuring_element)
    # gradient_otsu_threshold = opened_image
    print("")
    print("gradient mask done")
    
    #Closing and filling holes
    selem = morphology.disk(5)
    gradient_mask_closed = morphology.binary_closing(gradient_otsu_threshold, footprint=selem)
    # gradient_mask_filled = ndimage.binary_fill_holes(gradient_mask_closed)
    gradient_mask_filled = fill_holes_like_imagej(gradient_mask_closed)
    print("Gradient mask filled")
    
    # Detect if we have over masked or have large connected objects    
    # labeled_mask, num_features = label(gradient_mask_filled)
    labeled_mask, num_features = ndimage.label(gradient_mask_filled)
    # sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    flat_labels = labeled_mask.flatten()
    sizes = np.bincount(flat_labels)[1:]  # [1:] to exclude the background (label 0)
    if len(sizes) > 0:  # Check if there are any sizes to process
        max_size1 = sizes.max()
        max_label1 = sizes.argmax() + 1  # +1 to adjust for background label
        labeled_mask[labeled_mask == max_label1] = 0        
        # Compute the size of the largest connected object in the updated mask
        max_size2, max_label2 = largest_connected_object_size(labeled_mask > 0)
    else:
        max_size1 = max_label1 = max_size2 = max_label2 = 0
        print("No objects found in the binary image.")
        
    if (np.sum(gradient_mask_filled)>0.8*im_intensity.size)|(max_size1>5*max_size2):
        gradient_mask_filled = gradient_otsu_threshold
    
    #Mark low-gradient regions as 1
    low_gradient_mask = np.logical_not(gradient_mask_filled)
    
    #Make a binary cellpose mask, where pixels within organoids are 0
    binary_mask = (mask == 0).astype(int)
    
    # Define kernel size for the morphological operation
    kernel_size = 250  # Adjust based on your image
    # Create a kernel for the morphological operation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Apply morphological opening
    opening = cv2.morphologyEx(im_intensity, cv2.MORPH_OPEN, kernel)
    # Apply top-hat transformation
    tophat_result = cv2.subtract(im_intensity, opening)
    # Apply Otsu's thresholding on the top-hat transformed image
    _, intensity_otsu_threshold = cv2.threshold(tophat_result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_intensity_mask = np.logical_not(intensity_otsu_threshold)
    
    background_mask1 = binary_mask & low_gradient_mask & low_intensity_mask
    background_mask2 = binary_mask & low_gradient_mask
    
    max_size,_ = largest_connected_object_size(background_mask1)
    if max_size > 0.15 * im_intensity.size:
        background_mask = background_mask2
        if_state = "Tophat skipped"
    else:
        background_mask = background_mask1
        if_state = "Tophat performed"
    #Set background regions as 0
    not_background_mask = np.logical_not(background_mask)
    print("Not Background mask done")
    
    # Add top 2% brightest pixels to the not_background_mask
    flattened = im_intensity.flatten()
    percentile_value = np.percentile(flattened, 98)
    # Create a mask of the brightest pixels
    brightest_pixels_mask = im_intensity >= percentile_value
    not_background_mask = not_background_mask | brightest_pixels_mask
    
    ##### Remove less than 200 pixel objects
    filtered_image = morphology.remove_small_objects(not_background_mask, min_size=200)
    not_background_mask = filtered_image

    
  
    
    # Add one pixel border to frame before closing and filling holes in the low gradient masks
    selem = morphology.disk(10)  # Adjust the size of the structuring element if needed
    
    # not_background_mask[0, :] = True      # Top row
    # not_background_mask[-1, :] = True     # Bottom row
    # not_background_mask[:, 0] = True      # Left column
    closed_holes_mask = morphology.binary_closing(not_background_mask, footprint=selem)
    # #filled_holes_mask = ndimage.binary_fill_holes(closed_holes_mask)
    filled_holes_mask = fill_holes_like_imagej(closed_holes_mask)
    
    # filled_holes_mask[0, :] = False      # Top row
    # filled_holes_mask[-1, :] = False     # Bottom row
    # filled_holes_mask[:, 0] = False      # Left column
    # filled_holes_mask[:, -1] = True     # Right column
    # closed_holes_mask = morphology.binary_closing(filled_holes_mask, footprint=selem)
    # filled_holes_mask = ndimage.binary_fill_holes(closed_holes_mask)
    
    
    # filled_holes_mask[0, :] = True      # Top row
    # filled_holes_mask[-1, :] = True     # Bottom row
    # filled_holes_mask[:, 0] = True      # Left column
    final_background_mask = np.logical_not(filled_holes_mask)
    # filled_holes_mask[0, :] = False      # Top row
    # filled_holes_mask[-1, :] = False     # Bottom row
    # filled_holes_mask[:, 0] = False      # Left column
    # filled_holes_mask[:, -1] = False     # Right column
    print("Kernel 10 closing filling done")
    
    # Detect if we have over masked or have large connected objects    
    labeled_mask, num_features = ndimage.label(filled_holes_mask)
    # sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    flat_labels = labeled_mask.flatten()
    sizes = np.bincount(flat_labels)[1:]  # [1:] to exclude the background (label 0)
    if len(sizes) > 0:  # Check if there are any sizes to process
        max_size1 = sizes.max()
        max_label1 = sizes.argmax() + 1  # +1 to adjust for background label
        labeled_mask[labeled_mask == max_label1] = 0
        # Compute the size of the largest connected object in the updated mask
        max_size2, max_label2 = largest_connected_object_size(labeled_mask > 0)
    else:
        max_size1 = max_label1 = max_size2 = max_label2 = 0
        print("No objects found in the binary image.")
        
    if (np.sum(filled_holes_mask)>0.7*im_intensity.size)|(max_size1>3*max_size2):
        # flattened = im_intensity.flatten()
        # percentile_value = np.percentile(flattened, 98)
        # # Create a mask of the brightest pixels
        # brightest_pixels_mask = im_intensity >= percentile_value
        not_background_mask = np.logical_not(background_mask)
        ##### Remove less than 25 pixel objects
        filtered_image = morphology.remove_small_objects(not_background_mask, min_size=200)
        not_background_mask = filtered_image
        
        selem = morphology.disk(5)  # Adjust the size of the structuring element if needed
        
        # not_background_mask[0, :] = True      # Top row
        # not_background_mask[-1, :] = True     # Bottom row
        # not_background_mask[:, 0] = True      # Left column
        closed_holes_mask = morphology.binary_closing(not_background_mask, footprint=selem)
        # filled_holes_mask = ndimage.binary_fill_holes(closed_holes_mask)
        filled_holes_mask = fill_holes_like_imagej(closed_holes_mask)
        
        
        # filled_holes_mask[0, :] = False      # Top row
        # filled_holes_mask[-1, :] = False     # Bottom row
        # filled_holes_mask[:, 0] = False      # Left column
        # filled_holes_mask[:, -1] = True     # Right column
        # closed_holes_mask = morphology.binary_closing(filled_holes_mask, footprint=selem)
        # filled_holes_mask = ndimage.binary_fill_holes(closed_holes_mask)
        # filled_holes_mask = fill_holes_like_imagej(closed_holes_mask)
        # filled_holes_mask[0, :] = True      # Top row
        # filled_holes_mask[-1, :] = True     # Bottom row
        # filled_holes_mask[:, 0] = True      # Left column
        final_background_mask = np.logical_not(filled_holes_mask)
        # filled_holes_mask[0, :] = False      # Top row
        # filled_holes_mask[-1, :] = False     # Bottom row
        # filled_holes_mask[:, 0] = False      # Left column
        # filled_holes_mask[:, -1] = False     # Right column
        print("Kernel 5 closing filling done")
        
        # Detect if we have over masked or have large connected objects    
        labeled_mask, num_features = ndimage.label(filled_holes_mask)
        # sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
        flat_labels = labeled_mask.flatten()
        sizes = np.bincount(flat_labels)[1:]  # [1:] to exclude the background (label 0)
        if len(sizes) > 0:  # Check if there are any sizes to process
            max_size1 = sizes.max()
            max_label1 = sizes.argmax() + 1  # +1 to adjust for background label
            labeled_mask[labeled_mask == max_label1] = 0
            # Compute the size of the largest connected object in the updated mask
            max_size2, max_label2 = largest_connected_object_size(labeled_mask > 0)
        else:
            max_size1 = max_label1 = max_size2 = max_label2 = 0
            print("No objects found in the binary image.")
            
        if (np.sum(filled_holes_mask)>0.8*im_intensity.size)|(max_size1>4*max_size2):
            # flattened = im_intensity.flatten()
            # percentile_value = np.percentile(flattened, 98)
            # # Create a mask of the brightest pixels
            # brightest_pixels_mask = im_intensity >= percentile_value
            not_background_mask = np.logical_not(background_mask)
            ##### Remove less than 25 pixel objects
            filtered_image = morphology.remove_small_objects(not_background_mask, min_size=100)
            not_background_mask = filtered_image
            filled_holes_mask = not_background_mask
            final_background_mask = np.logical_not(not_background_mask)
            print("Zero closing was used")
        
    # If second closing and filling routine causes over masking or connected objects, revert to last result before this routine
    if np.mean(final_background_mask) < 0.3:
        # flattened = im_intensity.flatten()
        # percentile_value = np.percentile(flattened, 98)
        # # Create a mask of the brightest pixels
        # brightest_pixels_mask = im_intensity >= percentile_value
        not_background_mask = np.logical_not(background_mask)
        ##### Remove less than 25 pixel objects
        filtered_image = morphology.remove_small_objects(not_background_mask, min_size=100)
        not_background_mask = filtered_image
        final_background_mask = np.logical_not(not_background_mask)
    
    # Finally, remove small to medium sized background "objects". Background should be one monolithic connected object
    final_background_mask_temp = final_background_mask
    final_background_mask_temp[0, :] = True      # Top row
    final_background_mask_temp[-1, :] = True     # Bottom row
    final_background_mask_temp[:, 0] = True      # Left column
    final_background_mask_temp[:, -1] = True     # Right column
    
    filtered_image = morphology.remove_small_objects(final_background_mask_temp>0, min_size=200000, connectivity=2)
    
    if np.sum(filtered_image)!=np.sum(final_background_mask_temp):
        final_background_mask = filtered_image
        final_background_mask[0, :] = False      # Top row
        final_background_mask[-1, :] = False     # Bottom row
        final_background_mask[:, 0] = False      # Left column
        final_background_mask[:, -1] = False     # Right column
    
           
    
    print("")
    
    # #Visualization
    # rows = 3
    # cols = 3
    # fig, ax = plt.subplots(rows, cols, figsize=(20, 12))
       
    # ax[0, 0].imshow(np.logical_not(gradient_otsu_threshold), cmap='gray')
    # ax[0, 0].set_title(f"not(gradient_otsu_threshold): {base_name}")
    
    # ax[0, 1].imshow(low_gradient_mask, cmap='gray')
    # ax[0, 1].set_title('Gradient Mask Filled')
   
    # ax[0, 2].imshow(background_mask, cmap='gray')
    # ax[0, 2].set_title('Background Mask')
     
    # ax[1, 0].imshow(not_background_mask, cmap='gray')
    # ax[1, 0].set_title('Not Background Mask')
    
    # ax[1, 1].imshow(closed_holes_mask, cmap='gray')
    # ax[1, 1].set_title('Closed Holes Mask')
    
    # ax[1, 2].imshow(filled_holes_mask, cmap='gray')
    # ax[1, 2].set_title('Filled Holes Mask')
     
    # ax[2, 0].imshow(final_background_mask, cmap='gray')
    # ax[2, 0].set_title('Final Background Mask')
    
    # ax[2, 1].imshow(im_intensity)
    # ax[2, 1].set_title('Intensity Image')
        
    # ax[2, 2].set_axis_off()
    # ax[2, 2].set_title(f"{if_state}")
    
    # plt.show()
    
    #Visualization
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
    
    # Check if the number of True pixels in combined_mask is less than 10% of total pixels
    total_pixels = combined_mask.size
    true_pixels_combined = np.sum(combined_mask)
    
    if true_pixels_combined < 0.4 * total_pixels:
        # Select the mask with more True pixels
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

#%% Load set/dictionaries and compute omi parameters
    # load csv dicts with path sets
list_path_csv_image_sets = list((path_output / "dictionaries").glob("*_omi.csv"))

analysis_type = "whole_cell"
leading_edge = True
for dict_path in list_path_csv_image_sets[:1]:  # iterate through csv files
    pass
    print(f"processing: {dict_path.stem}")
    df_image_set = pd.read_csv(dict_path)
    df_image_set = df_image_set.set_index("base_name", drop=True)

    # keep running list of dataframes
    # df_all_dicts = df_image_set if df_all_dicts is None else df_all_dicts.append(df_image_set)

    # iterate through rows(image sets) in dataframe,
    for base_name, row_data in tqdm(
        list(df_image_set.iterrows())
    ):  # iterate through sets in csv file
        pass
        
        # load mask based on analysis type
        if analysis_type == "whole_cell":
            mask_path = Path(row_data.mask_cell)
            mask = load_image(mask_path)  # whole cell
        # elif analysis_type == "cytoplasm":
        #     mask_path = Path(row_data.mask_cytoplasm)
        #     mask = load_image(Path(row_data.mask_cytoplasm))  # cytoplasm
        # elif analysis_type == "nuclei":
        #     mask_path = Path(row_data.mask_nuclei)
        #     mask = load_image(Path(row_data.mask_nuclei))  # nuclei
            
            
        if leading_edge:
            leading_edge_mask = create_leading_edge_mask(mask, mask_path=mask_path)  # create leading edge

        # load images
        im_nadh_intensity = load_image(Path(row_data.nadh_photons))
        
        # im_nadh_a1 = load_image(Path(row_data.nadh_a1))
        # im_nadh_a2 = load_image(Path(row_data.nadh_a2))
        # im_nadh_t1 = load_image(Path(row_data.nadh_t1))
        # im_nadh_t2 = load_image(Path(row_data.nadh_t2))
        # im_nadh_intensity = load_image(Path(row_data.nadh_photons)); im_nadh_intensity = np.ma.masked_array(im_nadh_intensity, mask=im_nadh_intensity<5)
        ## im_nadh_a1 = load_image(Path(row_data.nadh_a1));  # temporarily load for masking purposes!
        ## im_nadh_intensity = load_image(Path(row_data.nadh_photons)); im_nadh_intensity = np.ma.masked_array(im_nadh_intensity, mask=im_nadh_a1==0)
        # im_nadh_a1 = load_image(Path(row_data.nadh_a1)); im_nadh_a1 = np.ma.masked_array(im_nadh_a1, mask=im_nadh_a1==0)
        # im_nadh_a2 = load_image(Path(row_data.nadh_a2)); im_nadh_a2 = np.ma.masked_array(im_nadh_a2, mask=im_nadh_a2==0)
        # im_nadh_t1 = load_image(Path(row_data.nadh_t1)); im_nadh_t1 = np.ma.masked_array(im_nadh_t1, mask=im_nadh_t1==0)
        # im_nadh_t2 = load_image(Path(row_data.nadh_t2)); im_nadh_t2 = np.ma.masked_array(im_nadh_t2, mask=im_nadh_t2==0)
        
        # # Comment the following FAD features for NADH-only data
        im_fad_intensity = load_image(Path(row_data.fad_photons))
        # # im_fad_a1 = load_image(Path(row_data.fad_a1))
        # # im_fad_a2 = load_image(Path(row_data.fad_a2))
        # # im_fad_t1 = load_image(Path(row_data.fad_t1))
        # # im_fad_t2 = load_image(Path(row_data.fad_t2))
        # ## im_fad_a1 = load_image(Path(row_data.fad_a1));  # temporarily load for masking purposes!
        # ## im_fad_intensity = load_image(Path(row_data.fad_photons)); im_fad_intensity = np.ma.masked_array(im_fad_intensity, mask=im_fad_a1==0)
        # im_fad_a1 = load_image(Path(row_data.fad_a1)); im_fad_a1 = np.ma.masked_array(im_fad_a1, mask=im_fad_a1==0)
        # im_fad_a2 = load_image(Path(row_data.fad_a2)); im_fad_a2 = np.ma.masked_array(im_fad_a2, mask=im_fad_a2==0)
        # im_fad_t1 = load_image(Path(row_data.fad_t1)); im_fad_t1 = np.ma.masked_array(im_fad_t1, mask=im_fad_t1==0)
        # im_fad_t2 = load_image(Path(row_data.fad_t2)); im_fad_t2 = np.ma.masked_array(im_fad_t2, mask=im_fad_t2==0)

        # compute ROI props
        omi_props = regionprops_omi(
            image_id=base_name,
            label_image=mask,  ### this selects what mask to summarize
            im_nadh_intensity=im_nadh_intensity,
            # im_nadh_a1=im_nadh_a1,
            # im_nadh_a2=im_nadh_a2,
            # im_nadh_t1=im_nadh_t1,
            # im_nadh_t2=im_nadh_t2,
            # # Comment the following FAD features for NADH-only data
            im_fad_intensity=im_fad_intensity,
            # im_fad_a1=im_fad_a1,
            # im_fad_a2=im_fad_a2,
            # im_fad_t1=im_fad_t1,
            # im_fad_t2=im_fad_t2,
            # other morphological features calculated from the binary mask image
            other_props=['area', 'perimeter', 'solidity', 'eccentricity', 'axis_major_length', 'axis_minor_length']
        )

        ## create dataframe
        df = pd.DataFrame(omi_props).transpose()
     #   df.index.name = "base_name"
     

        
        # Calculate whole cell NADH/FAD background, normalized intensity, redox ratio, and normalized redox ratio
        nadh_final_background_mask = make_background_mask(im_nadh_intensity, mask)
        fad_final_background_mask = make_background_mask(im_fad_intensity, mask)
        
            
          
        # Combine the final masks
        final_combined_mask = combine_masks(nadh_final_background_mask, fad_final_background_mask)
        # plt.imshow(final_combined_mask, cmap='gray')
        # plt.title('Final Combined Background Mask')
        # plt.show()
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
            background_mask_path_nadh =str(mask_path).replace(mask_suffix, "_backgndmask_NADH.tiff")
            success = cv2.imwrite(background_mask_path_nadh, (nadh_final_background_mask * 255).astype(np.uint8))
            background_mask_path_fad =str(mask_path).replace(mask_suffix, "_backgndmask_FAD.tiff")
            success = cv2.imwrite(background_mask_path_fad, (fad_final_background_mask * 255).astype(np.uint8))
            background_mask_path_final =str(mask_path).replace(mask_suffix, "_backgndmask_Final.tiff")
            success = cv2.imwrite(background_mask_path_final, (final_combined_mask * 255).astype(np.uint8))
    
        ## add other dictionary data to df
            df["base_name"] = base_name
            for item_key in row_data.keys():
                df[item_key] = row_data[item_key]
            
            
            if leading_edge:
                # Extract leading edge mask intensity values
                leading_edge_nadh_intensity_values = im_nadh_intensity[leading_edge_mask == 1]
                leading_edge_fad_intensity_values = im_fad_intensity[leading_edge_mask == 1]
                
                # Compute mean intensity for NADH and FAD within the leading edge mask
                leading_edge_nadh_intensity_mean = np.mean(leading_edge_nadh_intensity_values)
                leading_edge_fad_intensity_mean = np.mean(leading_edge_fad_intensity_values)
    
                leading_edge_omi_props = regionprops_omi(
                    image_id=base_name,
                    label_image=leading_edge_mask,  ### this selects what mask to summarize
                    im_nadh_intensity=im_nadh_intensity,
                    # im_nadh_a1=im_nadh_a1,
                    # im_nadh_a2=im_nadh_a2,
                    # im_nadh_t1=im_nadh_t1,
                    # im_nadh_t2=im_nadh_t2,
                    # # Comment the following FAD features for NADH-only data
                    im_fad_intensity=im_fad_intensity,
                    # im_fad_a1=im_fad_a1,
                    # im_fad_a2=im_fad_a2,
                    # im_fad_t1=im_fad_t1,
                    # im_fad_t2=im_fad_t2,
                    # other morphological features calculated from the binary mask image
                    other_props=['area']
                )
                leading_edge_df = pd.DataFrame(leading_edge_omi_props).transpose()
        #        leading_edge_df.index.name = "base_name"
    
                ## add other dictionary data to df
                leading_edge_df["base_name"] = base_name
                for item_key in row_data.keys():
                    leading_edge_df[item_key] = row_data[item_key]
                    
                
                if not leading_edge_df.empty:
                    leading_edge_df = leading_edge_df.rename(columns={'nadh_intensity_mean':'leading_edge_nadh_intensity_mean', 'fad_intensity_mean': 'leading_edge_fad_intensity_mean', 'area': 'leading_edge_area'})
                    df = pd.merge(df, leading_edge_df[['leading_edge_nadh_intensity_mean', 'leading_edge_fad_intensity_mean', 'leading_edge_area', 'mask_label']], on="mask_label", how='outer')
                    df['leading_edge_nadh_intensity_norm'] = df['leading_edge_nadh_intensity_mean'] / df['nadh_background']
                    df['leading_edge_fad_intensity_norm'] = df['leading_edge_fad_intensity_mean'] / df['fad_background']
                    df['leading_edge_redox_ratio'] = df['leading_edge_nadh_intensity_mean'] / (df['leading_edge_nadh_intensity_mean'] + df['leading_edge_fad_intensity_mean'])
                    df['leading_edge_normalized_redox_ratio'] = df['leading_edge_nadh_intensity_norm'] / (df['leading_edge_nadh_intensity_norm'] + df['leading_edge_fad_intensity_norm'])
            
                #Reorder columns
                first_column_name = df.columns[0]
                desired_order = [
                    'base_name',
                    #'mask_label',
                    'leading_edge_normalized_redox_ratio',
                    'nadh_intensity_mean',
                    'nadh_intensity_stdev',
                    'fad_intensity_mean',
                    'fad_intensity_stdev',
                    'area',
                    'perimeter',
                    'solidity',
                    'eccentricity',
                    'axis_major_length',
                    'axis_minor_length',
                    'nadh_background',
                    'fad_background',
                    'nadh_intensity_norm',
                    'fad_intensity_norm',
                    'redox_ratio',
                    'normalized_redox_ratio',
                    'leading_edge_nadh_intensity_mean',
                    'leading_edge_fad_intensity_mean',
                    'leading_edge_area',
                    'leading_edge_nadh_intensity_norm',
                    'leading_edge_fad_intensity_norm',
                    'leading_edge_redox_ratio',
                    'leading_edge_normalized_redox_ratio',
                    'mask_cell',
                    'PDCO_line',
                    'cancer',
                    'treatment',
                    'fad_photons',
                    'nadh_photons',
                    'resolution',
                ]
                
                
                #df = df.drop(columns=['redox_ratio_mean', 'redox_ratio_stdev', 'redox_ratio_norm_mean', 'redox_ratio_norm_stdev'])
        
                missing_columns = [col for col in desired_order if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing columns in DataFrame: {missing_columns}")
                
                # Reorder the DataFrame columns, including the first column
                df_reordered = df[[first_column_name] + desired_order]
                
                # Export the reordered DataFrame to a CSV file
                output_file = path_output / "features" / f"features_{base_name}_{analysis_type}.csv"
                df_reordered.to_csv(output_file)
                # finally export
                #df.to_csv(
                #    path_output / "features" / f"features_{base_name}_{analysis_type}.csv"
                #)
        except KeyError as e:
            if str(e) == "'nadh_intensity_mean'":
                print(f"Skipping {base_name} due to missing 'nadh_intensity_mean'")
                continue  # Skip the rest of the current iteration
    
    
path_output_props = path_output / "summary"
list_path_features_csv = list((path_output / "features").glob("*.csv"))

df_all_props = None
for path_feat_csv in tqdm(list_path_features_csv):
    pass
    # initialize first df entry else append to running df
    df_omi = pd.read_csv(path_feat_csv)
    df_all_props = (
        df_omi 
        if df_all_props is None
        else pd.concat([df_all_props, df_omi], ignore_index=True)
    )

# label index of complete dictionary
df_all_props = df_all_props.set_index("base_name", drop=True)
d = date.today().strftime("%Y_%m_%d")
df_all_props.to_csv(path_output_props / f"{d}_all_props.csv")
