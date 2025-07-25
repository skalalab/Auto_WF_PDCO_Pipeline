import glob
import matplotlib.pyplot as plt
import tifffile as tiff
import os
from pathlib import Path
import cv2
import numpy as np
import math
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion
from scipy.ndimage import shift
from scipy.spatial import KDTree
from skimage import transform
from scipy.spatial import distance

def load_mask_pairs(directory):
    # Find all day 0, day 1, and day 2 files
    day_0_files = glob.glob(os.path.join(directory, '*Day 0*.tif*'))
    day_1_files = glob.glob(os.path.join(directory, '*Day 1*.tif*'))
    day_2_files = glob.glob(os.path.join(directory, '*Day 2*.tif*'))
    
    # Create dictionaries to store both the original and cleaned basenames
    day_0_basenames = {}
    day_1_basenames = {}
    day_2_basenames = {}

    # Populate dictionaries with original and cleaned basenames for each day
    for f in day_0_files:
        original_basename = os.path.basename(f)
        cleaned_basename = original_basename.replace('Day 0', '')
        day_0_basenames[cleaned_basename] = (f, original_basename)

    for f in day_1_files:
        original_basename = os.path.basename(f)
        cleaned_basename = original_basename.replace('Day 1', '')
        day_1_basenames[cleaned_basename] = (f, original_basename)

    for f in day_2_files:
        original_basename = os.path.basename(f)
        cleaned_basename = original_basename.replace('Day 2', '')
        day_2_basenames[cleaned_basename] = (f, original_basename)

    # Create the pairs dictionary with exactly six elements
    pairs = {}

    for cleaned_basename in day_0_basenames.keys():
        # Load Day 0 image and original basename
        day_0_image = tiff.imread(day_0_basenames[cleaned_basename][0]).astype(np.uint8)
        original_basename_day_0 = day_0_basenames[cleaned_basename][1]

        # Load Day 1 image and basename
        day_1_image = tiff.imread(day_1_basenames[cleaned_basename][0]).astype(np.uint8)
        original_basename_day_1 = day_1_basenames[cleaned_basename][1]

        # Load Day 2 image and basename
        day_2_image = tiff.imread(day_2_basenames[cleaned_basename][0]).astype(np.uint8)
        original_basename_day_2 = day_2_basenames[cleaned_basename][1]

        # Store in pairs with exactly six elements
        pairs[cleaned_basename] = (
            day_0_image, day_1_image, day_2_image,
            original_basename_day_0, original_basename_day_1, original_basename_day_2
        )

    return pairs



def register_and_translate_masks(mask_day0, mask_day4, translation=None, reverse=False):
    """
    Registers mask_day4 to mask_day0 using cross-correlation maximization, or undoes the translation if reverse is True.
    
    Parameters:
    - mask_day0: The Day 0 mask (used for registering or reference). Required if reverse=False.
    - mask_day4: The Day 4 mask to be registered or translated.
    - translation: (Optional) Translation parameters (tx, ty). Required if reverse=True.
    - reverse: If True, undo the translation instead of applying it.
    
    Returns:
    - transformed_mask_day4: The registered or unregistered (translated back) mask.
    - translation: The translation parameters applied to align Day 4 to Day 0.
    """
    
    def apply_translation(image, translation, reverse=False):
        """Apply or undo translation on grayscale or color images using nearest neighbor interpolation."""
        tx, ty = translation
        if reverse:  # Invert the translation if undoing
            tx, ty = -tx, -ty
    
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
        if len(image.shape) == 3:  # Color image
            channels = cv2.split(image)
            transformed_channels = [
                cv2.warpAffine(channel, translation_matrix, (channel.shape[1], channel.shape[0]), flags=cv2.INTER_NEAREST)
                for channel in channels
            ]
            return cv2.merge(transformed_channels)
        else:  # Grayscale image
            return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
    
    def find_best_translation(mask_ref, mask_to_align):
        """Find the best translation using cross-correlation maximization with a restricted search range."""
        # Pad the reference mask to prevent the result from being too small
        pad_x = mask_to_align.shape[1] // 2
        pad_y = mask_to_align.shape[0] // 2

        padded_mask_ref = np.pad(mask_ref, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)

        # Compute cross-correlation between the reference and the mask to align
        result = cv2.matchTemplate(
            padded_mask_ref.astype(np.float32), 
            mask_to_align.astype(np.float32), 
            cv2.TM_CCORR_NORMED
        )

        # Define search range (10% of frame width and height)
        search_range_x = int(mask_ref.shape[1] * 0.25)
        search_range_y = int(mask_ref.shape[0] * 0.25)

        # Center of the cross-correlation map
        center_x = result.shape[1] // 2
        center_y = result.shape[0] // 2

        # Define the local search window around the center of the cross-correlation map
        x_min = max(center_x - search_range_x, 0)
        x_max = min(center_x + search_range_x, result.shape[1])
        y_min = max(center_y - search_range_y, 0)
        y_max = min(center_y + search_range_y, result.shape[0])

        # Extract the local region for searching
        local_region = result[y_min:y_max, x_min:x_max]

        # Find the maximum in the local region
        local_max_loc = np.unravel_index(np.argmax(local_region), local_region.shape)

        # Adjust local max location to global coordinates
        adjusted_max_loc = (x_min + local_max_loc[1], y_min + local_max_loc[0])

        # Calculate translation from the adjusted max location
        max_translation = np.array(adjusted_max_loc) - np.array(mask_ref.shape[::-1]) // 2

        return (max_translation[0], max_translation[1])
    
    # Ensure the masks are NumPy arrays and binarize for cross-correlation
    mask_day0 = np.asarray(mask_day0)
    mask_day4 = np.asarray(mask_day4)
   
    binarized_mask_day0 = (mask_day0 != 0).astype(np.uint8) * 255
    binarized_mask_day4 = (mask_day4 != 0).astype(np.uint8) * 255
    
    if not reverse:
        # Registering: Calculate translation based on cross-correlation maximization
        translation = find_best_translation(binarized_mask_day0, binarized_mask_day4)
    elif reverse and translation is None:
        raise ValueError("Translation parameters must be provided when reverse=True.")
    
    # Apply translation (or undo it)
    transformed_mask_day4 = apply_translation(mask_day4, translation, reverse=reverse)
    
    return transformed_mask_day4, translation

def calculate_minimum_distances(mask):
    """
    Calculates the minimum distances between the perimeters of objects in a mask.
    
    Parameters:
    - mask: A 2D numpy array where each object has a unique label.
    
    Returns:
    - A list of tuples (label1, label2, distance) representing the minimum distances
      between each pair of objects.
    """
    distances = []
    
    # Get unique object labels (excluding background labeled as 0)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Precompute the perimeters of all objects
    perimeters = {}
    for label in unique_labels:
        object_mask = (mask == label).astype(np.uint8)
        dilated_object = cv2.dilate(object_mask, np.ones((3,3), np.uint8), iterations=1)
        perimeter = dilated_object - object_mask  # Extract perimeter pixels
        perimeter_points = np.column_stack(np.where(perimeter == 1))  # Get (row, col) points
        perimeters[label] = perimeter_points
    
    # Compare each pair of objects and compute the minimum distance
    for i, label1 in enumerate(unique_labels):
        perimeter1 = perimeters[label1]
        
        # Build a KD-Tree for the perimeter points of object 1
        tree1 = KDTree(perimeter1)
        
        for label2 in unique_labels[i+1:]:
            perimeter2 = perimeters[label2]
            
            # Find the nearest neighbor distances from perimeter2 to perimeter1
            distances_to_label1, _ = tree1.query(perimeter2)
            
            # Find the minimum distance
            min_distance = np.min(distances_to_label1)
            
            # Store the distance between these two objects
            distances.append((label1, label2, min_distance))
    
    return distances

def filter_close_objects(mask, distances, threshold_percentage=0.005):
    
    distances = sorted(distances, key=lambda x: x[2])  # Sort by distance (x[2] is the distance value)
    
    # Get the dimensions of the mask
    img_height, img_width = mask.shape
    
    # Define the threshold as a percentage of the image size
    threshold = threshold_percentage * min(img_width, img_height)
    
    # Create a flag to determine if the mask was updated
    updated = True
    
    while updated:
        updated = False
        # Iterate through the distances and eliminate objects too close together
        for label1, label2, distance in distances:
            if distance < threshold:
                obj1_size = np.sum(mask == label1)
                obj2_size = np.sum(mask == label2)
                
                # Eliminate the smaller object by setting its label to 0 (background)
                if obj1_size < obj2_size:
                    mask[mask == label1] = 0
                else:
                    mask[mask == label2] = 0
                
                # Set the flag to indicate the mask was updated
                updated = True
                # Break the loop to recalculate distances with the updated mask
                break

        if updated:
            # Recalculate distances with the updated mask
            distances = calculate_minimum_distances(mask)
            distances = sorted(distances, key=lambda x: x[2])  # Sort by distance (x[2] is the distance value)
    
    return mask

def normalize_region_ids(search_region, base_id=255):
    """
    Renormalizes object IDs in the search_region to large numbers starting from base_id.
    """
    unique_ids = np.unique(search_region)
    unique_ids = unique_ids[unique_ids != 0]  # Exclude background (0)
    
    # Create a mapping from original IDs to new large IDs starting from base_id
    id_mapping = {orig_id: base_id - i for i, orig_id in enumerate(unique_ids)}
    
    # Apply the mapping to the search_region
    normalized_region = search_region.copy()
    for orig_id, new_id in id_mapping.items():
        normalized_region[search_region == orig_id] = new_id
    
    return normalized_region

def find_matching_objects(img1, img2, search_window_frac = 0.02):
    matches = []
    min_score_threshold = 0.2  # Minimum cross-correlation threshold
    max_score_threshold = 0.5  # Maximum cross-correlation threshold
    

    # Extract unique labels from the first image (excluding background labeled as 0)
    unique_labels_img1 = np.unique(img1)
    unique_labels_img1 = unique_labels_img1[unique_labels_img1 != 0]

    # Sort objects in img1 by size in descending order (larger objects first)
    object_sizes = {label: np.sum(img1 == label) for label in unique_labels_img1}
    sorted_labels_img1 = sorted(unique_labels_img1, key=lambda l: object_sizes[l], reverse=True)

    # Define search window sizes as percentages of the image size
    # initial_search_window = 0.05  # 10% of the image size
    # secondary_search_window = 0.20  # 20% of the image size

    for label1 in sorted_labels_img1:
        if label1 == 0:
            continue
        
        # Create a binary mask for the current object in img1
        mask1 = (img1 == label1).astype(np.uint8) * 255
        
        # Find contours in the binary mask (img1 object)
        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour1 in contours1:
            x, y, w, h = cv2.boundingRect(contour1)
            template = mask1[y:y+h, x:x+w]
            
            # Define the initial search window region around the object in img2
            search_window_h = int(search_window_frac * img2.shape[0])
            search_window_w = int(search_window_frac * img2.shape[1])
            top = max(0, y - search_window_h)
            bottom = min(img2.shape[0], y + h + search_window_h)
            left = max(0, x - search_window_w)
            right = min(img2.shape[1], x + w + search_window_w)
            
            # Extract the search region from img2
            search_region = img2[top:bottom, left:right]
            
            search_region = normalize_region_ids(search_region)
            
            # Iterate over each unique label in the search region
            unique_labels_search_region = np.unique(search_region)
            unique_labels_search_region = unique_labels_search_region[unique_labels_search_region != 0]
            
            best_score = -np.inf
            best_match_location = None
            best_label2 = None
            
            for label2 in unique_labels_search_region:
                # Create a binary mask for the current object in the search region
                mask2 = (search_region == label2).astype(np.uint8) * 255
                
                # Perform template matching within the search region using template
                result = cv2.matchTemplate(mask2, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                # Track the best score and corresponding location
                if max_val > best_score:
                    best_score = max_val
                    best_match_location = max_loc
                    best_label2 = label2
            
            # Only proceed if the best score meets or exceeds the threshold
            score_threshold = min_score_threshold + (max_score_threshold - min_score_threshold) * (10 * np.sum(img1 == label1) / (img1.shape[0] * img1.shape[1]))
            
            if best_score >= score_threshold and best_match_location is not None:
                # Adjust match_location to global coordinates in img2
                match_x = left + best_match_location[0]
                match_y = top + best_match_location[1]
                
                # Get the bounding box of the matched area in img2
                match_box = img2[match_y:match_y+h, match_x:match_x+w]
                
                # Find the most common label in the matched box (ignoring background 0)
                match_labels, match_counts = np.unique(match_box[match_box != 0], return_counts=True)
                if len(match_labels) > 0:
                    matched_label = match_labels[np.argmax(match_counts)]  # Most frequent label
                
                    # Store the match details
                    matches.append((label1, matched_label, (x, y, w, h), (match_x, match_y), best_score, 0))  # Distance = 0
                
                    # Remove the matched object from img2 to prevent duplicate matches
                    img2[img2 == matched_label] = 0
    
    return matches

def merge_touching_objects(mask, perimeter_threshold_fraction=0.40, size_threshold_frac=0.010):
    """
    Merges touching objects in the mask by examining the boundary pixels of each object.
    The object that touches another object the most in terms of boundary pixels is merged
    into the larger object, only if the number of touching pixels exceeds a threshold
    defined as a fraction of the object's perimeter, and the object sizes are below a threshold.
    
    Parameters:
    - mask: A 2D numpy array where each object has a unique label.
    - perimeter_threshold_fraction: The fraction of an object's perimeter required for merging.
    - size_threshold_frac: The fraction of the total number of pixels to determine the size threshold for merging.
    
    Returns:
    - A modified mask where touching objects have been merged.
    """
    
    # mask = remove_small_objects(mask, min_size=1000)
    
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    label_sizes = [(label, np.sum(mask == label)) for label in unique_labels]
    sorted_labels = [label for label, _ in sorted(label_sizes, key=lambda x: x[1], reverse=True)]
    
    for label in sorted_labels:
        object_mask = (mask == label).astype(np.uint8)
        
        # Find contours of the current object to identify its perimeter
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        
        # Create a blank mask for drawing the contour (perimeter)
        perimeter_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(perimeter_mask, contours, -1, 1, 2)  # Draw contours with thickness 2
        # Remove pixels that are part of the object itself
        perimeter_mask = perimeter_mask & (~object_mask)
        
        # Identify neighboring object labels by looking at pixels just outside the contour
        neighboring_labels = mask[perimeter_mask == 1]
        neighboring_labels = neighboring_labels[(neighboring_labels != 0) & (neighboring_labels != label)]
        
        if len(neighboring_labels) == 0:
            continue
        
        # Find the most common neighboring label (the one this object touches the most)
        unique, counts = np.unique(neighboring_labels, return_counts=True)
        most_frequent_neighbor_label = unique[np.argmax(counts)]
        
        # Create a mask for the neighbor's perimeter
        neighbor_object_mask = (mask == most_frequent_neighbor_label).astype(np.uint8)
        neighbor_contours, _ = cv2.findContours(neighbor_object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        neighbor_perimeter_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(neighbor_perimeter_mask, neighbor_contours, -1, 1, 1)
        # # Remove pixels that are part of the neighbor object itself
        # neighbor_perimeter_mask = neighbor_perimeter_mask - neighbor_object_mask
        
        # Check if the two objects are touching by checking the overlap of perimeter pixels
        touching_pixels = np.sum((perimeter_mask == 1) & (neighbor_perimeter_mask == 1))
        if touching_pixels == 0:
            continue
        
        # Get sizes of the current object and the neighboring object
        current_object_size = np.sum(mask == label)
        neighbor_object_size = np.sum(mask == most_frequent_neighbor_label)
        
        size_threshold = size_threshold_frac * (mask.shape[0] * mask.shape[1])
        if current_object_size > size_threshold and neighbor_object_size > size_threshold:
            continue
        
        # Calculate the perimeter of the current object
        current_object_perimeter = cv2.arcLength(contours[0], True)
        
        # Calculate the perimeter of the neighboring object
        neighbor_object_perimeter = cv2.arcLength(neighbor_contours[0], True)
        
        # Calculate the threshold for merging based on the perimeter
        threshold = perimeter_threshold_fraction * min(current_object_perimeter, neighbor_object_perimeter)
        
        # Merge the objects if the touching pixel count exceeds the threshold
        if touching_pixels > threshold:
            if current_object_size < neighbor_object_size:
                mask[mask == label] = most_frequent_neighbor_label
            else:
                mask[mask == most_frequent_neighbor_label] = label
    
    return mask


def find_matching_objects_with_distance_filtering(mask1, mask2):
    mask1 = merge_touching_objects(mask1)
    mask2 = merge_touching_objects(mask2)
  
    ### Filter out objects that are too close to each other
    # mask1_filtered = filter_close_objects(mask1, distances)
    ### Do not filter out objects that are too close to each other
    mask1_filtered = np.copy(mask1)
    
    mask2_filtered = np.copy(mask2)
    
    # Proceed with finding matches using the filtered mask1
    matches = find_matching_objects(mask1, mask2)
    
    return mask1_filtered, mask2_filtered, matches

def update_mask_with_matches(original_mask1, original_mask2, matches):
    """
    Updates the masks with new unique identifiers for matched objects.

    Parameters:
    - original_mask1: The original mask containing objects from the first image.
    - original_mask2: The original mask containing objects from the second image.
    - matches: The list of matches returned by the `find_matching_objects` function.

    Returns:
    - updated_mask1: The updated mask for the first image.
    - updated_mask2: The updated mask for the second image.
    """
    
    # Create new masks with the same dimensions as the original masks
    updated_mask1 = np.zeros_like(original_mask1, dtype=np.uint8)
    updated_mask2 = np.zeros_like(original_mask2, dtype=np.uint8)
    
    # Initialize a unique identifier counter
    new_id_counter = 2
    
    # Iterate through matches to update masks
    for label1, matched_label, (obj_x, obj_y, obj_w, obj_h), (match_x, match_y),best_score, best_distance in matches:
        
        # Extract object mask from the original masks
        object_mask1 = (original_mask1 == label1).astype(np.uint8)
        object_mask2 = (original_mask2 == matched_label).astype(np.uint8)
        
        if np.any(object_mask2):  # Ensure that the matched object exists in original_mask2
            # Create a new identifier for the matched objects
            new_id = new_id_counter
            new_id_counter += 1
            
            # Assign new IDs to the objects in both masks
            object_mask1_with_new_id = object_mask1 * new_id
            object_mask2_with_new_id = object_mask2 * new_id
            
            # Update the new masks with the new object IDs
            updated_mask1 = np.maximum(updated_mask1, object_mask1_with_new_id)
            updated_mask2 = np.maximum(updated_mask2, object_mask2_with_new_id)
    
    return updated_mask1, updated_mask2

def draw_matches(img1, img2, matches):
    img1_colored = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_colored = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    for label, _, (x, y, obj_w, obj_h), (match_x, match_y), best_score, best_distance in matches:
        print(f"Match between label {label}: Cross-correlation score = {best_score}, Distance = {best_distance}")
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(img1_colored, (x, y), (x + obj_w, y + obj_h), color, 3)
        cv2.putText(img1_colored, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.rectangle(img2_colored, (match_x, match_y), (match_x + obj_w, match_y + obj_h), color, 3)
        cv2.putText(img2_colored, str(label), (match_x, match_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    return img1_colored, img2_colored

def remove_small_objects(mask, min_size=1000):
    """
    Remove small objects or rare disjointed pixels from a labeled mask based on the size threshold.

    Args:
        mask (np.ndarray): Labeled mask where each unique value represents a different object.
        min_size (int): Minimum size of objects to keep. Smaller objects will be removed.

    Returns:
        np.ndarray: Mask with small objects removed.
    """
    # Get unique labels and their pixel counts
    unique_labels, counts = np.unique(mask, return_counts=True)

    # Identify labels to remove
    labels_to_remove = set(unique_labels[counts < min_size])

    # Remove small objects by setting pixels to 0
    filtered_mask = np.where(np.isin(mask, list(labels_to_remove)), 0, mask)

    return filtered_mask

def consolidate_ids(mask1, mask2):
    """
    Ensures consistency of IDs between two masks by resolving overlaps.

    Returns:
    - consolidated_mask: The consolidated version of mask2 with IDs from mask1 (not likely to be used ever).
    - id_mappings: A dictionary mapping IDs in mask2 (which is the refernce) to (old) IDs from mask1.
    """
    id_mappings = {}
    consolidated_mask = np.zeros_like(mask2)

    unique_ids_mask1 = np.unique(mask1[mask1 > 0])
    unique_ids_mask2 = np.unique(mask2[mask2 > 0])

    for id_mask2 in unique_ids_mask2:
        overlapping_ids = mask1[mask2 == id_mask2]
        overlapping_ids = overlapping_ids[overlapping_ids > 0]

        if len(overlapping_ids) > 0:
            most_common_id = np.bincount(overlapping_ids).argmax()
            id_mappings[id_mask2] = most_common_id
            consolidated_mask[mask2 == id_mask2] = most_common_id

    return consolidated_mask, id_mappings

def propagate_ids(mask, id_mappings):
    """
    Updates the IDs in the mask based on a given ID mapping.
    """
    updated_mask = np.zeros_like(mask)

    for new_id, old_id in id_mappings.items():
        updated_mask[mask == old_id] = new_id

    return updated_mask

def process_mask_pairs_with_distance_filtering(mask_pairs, base_directory):
    # Create a new directory named "new masks" inside the base directory
    output_directory = Path(base_directory) / "new masks"
    output_directory.mkdir(exist_ok=True)
    
    for cleaned_basename, (mask_day_0, mask_day_1, mask_day_2, original_basename_day_0, original_basename_day_1, original_basename_day_2) in mask_pairs.items():
        # Remove small objects based on size threshold
        mask_day_0 = remove_small_objects(mask_day_0, min_size=1000)
        mask_day_1 = remove_small_objects(mask_day_1, min_size=1000)
        mask_day_2 = remove_small_objects(mask_day_2, min_size=1000)
        
        # Make copies before processing
        mask_day_0_orig = np.copy(mask_day_0)
        mask_day_1_orig = np.copy(mask_day_1)
        mask_day_2_orig = np.copy(mask_day_2)
        
        # Register Day 0 to Day 1 (middle day) and filter matched objects
        mask_day_0_registered, translation_day_0 = register_and_translate_masks(mask_day_1, mask_day_0)
        mask_day_1_filtered, mask_day_0_filtered, matches_day_1_to_day_0 = find_matching_objects_with_distance_filtering(mask_day_1, mask_day_0_registered)
        # only keep matched objects in both masks
        mask_day_1_intermediate, mask_day_0_intermediate = update_mask_with_matches(mask_day_1_filtered, mask_day_0_filtered, matches_day_1_to_day_0)
                        
        
        # Register Day 2 to the filtered Day 1 (already matched to Day 0)
        mask_day_2_registered, translation_day_2 = register_and_translate_masks(mask_day_1_filtered, mask_day_2)
        mask_day_1_filtered_new, mask_day_2_filtered, matches_day_1_to_day_2 = find_matching_objects_with_distance_filtering(mask_day_1_intermediate, mask_day_2_registered)
        # only keep matched objects in both masks
        mask_day_1_final, new_mask_day_2 = update_mask_with_matches(mask_day_1_filtered_new, mask_day_2_filtered, matches_day_1_to_day_2)

        # Reverse translation on Day 2 to undo the registration offset
        mask_day_2_final, _ = register_and_translate_masks(None, new_mask_day_2, translation=translation_day_2, reverse=True)
        
        # Consolidate IDs between new day 1 mask and intermediate day 1 mask and propagate to day 0 mask
        _, id_mappings_day1 = consolidate_ids(mask_day_1_intermediate, mask_day_1_final)
        new_mask_day_0 = propagate_ids(mask_day_0_intermediate, id_mappings_day1)
            
        # Reverse translation on Day 0 to undo the registration offset
        mask_day_0_final, _ = register_and_translate_masks(None, new_mask_day_0, translation=translation_day_0, reverse=True)
        
        
        # Draw matches on Day 1 and Day 2 images
        result_image_day_1, result_image_day_0 = draw_matches(mask_day_1_orig, mask_day_0_registered, matches_day_1_to_day_0)
        result_image_day_0, _ = register_and_translate_masks(None, result_image_day_0, translation=translation_day_0, reverse=True)
        
        result_image_day_1_filtered, result_image_day_2 = draw_matches(mask_day_1_orig, mask_day_2_registered, matches_day_1_to_day_2)
        result_image_day_2, _ = register_and_translate_masks(None, result_image_day_2, translation=translation_day_2, reverse=True)
        
        # Define output paths for saving the filtered masks
        basename_output_day_0 = original_basename_day_0.replace("_cellpose_mask", "_filtered").replace("__", "_")
        basename_output_day_1 = original_basename_day_1.replace("_cellpose_mask", "_filtered").replace("__", "_")
        basename_output_day_2 = original_basename_day_2.replace("_cellpose_mask", "_filtered").replace("__", "_")
        
        output_path_mask_day_0 = output_directory / f"{basename_output_day_0}"
        output_path_mask_day_1 = output_directory / f"{basename_output_day_1}"
        output_path_mask_day_2 = output_directory / f"{basename_output_day_2}"
        
        # Save the filtered masks
        tiff.imwrite(str(output_path_mask_day_0), mask_day_0_final)
        tiff.imwrite(str(output_path_mask_day_1), mask_day_1_final)
        tiff.imwrite(str(output_path_mask_day_2), mask_day_2_final)
        
        # Plot and show images
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0, 0].imshow(cv2.cvtColor(result_image_day_0, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Day 0 Matches - {cleaned_basename}")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(result_image_day_1, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"Day 1 Matches - {cleaned_basename}")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(result_image_day_2, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"Day 2 Matches - {cleaned_basename}")
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(new_mask_day_0, cmap='tab20b')
        axes[1, 0].set_title(f"New Mask Day 0 - {cleaned_basename}")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask_day_1_final, cmap='tab20b')
        axes[1, 1].set_title(f"New Mask Day 1 - {cleaned_basename}")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(new_mask_day_2, cmap='tab20b')
        axes[1, 2].set_title(f"New Mask Day 2 - {cleaned_basename}")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Saved new filtered Day 0 mask to {output_path_mask_day_0}")
        print(f"Saved new filtered Day 1 mask to {output_path_mask_day_1}")
        print(f"Saved new filtered Day 2 mask to {output_path_mask_day_2}")


#%%
# Example usage with the new filtering step
directory = Path(r"Z:\skala\Angela Hsu\060325_keyence redo\tracking representative image")

mask_pairs = load_mask_pairs(directory)

process_mask_pairs_with_distance_filtering(mask_pairs, directory)
