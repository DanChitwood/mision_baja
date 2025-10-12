# ==============================================================================
# 0. PARAMETERS
# ==============================================================================

# File and directory parameters
DATA_DIR = "../data/high_res_data/"
OUTPUT_DIR = "../outputs/figures/"
OUTPUT_FILENAME = "fig_morphospace.png"
METADATA_FILE = "../data/leaf_metadata.csv"

# Visualization parameters
EIGENLEAF_SCALE_FACTOR = 0.8
EIGENLEAF_BLADE_COLOR = "lightgray"
EIGENLEAF_VEIN_COLOR = "darkgray"

# PCA and grid parameters
PC_NUMBER = 2
NUM_PC1_EIGENLEAVES = 10
NUM_PC2_EIGENLEAVES = 5


# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from os import makedirs
from os.path import join, exists
import h5py


# ==============================================================================
# 2. FUNCTIONS
# ==============================================================================

def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + math.degrees(math.atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + math.degrees(math.atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def rotate_points(xvals, yvals, degrees):
    rads = np.deg2rad(degrees)
    new_xvals = xvals * np.cos(rads) - yvals * np.sin(rads)
    new_yvals = xvals * np.sin(rads) + yvals * np.cos(rads)
    return new_xvals, new_yvals

def interpolation(x, y, number):
    # Check for too few points before calculating distance
    if len(x) < 2 or len(y) < 2:
         # This case should be caught by the caller (interpolated_intervals)
         raise ValueError("Interpolation requires at least two points.")
         
    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    distance = distance / distance[-1]
    fx, fy = interp1d(distance, x), interp1d(distance, y)
    alpha = np.linspace(0, 1, number)
    x_regular, y_regular = fx(alpha), fy(alpha)
    return x_regular, y_regular

def euclid_dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def detect_landmark(vein, tip_indices, start_ind, end_ind, ref_ind, use_forward=True, use_max=True):
    ref_dist = []
    dist_ind = []
    if use_forward:
        for i in range(start_ind + 1, end_ind):
            ref_dist.append(euclid_dist(vein[ref_ind, 0], vein[ref_ind, 1], vein[i, 0], vein[i, 1]))
            dist_ind.append(i)
    else:
        for i in range(end_ind + 1, start_ind):
            ref_dist.append(euclid_dist(vein[ref_ind, 0], vein[ref_ind, 1], vein[i, 0], vein[i, 1]))
            dist_ind.append(i)
    if use_max:
        if not ref_dist: # Handle case where loop range is empty
            return start_ind 
        max_dist_ind = ref_dist.index(max(ref_dist))
        pt_ind = dist_ind[max_dist_ind]
    else:
        if not ref_dist: # Handle case where loop range is empty
            return start_ind
        min_dist_ind = ref_dist.index(min(ref_dist))
        pt_ind = dist_ind[min_dist_ind]
    return pt_ind

def internal_landmarks(vein, tip_indices):
    ptB_ind = detect_landmark(vein, tip_indices, tip_indices[1], tip_indices[2], tip_indices[2], use_forward=True, use_max=True)
    ptA_ind = detect_landmark(vein, tip_indices, tip_indices[0], tip_indices[1], ptB_ind, use_forward=True, use_max=False)
    ptD_ind = detect_landmark(vein, tip_indices, tip_indices[2], tip_indices[3], tip_indices[3], use_forward=True, use_max=True)
    ptC_ind = detect_landmark(vein, tip_indices, tip_indices[1], tip_indices[2], ptD_ind, use_forward=True, use_max=False)
    ptF_ind = detect_landmark(vein, tip_indices, tip_indices[3], tip_indices[4], tip_indices[4], use_forward=True, use_max=True)
    ptE_ind = detect_landmark(vein, tip_indices, tip_indices[2], tip_indices[3], ptF_ind, use_forward=True, use_max=False)
    ptG_ind = detect_landmark(vein, tip_indices, tip_indices[4], tip_indices[5], tip_indices[5], use_forward=True, use_max=True)
    ptI_ind = detect_landmark(vein, tip_indices, tip_indices[5], tip_indices[6], tip_indices[6], use_forward=True, use_max=True)
    ptH_ind = detect_landmark(vein, tip_indices, tip_indices[4], tip_indices[5], ptI_ind, use_forward=True, use_max=False)
    ptK_ind = detect_landmark(vein, tip_indices, tip_indices[6], tip_indices[7], tip_indices[7], use_forward=True, use_max=True)
    ptJ_ind = detect_landmark(vein, tip_indices, tip_indices[5], tip_indices[6], ptK_ind, use_forward=True, use_max=False)
    ptM_ind = detect_landmark(vein, tip_indices, tip_indices[7], tip_indices[8], tip_indices[8], use_forward=True, use_max=True)
    ptL_ind = detect_landmark(vein, tip_indices, tip_indices[6], tip_indices[7], ptM_ind, use_forward=True, use_max=False)
    ptN_ind = detect_landmark(vein, tip_indices, tip_indices[8], tip_indices[9], tip_indices[9], use_forward=True, use_max=True)
    ptP_ind = detect_landmark(vein, tip_indices, tip_indices[9], tip_indices[10], tip_indices[10], use_forward=True, use_max=True)
    ptO_ind = detect_landmark(vein, tip_indices, tip_indices[8], tip_indices[9], ptP_ind, use_forward=True, use_max=False)
    ptR_ind = detect_landmark(vein, tip_indices, tip_indices[10], tip_indices[11], tip_indices[11], use_forward=True, use_max=True)
    ptQ_ind = detect_landmark(vein, tip_indices, tip_indices[9], tip_indices[10], ptR_ind, use_forward=True, use_max=False)
    ptT_ind = detect_landmark(vein, tip_indices, tip_indices[11], tip_indices[12], tip_indices[12], use_forward=True, use_max=True)
    ptS_ind = detect_landmark(vein, tip_indices, tip_indices[10], tip_indices[11], ptT_ind, use_forward=True, use_max=False)
    ptzB_ind = detect_landmark(vein, tip_indices, tip_indices[-2], tip_indices[-3], tip_indices[-3], use_forward=False, use_max=True)
    ptzA_ind = detect_landmark(vein, tip_indices, tip_indices[-1], tip_indices[-2], ptzB_ind, use_forward=False, use_max=False)
    ptzD_ind = detect_landmark(vein, tip_indices, tip_indices[-3], tip_indices[-4], tip_indices[-4], use_forward=False, use_max=True)
    ptzC_ind = detect_landmark(vein, tip_indices, tip_indices[-2], tip_indices[-3], ptzD_ind, use_forward=False, use_max=False)
    ptzF_ind = detect_landmark(vein, tip_indices, tip_indices[-4], tip_indices[-5], tip_indices[-5], use_forward=False, use_max=True)
    ptzE_ind = detect_landmark(vein, tip_indices, tip_indices[-3], tip_indices[-4], ptzF_ind, use_forward=False, use_max=False)
    ptzG_ind = detect_landmark(vein, tip_indices, tip_indices[-5], tip_indices[-6], tip_indices[-6], use_forward=False, use_max=True)
    ptzI_ind = detect_landmark(vein, tip_indices, tip_indices[-6], tip_indices[-7], tip_indices[-7], use_forward=False, use_max=True)
    ptzH_ind = detect_landmark(vein, tip_indices, tip_indices[-5], tip_indices[-6], ptzI_ind, use_forward=False, use_max=False)
    ptzK_ind = detect_landmark(vein, tip_indices, tip_indices[-7], tip_indices[-8], tip_indices[-8], use_forward=False, use_max=True)
    ptzJ_ind = detect_landmark(vein, tip_indices, tip_indices[-6], tip_indices[-7], ptzK_ind, use_forward=False, use_max=False)
    ptzM_ind = detect_landmark(vein, tip_indices, tip_indices[-8], tip_indices[-9], tip_indices[-9], use_forward=False, use_max=True)
    ptzL_ind = detect_landmark(vein, tip_indices, tip_indices[-7], tip_indices[-8], ptzM_ind, use_forward=False, use_max=False)
    ptzN_ind = detect_landmark(vein, tip_indices, tip_indices[-9], tip_indices[-10], tip_indices[-10], use_forward=False, use_max=True)
    ptzP_ind = detect_landmark(vein, tip_indices, tip_indices[-10], tip_indices[-11], tip_indices[-11], use_forward=False, use_max=True)
    ptzO_ind = detect_landmark(vein, tip_indices, tip_indices[-9], tip_indices[-10], ptzP_ind, use_forward=False, use_max=False)
    ptzR_ind = detect_landmark(vein, tip_indices, tip_indices[-11], tip_indices[-12], tip_indices[-12], use_forward=False, use_max=True)
    ptzQ_ind = detect_landmark(vein, tip_indices, tip_indices[-10], tip_indices[-11], ptzR_ind, use_forward=False, use_max=False)
    ptzT_ind = detect_landmark(vein, tip_indices, tip_indices[-12], tip_indices[-13], tip_indices[-13], use_forward=False, use_max=True)
    ptzS_ind = detect_landmark(vein, tip_indices, tip_indices[-11], tip_indices[-12], ptzT_ind, use_forward=False, use_max=False)
    
    landmark_indices = [ptA_ind,ptB_ind,ptC_ind,ptD_ind,ptE_ind,ptF_ind,ptG_ind,ptH_ind,ptI_ind,ptJ_ind,
                        ptK_ind,ptL_ind,ptM_ind,ptN_ind,ptO_ind,ptP_ind,ptQ_ind,ptR_ind,ptS_ind,ptT_ind,
                        ptzT_ind,ptzS_ind,ptzR_ind,ptzQ_ind,ptzP_ind,ptzO_ind,ptzN_ind,ptzM_ind,ptzL_ind,ptzK_ind,
                        ptzJ_ind,ptzI_ind,ptzH_ind,ptzG_ind,ptzF_ind,ptzE_ind,ptzD_ind,ptzC_ind,ptzB_ind,ptzA_ind]
    return landmark_indices

def interpolated_intervals(land_indices, new_xvals, new_yvals, num_land):
    inter_points_x = []
    inter_points_y = []
    for i in range(len(land_indices) - 1):
        beg_ind = land_indices[i]
        end_ind = land_indices[i + 1]
        
        # ** FIX: Skip intervals with only 0 or 1 point, which breaks interpolation **
        if end_ind - beg_ind < 2:
            continue
        # **************************************************
        
        interval_xvals = new_xvals[beg_ind:end_ind]
        interval_yvals = new_yvals[beg_ind:end_ind]
        
        # Guard against zero-length arrays which can happen if interpolation fails 
        try:
            curr_inter_xvals, curr_inter_yvals = interpolation(interval_xvals, interval_yvals, num_land)
        except ValueError:
             # If interpolation fails despite the beg-end check, return partial result
             # which will be caught by the shape check later.
             print(f"Warning: Interpolation failed for interval ({beg_ind}, {end_ind}).")
             return [], []

        curr_inter_xvals = list(curr_inter_xvals)
        curr_inter_yvals = list(curr_inter_yvals)
        if i == 0:
            del curr_inter_xvals[-1]
            del curr_inter_yvals[-1]
        if i != 0:
            del curr_inter_xvals[0]
            del curr_inter_yvals[0]
        for j in range(len(curr_inter_xvals)):
            inter_points_x.append(curr_inter_xvals[j])
            inter_points_y.append(curr_inter_yvals[j])
            
    # Add the very last point explicitly which is often missed due to slicing logic
    if len(land_indices) > 0:
        last_ind = land_indices[-1]
        inter_points_x.append(new_xvals[last_ind])
        inter_points_y.append(new_yvals[last_ind])
            
    return inter_points_x, inter_points_y

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calc_gpa_mean(shape_arr, num_pseuds, num_coords):
    shape_list = shape_arr
    ref_ind = 0
    ref_shape = shape_list[ref_ind]
    mean_diff = 10**(-30)
    old_mean = ref_shape
    d = 1000000
    while d > mean_diff:
        arr = np.zeros(((len(shape_list)), num_pseuds, num_coords))
        for i in range(len(shape_list)):
            s1, s2, distance = procrustes(old_mean, shape_list[i])
            arr[i] = s2
        new_mean = np.mean(arr, axis=(0))
        s1, s2, d = procrustes(old_mean, new_mean)
        old_mean = new_mean
    gpa_mean = new_mean
    return gpa_mean

def rotate_and_center(vein_xvals, vein_yvals, blade_xvals, blade_yvals, tip_ind, base_ind, end_ind):
    """
    Rotates the leaf to align the petiole-to-tip axis and centers the leaf based on the PCA centroid.
    Scaling to cm is removed as it's not needed for shape analysis (GPA handles scale).
    """
    vein_arr = np.column_stack((vein_xvals, vein_yvals)) # create vein coordinates array
    blade_arr = np.column_stack((blade_xvals, blade_yvals)) # create blade coordinates array
    vein_len = np.shape(vein_arr)[0] # get lengths of vein and blade arrays to retrieve coords later
    # blade_len = np.shape(blade_arr)[0] # Not needed
    overall_arr = np.row_stack((vein_arr, blade_arr)) # stack vein and blade arrays into single array
    
    # perform a principal component analysis on data to center
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(overall_arr)
    df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
    
    # find the angle of the leaf tip relative to the origin (centroid of the original PCA)
    p1 = (df["pc1"].loc[tip_ind,], df["pc2"].loc[tip_ind,]) # get leaf tip PC1/PC2 coordinate value
    p2 = (0,0) # find angle relative to vertex at origin (PCA centroid)
    p3 = (10,0) # an arbitrary positive point along the x axis to find angle in anticlockwise direction
    angle = angle_between(p1, p2, p3) # find the angle in degrees of tip point relative to origin, anticlockwise
    rotated_xvals, rotated_yvals = rotate_points(df["pc1"], df["pc2"], angle)
    rotated_arr = np.column_stack((rotated_xvals, rotated_yvals)) # stack x and y vals back into one array

    # The array is already centered due to PCA, now we just translate the petiole junction to the origin
    pet_junc = np.mean(rotated_arr[[base_ind,end_ind],:],axis=0)
    
    trans_x = rotated_arr[:,0] - pet_junc[0]
    trans_y = rotated_arr[:,1] - pet_junc[1]
    
    centered_arr = np.column_stack((trans_x, trans_y))
    
    # Final orientation check
    # Check against the 10th index of the vein (an arbitrary point on the left side)
    if centered_arr[10,0] < 0: 
        centered_arr[:,0] = -centered_arr[:,0]
        
    scaled_vein = centered_arr[0:vein_len,] # isolate just vein coords
    scaled_blade = centered_arr[vein_len:,] # isolate just blade coords
    
    return scaled_vein, scaled_blade # return rotated and translated vein and blade

# Function to rotate and center a single shape (e.g., the mean leaf)
def rotate_to_negative_y(leaf_arr, base_ind, tip_ind):
    """
    Rotates a leaf shape so that the vector from base to tip points down the negative y-axis.
    """
    xvals = leaf_arr[:, 0]
    yvals = leaf_arr[:, 1]

    # Vector from base to tip
    tip_vec = np.array([xvals[tip_ind] - xvals[base_ind], yvals[tip_ind] - yvals[base_ind]])
    
    # Target vector (negative y-axis)
    target_vec = np.array([0, -1])
    
    # Calculate angle between vectors
    dot_product = np.dot(tip_vec, target_vec)
    magnitude_product = np.linalg.norm(tip_vec) * np.linalg.norm(target_vec)
    
    # Guard against zero division if magnitude_product is close to zero (highly unlikely here)
    if magnitude_product < 1e-6:
        angle_rads = 0
    else:
        # Clip value to [-1, 1] to avoid math domain error from floating point inaccuracies
        angle_rads = np.arccos(np.clip(dot_product / magnitude_product, -1.0, 1.0))
    
    # Cross product to determine direction of rotation
    cross_product = np.cross(tip_vec, target_vec)
    if cross_product < 0:
        angle_rads = -angle_rads
    
    angle_degrees = np.degrees(angle_rads)
    
    # Rotate the points
    rotated_xvals, rotated_yvals = rotate_points(xvals, yvals, angle_degrees)
    rotated_arr = np.column_stack((rotated_xvals, rotated_yvals))
    
    # Re-center the leaf on the origin
    centroid = np.mean(rotated_arr, axis=0)
    rotated_arr -= centroid

    # Ensure left side of the leaf is left so labels are on right side of plot
    if rotated_arr[10,0] < 0:
        rotated_arr[:,0] = -rotated_arr[:,0]
    
    return rotated_arr

# Function to save PCA parameters and the mean shape
def save_pca_model(filename, pca_model, gpa_mean_shape):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('pca_components', data=pca_model.components_)
        f.create_dataset('pca_explained_variance_ratio', data=pca_model.explained_variance_ratio_)
        f.create_dataset('pca_mean', data=pca_model.mean_)
        f.create_dataset('pca_n_components', data=pca_model.n_components_)
        f.create_dataset('gpa_mean_shape', data=gpa_mean_shape)
    print(f"PCA model parameters and mean shape saved to {filename}")

# Function to save PCA scores and labels
def save_pca_scores(filename, pca_scores, final_labels):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('pca_scores', data=pca_scores)
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('final_labels', data=np.array(final_labels, dtype=dt))
    print(f"PCA scores and labels saved to {filename}")


# ==============================================================================
# 3. DATA PREPARATION (WITH SHAPE HOMOGENEITY CHECK)
# ==============================================================================

# Read the metadata file
try:
    metadata_df = pd.read_csv(METADATA_FILE)
except FileNotFoundError:
    print(f"Error: Metadata file not found at '{METADATA_FILE}'.")
    exit()

# Filter out error leaves and get valid leaf IDs and labels
valid_metadata = metadata_df[metadata_df['is_error'] == False].reset_index(drop=True)
valid_leaf_list = valid_metadata['leaf_id'].tolist()
final_geno_labels = valid_metadata['genotype_label'].tolist()
initial_leaf_count = len(valid_leaf_list)

print(f"Starting with {initial_leaf_count} leaves from the metadata file.")

# Parameters for landmarking
res = 1000
dist = 5
num_land = 20 # Number of pseudo-landmarks per interval (excluding ends)

# Calculate the expected number of total pseudo-landmarks (vein + blade)
# Vein has 64 intervals: 64 * num_land + 1 (last point) = 1281 points.
# Blade has 25 intervals (from tip_indices size 25+): 25 * num_land + 1 (last point) = 501 points.
# Total expected pseudo-landmarks (vein + blade) = (num_intervals * num_land) + num_initial_landmarks
# Based on the original code logic, a full shape is:
# Vein: (64 * 20) + 1 = 1281. Vein intervals: (curr_vei_ind has 65 points, so 64 intervals)
# Blade: (25 * 20) + 1 = 501. Blade intervals: (curr_bla_ind has 26 points, so 25 intervals)
# total_expected_vein_pseudolms = 64 * (num_land - 1) + 65 # roughly
# total_expected_blade_pseudolms = 25 * (num_land - 1) + 26 # roughly
# Since the exact number is hard to trace due to the complex slicing logic in interpolated_intervals,
# we'll determine the expected size from the FIRST successfully processed leaf.

# Lists to store processed data
tip_indices = []
land_indices = []
blade_indices = []
shape_list = []
scaled_vein_list = []
scaled_blade_list = []
leaves_to_remove = [] # List to track indices of leaves with bad shape

expected_num_pseuds = None

# Main loop for data processing
for i, curr_leaf in enumerate(valid_leaf_list):
    print(f"Processing leaf {i+1}: {curr_leaf}")

    # Read in data
    try:
        vein_trace = np.loadtxt(join(DATA_DIR, curr_leaf + "_veins.txt"))
        inter_vein_x, inter_vein_y = interpolation(vein_trace[:, 0], vein_trace[:, 1], res)
        blade_trace = np.loadtxt(join(DATA_DIR, curr_leaf + "_blade.txt"))
        inter_blade_x, inter_blade_y = interpolation(blade_trace[:, 0], blade_trace[:, 1], res)
    except FileNotFoundError:
        print(f"Skipping leaf {curr_leaf}: Data file not found.")
        leaves_to_remove.append(i)
        continue
    except ValueError:
        print(f"Skipping leaf {curr_leaf}: Data file is corrupt or empty.")
        leaves_to_remove.append(i)
        continue

    # Find tip landmarks
    origin = np.mean((vein_trace[0], vein_trace[-1]), axis=0)
    dist_ori = [euclid_dist(origin[0], origin[1], inter_vein_x[j], inter_vein_y[j]) for j in range(res)]
    peaks, _ = find_peaks(dist_ori, height=0, distance=dist)
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, res - 1)
    
    # Check if a sufficient number of peaks were found (original code assumes 25)
    if len(peaks) < 25:
        print(f"Skipping leaf {curr_leaf}: Insufficient number of tip landmarks found ({len(peaks)} < 25).")
        leaves_to_remove.append(i)
        continue
        
    curr_tip_ind = peaks
    tip_indices.append(curr_tip_ind)

    # Find internal landmarks
    inter_vein = np.column_stack((inter_vein_x, inter_vein_y))
    landmark_indices = internal_landmarks(inter_vein, curr_tip_ind)
    land_indices.append(landmark_indices)
    
    # Find blade landmarks
    blade_pts = []
    for k in range(len(curr_tip_ind)):
        blade_dists = [euclid_dist(inter_vein_x[curr_tip_ind[k]], inter_vein_y[curr_tip_ind[k]], inter_blade_x[l], inter_blade_y[l]) for l in range(res)]
        blade_pts.append(blade_dists.index(min(blade_dists)))
    curr_bla_ind = blade_pts
    blade_indices.append(curr_bla_ind)

    # Combine landmarks into vein and blade pseudo-landmarks
    curr_int_ind = land_indices[-1]
    
    # Check for correct number of indices before defining curr_vei_ind
    if len(curr_tip_ind) != 25 or len(curr_int_ind) != 40:
        print(f"Skipping leaf {curr_leaf}: Incorrect number of total landmarks: Tips={len(curr_tip_ind)}, Internal={len(curr_int_ind)}. Expected Tips=25, Internal=40.")
        leaves_to_remove.append(i)
        continue
        
    curr_vei_ind = [curr_tip_ind[0], curr_int_ind[0], curr_tip_ind[1], curr_int_ind[1], curr_int_ind[2], curr_tip_ind[2], curr_int_ind[3], curr_int_ind[4], curr_tip_ind[3], curr_int_ind[5], curr_tip_ind[4], curr_int_ind[6], curr_int_ind[7], curr_tip_ind[5], curr_int_ind[8], curr_int_ind[9], curr_tip_ind[6], curr_int_ind[10], curr_int_ind[11], curr_tip_ind[7], curr_int_ind[12], curr_tip_ind[8], curr_int_ind[13], curr_int_ind[14], curr_tip_ind[9], curr_int_ind[15], curr_int_ind[16], curr_tip_ind[10], curr_int_ind[17], curr_int_ind[18], curr_tip_ind[11], curr_int_ind[19], curr_tip_ind[12], curr_int_ind[20], curr_tip_ind[13], curr_int_ind[21], curr_int_ind[22], curr_tip_ind[14], curr_int_ind[23], curr_int_ind[24], curr_tip_ind[15], curr_int_ind[25], curr_int_ind[26], curr_tip_ind[16], curr_int_ind[27], curr_tip_ind[17], curr_int_ind[28], curr_int_ind[29], curr_tip_ind[18], curr_int_ind[30], curr_int_ind[31], curr_tip_ind[19], curr_int_ind[32], curr_int_ind[33], curr_tip_ind[20], curr_int_ind[34], curr_tip_ind[21], curr_int_ind[35], curr_int_ind[36], curr_tip_ind[22], curr_int_ind[37], curr_int_ind[38], curr_tip_ind[23], curr_int_ind[39], curr_tip_ind[24]]

    vein_pseudx, vein_psuedy = interpolated_intervals(curr_vei_ind, inter_vein_x, inter_vein_y, num_land)
    blade_pseudx, blade_psuedy = interpolated_intervals(curr_bla_ind, inter_blade_x, inter_blade_y, num_land)
    
    # Use the new rotate_and_center function
    vein_len = len(vein_pseudx)
    blade_len = len(blade_pseudx)
    total_len = vein_len + blade_len
    
    # Determine the expected number of pseudo-landmarks from the first successfully processed leaf
    if expected_num_pseuds is None:
        expected_num_pseuds = total_len
        expected_num_vein_coords = vein_len
        print(f"First leaf sets the expected shape size: {expected_num_pseuds} total pseudo-landmarks.")

    # **CRITICAL FIX**: Check for homogeneous shape size
    if total_len != expected_num_pseuds:
        print(f"Skipping leaf {curr_leaf}: Inhomogeneous shape size ({total_len} != {expected_num_pseuds}). This leaf has an error in landmark detection or interpolation.")
        leaves_to_remove.append(i)
        continue
        
    # Calculate tip and base indices for this specific leaf's data
    tip_ind = int(vein_len / 2) # Tip is roughly at the halfway point of the vein coords
    base_ind = 0
    end_ind = total_len - 1
    
    scaled_vein, scaled_blade = rotate_and_center(vein_pseudx, vein_psuedy,
                                                  blade_pseudx, blade_psuedy,
                                                  tip_ind=tip_ind,
                                                  base_ind=base_ind,
                                                  end_ind=end_ind)
    
    scaled_vein_list.append(scaled_vein)
    scaled_blade_list.append(scaled_blade)
    
    # Combine the scaled, rotated, and translated vein and blade for GPA
    shape_list.append(np.row_stack((scaled_vein, scaled_blade)))

# ** Apply Filtering **
# Remove elements in reverse order to maintain correct indices
for index in sorted(leaves_to_remove, reverse=True):
    # Ensure all lists maintain the same length
    if 0 <= index < len(valid_leaf_list):
        print(f"Removed leaf index {index}: {valid_leaf_list[index]}")
        del valid_leaf_list[index]
        del final_geno_labels[index]
        
# Check if any data remains
num_samples = len(shape_list)
if num_samples == 0:
    print("\nError: No valid leaves remain after filtering. Cannot proceed with GPA and PCA.")
    exit()

print(f"\nSuccessfully processed {num_samples} out of {initial_leaf_count} initial leaves.")
num_pseuds = expected_num_pseuds
num_coords = 2


# Reshape data for Procrustes analysis
shape_arr = np.zeros((num_samples, num_pseuds, num_coords))
for i in range(num_samples):
    shape_arr[i] = shape_list[i]

# Perform GPA on the pre-oriented leaves
proc_arr = np.zeros(shape_arr.shape)
gpa_mean = calc_gpa_mean(shape_arr, num_pseuds, num_coords)
for i in range(num_samples):
    s1, s2, distance = procrustes(gpa_mean, shape_arr[i,:,:])
    proc_arr[i,:,:] = s2
    
# Orient the GPA mean leaf with a more robust method
# Use the expected vein length for indexing
tip_of_mean_leaf_index = int(expected_num_vein_coords / 2) 
base_of_mean_leaf_index = 0

# Apply the new, more robust rotation function to the GPA mean leaf
oriented_gpa_mean = rotate_to_negative_y(gpa_mean, base_ind=base_of_mean_leaf_index, tip_ind=tip_of_mean_leaf_index)

# Re-run Procrustes analysis using the newly oriented mean leaf as the reference shape
final_proc_arr = np.zeros(shape_arr.shape)
for i in range(num_samples):
    s1, s2, distance = procrustes(oriented_gpa_mean, shape_arr[i,:,:])
    final_proc_arr[i,:,:] = s2

# Reshape for PCA
reshaped_arr = final_proc_arr.reshape(final_proc_arr.shape[0], final_proc_arr.shape[1] * final_proc_arr.shape[2])

# Find number of vein and blade coords for plotting
num_vein_coords = expected_num_vein_coords
num_blade_coords = num_pseuds - num_vein_coords


# ==============================================================================
# 4. PCA AND VISUALIZATION
# ==============================================================================

# Perform PCA
pca = PCA(n_components=PC_NUMBER)
PCs = pca.fit_transform(reshaped_arr)
geno_pca_df = pd.DataFrame(data=PCs, columns=['PC1', 'PC2'])
geno_pca_df['label'] = final_geno_labels

print("\nExplained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", pca.explained_variance_ratio_.cumsum())

# Create PC values for eigenleaf grid
PC1_vals = np.linspace(np.min(PCs[:,0]), np.max(PCs[:,0]), NUM_PC1_EIGENLEAVES)
PC2_vals = np.linspace(np.min(PCs[:,1]), np.max(PCs[:,1]), NUM_PC2_EIGENLEAVES)

# Create a new figure and a single subplot for the morphospace
fig, ax = plt.subplots(figsize=(8.5, 6))

# Plot eigenleaves in the background of the main morphospace
for i in PC1_vals:
    for j in PC2_vals:
        inv_leaf = pca.inverse_transform(np.array([i, j]))
        inv_leaf = np.reshape(inv_leaf, (num_pseuds, num_coords))
        
        inv_leaf_bladeX = inv_leaf[num_vein_coords:, 0]
        inv_leaf_bladeY = inv_leaf[num_vein_coords:, 1]
        inv_leaf_veinX = inv_leaf[0:num_vein_coords, 0]
        inv_leaf_veinY = inv_leaf[0:num_vein_coords, 1]

        ax.fill(inv_leaf_bladeX * EIGENLEAF_SCALE_FACTOR + i, 
                inv_leaf_bladeY * EIGENLEAF_SCALE_FACTOR + j, 
                facecolor=EIGENLEAF_BLADE_COLOR, edgecolor="none", lw=1, zorder=1)
        
        ax.fill(inv_leaf_veinX * EIGENLEAF_SCALE_FACTOR + i, 
                inv_leaf_veinY * EIGENLEAF_SCALE_FACTOR + j, 
                facecolor=EIGENLEAF_VEIN_COLOR, edgecolor="none", lw=1, zorder=1)

# Generate a custom color palette
unique_labels = sorted(geno_pca_df['label'].dropna().unique())
full_palette = sns.color_palette("Set2")
custom_colors = full_palette[1:len(unique_labels)+1]
label_pal = {label: color for label, color in zip(unique_labels, custom_colors)}

# Plot data points in the foreground
sns.scatterplot(data=geno_pca_df, x="PC1", y="PC2", hue="label", 
                        palette=label_pal, zorder=2, s=50, alpha=1, ax=ax)

# Customize the legend for the main plot
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title=None, 
          frameon=False, fontsize=10, labelspacing=0.5, bbox_to_anchor=(1.01, 1), loc='upper left')

# Final plot formatting for the main morphospace
xlab = f"PC1, {np.round(pca.explained_variance_ratio_[0]*100, 1)}%"
ylab = f"PC2, {np.round(pca.explained_variance_ratio_[1]*100, 1)}%"
ax.set_xlabel(xlab, fontsize=10)
ax.set_ylabel(ylab, fontsize=10)
ax.tick_params(labelsize=8)
ax.set_aspect("equal")
ax.set_facecolor("white")
ax.grid()
ax.set_axisbelow(True)

plt.tight_layout()

# Save the figure
if not exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)
plt.savefig(join(OUTPUT_DIR, OUTPUT_FILENAME), bbox_inches='tight', dpi=300)

print(f"\nFigure saved to {join(OUTPUT_DIR, OUTPUT_FILENAME)}")

# ==============================================================================
# 5. HDF5 FILE SAVING
# ==============================================================================

try:
    # Call the saving functions (per your saved instructions)
    OUTPUT_HDF5_DIR = "../outputs/h5_data" 
    if not exists(OUTPUT_HDF5_DIR):
        makedirs(OUTPUT_HDF5_DIR)

    save_pca_model(join(OUTPUT_HDF5_DIR, "pca_model_and_mean_shape.h5"), pca, oriented_gpa_mean)
    save_pca_scores(join(OUTPUT_HDF5_DIR, "pca_scores_and_labels.h5"), PCs, final_geno_labels)

except ImportError:
    print("\nSkipping HDF5 save: 'h5py' library is not installed.")
    print("Install with 'pip install h5py' to enable model saving.")