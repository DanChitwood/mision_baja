# scripts/05_CNN_classify_gradCAM_fig.py

# ==============================================================================
# 0. PARAMETERS (UPDATED)
# ==============================================================================

# File and directory parameters
ROOT_DIR = "../"
DATA_DIR = "../data/high_res_data/"
METADATA_FILE = "../data/leaf_metadata.csv"
GRADCAM_DIR = "../outputs/CNN_classification/trained_models/grad_cam_images/"
OUTPUT_DIR = "../outputs/figures/"
OUTPUT_FILENAME = "fig_gradCAM.png"

# Visualization parameters
MEANLEAF_BLADE_COLOR = "lightgray"
MEANLEAF_VEIN_COLOR = "darkgray"
FIGURE_WIDTH = 8.5 # inches
FIGURE_DPI = 300

# --- UPDATED CLASS LABELS (6 classes for 2x3 layout) ---
CLASS_LABELS = [
    "algeria", "rootstock", "vinifera", 
    "wild", "dissected", "mision" 
]
NUM_ROWS = 2
NUM_COLS = 3

# Landmarking and PCA parameters (from previous script, adapted)
res = 1000
dist = 5
num_land = 20
PC_NUMBER = 2


# ==============================================================================
# 1. IMPORTS
# ==============================================================================

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from os.path import isfile, join, exists, splitext
from os import makedirs


# ==============================================================================
# 2. FUNCTIONS (LANDMARKING AND ROTATION)
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
    if len(x) < 2:
        return np.array([]), np.array([])
    
    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    
    if distance[-1] < 1e-9:
        if len(x) >= 1:
            return np.full(number, x[0]), np.full(number, y[0])
        return np.array([]), np.array([])

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
    
    r = []
    if use_forward:
        r = range(start_ind + 1, end_ind)
    else:
        r = range(start_ind - 1, end_ind, -1)

    for i in r:
        if 0 <= i < len(vein):
            ref_dist.append(euclid_dist(vein[ref_ind, 0], vein[ref_ind, 1], vein[i, 0], vein[i, 1]))
            dist_ind.append(i)

    if not ref_dist:
        return start_ind 

    if use_max:
        max_dist_ind = ref_dist.index(max(ref_dist))
        pt_ind = dist_ind[max_dist_ind]
    else:
        min_dist_ind = ref_dist.index(min(ref_dist))
        pt_ind = dist_ind[min_dist_ind]
        
    return pt_ind

def internal_landmarks(vein, tip_indices):
    if len(tip_indices) < 25:
        return []
        
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
    
    # Backward Landmarks
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
        
        if beg_ind >= end_ind or end_ind > len(new_xvals) or beg_ind < 0 or (end_ind - beg_ind) < 2:
            continue
        
        interval_xvals = new_xvals[beg_ind:end_ind]
        interval_yvals = new_yvals[beg_ind:end_ind]
        
        if len(interval_xvals) < 2:
            continue
            
        curr_inter_xvals, curr_inter_yvals = interpolation(interval_xvals, interval_yvals, num_land)
        
        if len(curr_inter_xvals) == 0:
            continue
            
        curr_inter_xvals = list(curr_inter_xvals)
        curr_inter_yvals = list(curr_inter_yvals)
        
        if i == 0:
            if len(curr_inter_xvals) > 1:
                del curr_inter_xvals[-1]
                del curr_inter_yvals[-1]
        elif i != 0:
            if len(curr_inter_xvals) > 0:
                del curr_inter_xvals[0]
                del curr_inter_yvals[0]
            
        for j in range(len(curr_inter_xvals)):
            inter_points_x.append(curr_inter_xvals[j])
            inter_points_y.append(curr_inter_yvals[j])
            
    if len(land_indices) > 0:
        last_ind = land_indices[-1]
        if last_ind < len(new_xvals):
             inter_points_x.append(new_xvals[last_ind])
             inter_points_y.append(new_yvals[last_ind])
            
    return inter_points_x, inter_points_y

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
    vein_arr = np.column_stack((vein_xvals, vein_yvals))
    blade_arr = np.column_stack((blade_xvals, blade_yvals))
    vein_len = np.shape(vein_arr)[0]
    overall_arr = np.row_stack((vein_arr, blade_arr))
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(overall_arr)
    df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
    
    p1 = (df["pc1"].loc[tip_ind,], df["pc2"].loc[tip_ind,])
    p2 = (0,0)
    p3 = (10,0)
    angle = angle_between(p1, p2, p3)
    rotated_xvals, rotated_yvals = rotate_points(df["pc1"], df["pc2"], angle)
    rotated_arr = np.column_stack((rotated_xvals, rotated_yvals))

    pet_junc = np.mean(rotated_arr[[base_ind,end_ind],:],axis=0)
    
    trans_x = rotated_arr[:,0] - pet_junc[0]
    trans_y = rotated_arr[:,1] - pet_junc[1]
    
    centered_arr = np.column_stack((trans_x, trans_y))
    
    if centered_arr[10,0] < 0: 
        centered_arr[:,0] = -centered_arr[:,0]
        
    scaled_vein = centered_arr[0:vein_len,]
    scaled_blade = centered_arr[vein_len:,]
    
    return scaled_vein, scaled_blade

def rotate_to_negative_y(leaf_arr, base_ind, tip_ind):
    xvals = leaf_arr[:, 0]
    yvals = leaf_arr[:, 1]

    tip_vec = np.array([xvals[tip_ind] - xvals[base_ind], yvals[tip_ind] - yvals[base_ind]])
    target_vec = np.array([0, -1])
    
    dot_product = np.dot(tip_vec, target_vec)
    magnitude_product = np.linalg.norm(tip_vec) * np.linalg.norm(target_vec)
    
    if magnitude_product < 1e-9:
        angle_rads = 0
    else:
        angle_rads = np.arccos(np.clip(dot_product / magnitude_product, -1.0, 1.0))

    cross_product = np.cross(tip_vec, target_vec)
    if cross_product < 0:
        angle_rads = -angle_rads

    angle_degrees = np.degrees(angle_rads)

    rotated_xvals, rotated_yvals = rotate_points(xvals, yvals, angle_degrees)
    rotated_arr = np.column_stack((rotated_xvals, rotated_yvals))

    centroid = np.mean(rotated_arr, axis=0)
    rotated_arr -= centroid

    if rotated_arr[10,0] < 0:
        rotated_arr[:,0] = -rotated_arr[:,0]

    return rotated_arr

# --- UPDATED HELPER FUNCTION FOR CLASSIFYING LEAVES ---
def classify_leaf(filename, genotype_label):
    """Collapses specific Algeria types into one 'algeria' class and adds 'mision' class."""
    
    # 1. Collapse all specific Algeria varieties into a single 'algeria' class
    algeria_vars = ["AHMEUR BOU AHMEUR", "BABARI", "GENOTYPE 1", "GENOTYPE 2", 
                    "GENOTYPE 3", "GENOTYPE 4", "LOUALI", "TIZI OUININE"]
    
    if genotype_label == "algeria":
        for var in algeria_vars:
            if var in filename:
                return "algeria"
        # If it was labeled 'algeria' in metadata but didn't match the specific list 
        # (e.g., if the metadata label was just 'algeria'), treat it as 'algeria'
        return "algeria" 
    
    # 2. Assign 'mision' class
    # Assuming 'mision' leaves are explicitly labeled 'mision' in the genotype_label column.
    if genotype_label == "mision":
        return "mision"

    # 3. For all other classes (rootstock, vinifera, wild, dissected)
    return genotype_label


# ==============================================================================
# 3. DATA PREPARATION
# ==============================================================================

try:
    metadata_df = pd.read_csv(METADATA_FILE)
except FileNotFoundError:
    print(f"Error: Metadata file not found at '{METADATA_FILE}'.")
    exit()

valid_metadata = metadata_df[metadata_df['is_error'] == False].reset_index(drop=True)
valid_leaf_list = valid_metadata['leaf_id'].tolist()
final_geno_labels = valid_metadata['genotype_label'].tolist()
initial_leaf_count = len(valid_leaf_list)

print(f"Starting with {initial_leaf_count} leaves from the metadata file.")

processed_shapes = {}
leaves_to_remove = []

for label in CLASS_LABELS:
    processed_shapes[label] = []

expected_num_pseuds = None
expected_num_vein_coords = None

# Main loop for data processing and classification
for i, curr_leaf in enumerate(valid_leaf_list):
    print(f"Processing leaf {i+1}: {curr_leaf}")

    # Read in trace files
    try:
        vein_trace = np.loadtxt(join(DATA_DIR, curr_leaf + "_veins.txt"))
        inter_vein_x, inter_vein_y = interpolation(vein_trace[:, 0], vein_trace[:, 1], res)
        blade_trace = np.loadtxt(join(DATA_DIR, curr_leaf + "_blade.txt"))
        inter_blade_x, inter_blade_y = interpolation(blade_trace[:, 0], blade_trace[:, 1], res)
    except (FileNotFoundError, ValueError):
        print(f"Skipping leaf {curr_leaf}: Data file issue.")
        leaves_to_remove.append(i)
        continue
    
    # Placeholder for scale factor
    px2_cm2 = 1.0
    try:
        info_df = pd.read_csv(join(DATA_DIR, curr_leaf + "_info.csv"))
        px2_cm2 = float(info_df.iloc[6, 1])
    except:
        pass

    # Find tip landmarks
    origin = np.mean((vein_trace[0], vein_trace[-1]), axis=0)
    dist_ori = [euclid_dist(origin[0], origin[1], inter_vein_x[j], inter_vein_y[j]) for j in range(res)]
    
    if not dist_ori:
        leaves_to_remove.append(i)
        continue
        
    peaks, _ = find_peaks(dist_ori, height=0, distance=dist)
    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, res - 1)

    # Find internal landmarks
    inter_vein = np.column_stack((inter_vein_x, inter_vein_y))
    
    if len(peaks) < 25:
        leaves_to_remove.append(i)
        continue
        
    landmark_indices = internal_landmarks(inter_vein, peaks)
    
    if not landmark_indices:
        leaves_to_remove.append(i)
        continue
    
    # Find blade landmarks
    blade_pts = []
    blade_issue = False
    for k in range(len(peaks)):
        blade_dists = [euclid_dist(inter_vein_x[peaks[k]], inter_vein_y[peaks[k]], inter_blade_x[l], inter_blade_y[l]) for l in range(res)]
        if not blade_dists: 
            blade_issue = True
            break
        blade_pts.append(blade_dists.index(min(blade_dists)))
    
    if blade_issue or len(blade_pts) != len(peaks): 
        leaves_to_remove.append(i)
        continue 

    # Combine landmarks into vein and blade pseudo-landmarks
    curr_tip_ind = peaks
    curr_int_ind = landmark_indices
    
    if len(curr_tip_ind) != 25 or len(curr_int_ind) != 40:
        leaves_to_remove.append(i)
        continue
        
    curr_vei_ind = [curr_tip_ind[0], curr_int_ind[0], curr_tip_ind[1], curr_int_ind[1], curr_int_ind[2], curr_tip_ind[2], curr_int_ind[3], curr_int_ind[4], curr_tip_ind[3], curr_int_ind[5], curr_tip_ind[4], curr_int_ind[6], curr_int_ind[7], curr_tip_ind[5], curr_int_ind[8], curr_int_ind[9], curr_tip_ind[6], curr_int_ind[10], curr_int_ind[11], curr_tip_ind[7], curr_int_ind[12], curr_tip_ind[8], curr_int_ind[13], curr_int_ind[14], curr_tip_ind[9], curr_int_ind[15], curr_int_ind[16], curr_tip_ind[10], curr_int_ind[17], curr_int_ind[18], curr_tip_ind[11], curr_int_ind[19], curr_tip_ind[12], curr_int_ind[20], curr_tip_ind[13], curr_int_ind[21], curr_int_ind[22], curr_tip_ind[14], curr_int_ind[23], curr_int_ind[24], curr_tip_ind[15], curr_int_ind[25], curr_int_ind[26], curr_tip_ind[16], curr_int_ind[27], curr_tip_ind[17], curr_int_ind[28], curr_int_ind[29], curr_tip_ind[18], curr_int_ind[30], curr_int_ind[31], curr_tip_ind[19], curr_int_ind[32], curr_int_ind[33], curr_tip_ind[20], curr_int_ind[34], curr_tip_ind[21], curr_int_ind[35], curr_int_ind[36], curr_tip_ind[22], curr_int_ind[37], curr_int_ind[38], curr_tip_ind[23], curr_int_ind[39], curr_tip_ind[24]]

    vein_pseudx, vein_psuedy = interpolated_intervals(curr_vei_ind, inter_vein_x, inter_vein_y, num_land)
    blade_pseudx, blade_psuedy = interpolated_intervals(blade_pts, inter_blade_x, inter_blade_y, num_land)
    
    if len(vein_pseudx) < 2 or len(blade_pseudx) < 2:
        leaves_to_remove.append(i)
        continue
        
    vein_len = len(vein_pseudx)
    blade_len = len(blade_pseudx)
    total_len = vein_len + blade_len

    if expected_num_pseuds is None:
        expected_num_pseuds = total_len
        expected_num_vein_coords = vein_len

    if total_len != expected_num_pseuds:
        leaves_to_remove.append(i)
        continue

    tip_ind = int(vein_len / 2)
    base_ind = 0
    end_ind = total_len - 1

    scaled_vein, scaled_blade = rotate_and_center(vein_pseudx, vein_psuedy,
                                                  blade_pseudx, blade_psuedy,
                                                  tip_ind=tip_ind,
                                                  base_ind=base_ind,
                                                  end_ind=end_ind)

    full_shape = np.row_stack((scaled_vein, scaled_blade))
    
    leaf_class = classify_leaf(curr_leaf, valid_metadata.loc[i, 'genotype_label'])
    
    if leaf_class in processed_shapes:
        processed_shapes[leaf_class].append(full_shape)
    
# Apply Filtering to labels list
for index in sorted(leaves_to_remove, reverse=True):
    if 0 <= index < len(valid_leaf_list):
        del valid_leaf_list[index]
        del final_geno_labels[index]
        
num_samples = len(valid_leaf_list)

if num_samples == 0 or expected_num_pseuds is None:
    print("\nError: No valid leaf shapes remain after filtering. Cannot generate figure.")
    exit()

num_pseuds = expected_num_pseuds
num_vein_coords = expected_num_vein_coords
tip_of_mean_leaf_index = num_vein_coords // 2
base_of_mean_leaf_index = 0

# Calculate GPA mean for each class
mean_leaves = {}
for label in CLASS_LABELS:
    shapes = processed_shapes.get(label, [])
    if shapes:
        class_shape_arr = np.array(shapes)
        mean_shape = calc_gpa_mean(class_shape_arr, num_pseuds, 2)
        
        # Orient the mean leaf with the tip pointing downwards
        oriented_mean_leaf = rotate_to_negative_y(mean_shape,
                                                  base_ind=base_of_mean_leaf_index,
                                                  tip_ind=tip_of_mean_leaf_index)
                                                  
        mean_leaves[label] = oriented_mean_leaf


# ==============================================================================
# 4. FIGURE GENERATION (UPDATED)
# ==============================================================================

print("\nGenerating figure with 2 rows and 3 columns...")

# Create figure with 2 rows (for mean leaf) + 2 rows (for Grad-CAM) = 4 total rows
fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * (4/3) * 1.1)) # Adjust height for 4 rows
gs = plt.GridSpec(NUM_ROWS * 2, NUM_COLS, figure=fig, wspace=0.05, hspace=0.5)

if not exists(OUTPUT_DIR):
    makedirs(OUTPUT_DIR)

# Loop through each of the 6 classes
for i, class_name in enumerate(CLASS_LABELS):

    # Calculate the row and column index for the current class (0 to 5)
    row_offset = i // NUM_COLS
    col_idx = i % NUM_COLS

    # --- TOP PANEL (MEAN LEAF) - Grid Rows 0 and 1 (for first 3 classes) or 2 and 3 (for last 3 classes) ---
    # The mean leaf for class i goes in the top half of its column
    ax_mean = fig.add_subplot(gs[row_offset, col_idx])
    
    if class_name in mean_leaves:
        mean_leaf = mean_leaves[class_name]

        rotated_vein_coords = mean_leaf[0:num_vein_coords]
        rotated_blade_coords = mean_leaf[num_vein_coords:]

        # Plot the mean leaf
        ax_mean.fill(rotated_blade_coords[:, 0], rotated_blade_coords[:, 1], facecolor=MEANLEAF_BLADE_COLOR, edgecolor="none")
        ax_mean.fill(rotated_vein_coords[:, 0], rotated_vein_coords[:, 1], facecolor=MEANLEAF_VEIN_COLOR, edgecolor="none")

    ax_mean.set_title(class_name, fontsize=8, pad=5)
    ax_mean.set_aspect('equal')
    ax_mean.axis('off')

    # --- BOTTOM PANEL (GRAD-CAM IMAGE) - Grid Rows 2 and 3 (for first 3 classes) or 4 and 5 (for last 3 classes) ---
    # The Grad-CAM for class i goes in the bottom half of its column (row index is offset by NUM_ROWS)
    ax_gradcam = fig.add_subplot(gs[row_offset + NUM_ROWS, col_idx])

    # Construct Grad-CAM filename (The Grad-CAMs might still be named after the original 8 Algeria varieties, 
    # but since you're showing one mean, we assume there is a single combined GradCAM for 'algeria', or 
    # we use a proxy/placeholder if not available)
    
    # Use the class name directly for the Grad-CAM filename
    gradcam_class_name = class_name.replace(" ", "_")
    
    # A potential issue here: if the GradCAM filenames for Algeria are still the 8 individual names,
    # this will fail. I'm assuming for the figure, you either have a combined 'algeria' GradCAM 
    # or you'll need to create one. For now, I use the simplified name.
    
    filename = f"ECT_Mask_4Channel_CNN_Ensemble_Improved_GradCAM_{gradcam_class_name}.png"
    filepath = join(GRADCAM_DIR, filename)

    if exists(filepath):
        gradcam_img = Image.open(filepath)
        ax_gradcam.imshow(gradcam_img)
    else:
        # Placeholder for missing Grad-CAM file (e.g., if 'algeria' GradCAM hasn't been generated yet)
        print(f"Warning: Grad-CAM file not found for {class_name} at {filepath}")
        ax_gradcam.text(0.5, 0.5, f"Grad-CAM Missing: {class_name}", ha='center', va='center', fontsize=6)
    
    ax_gradcam.axis('off')


plt.tight_layout()

output_filepath = join(OUTPUT_DIR, OUTPUT_FILENAME)
plt.savefig(output_filepath, bbox_inches='tight', dpi=FIGURE_DPI)

print(f"\nFigure saved to {output_filepath}")
print("\nScript finished successfully! 🎉")