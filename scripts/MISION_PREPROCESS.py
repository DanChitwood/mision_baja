# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from os import makedirs
from os.path import join, exists, basename, dirname
import glob

# ==============================================================================
# 0. PARAMETERS
# ==============================================================================

# --- Data File and Directory Parameters ---
BASE_DATA_DIR = "../data/"
MISION_DATA_DIR = join(BASE_DATA_DIR, "mision")
TARGET_GENOTYPE = 'mision'

# OUTPUT DIRECTORIES
OUTPUT_ROOT_DIR = "../outputs/"
VERIFICATION_DIR = join(OUTPUT_ROOT_DIR, "verification_plots_mision/")
# FILENAME for the successful leaves CSV
MISION_SUCCESS_CSV = join(VERIFICATION_DIR, "mision_successful_leaves_metadata.csv")

# General parameters
res = 1000
dist = 5
num_land = 20
MIN_TIPS_REQUIRED = 25 

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
    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    if distance[-1] == 0:
        raise ValueError("Trace length is zero. Cannot perform interpolation.")
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

    if not ref_dist:
        raise ValueError(f"Interval between indices {start_ind} and {end_ind} is too small for landmark detection.")

    if use_max:
        max_dist_ind = ref_dist.index(max(ref_dist))
        pt_ind = dist_ind[max_dist_ind]
    else:
        min_dist_ind = ref_dist.index(min(ref_dist))
        pt_ind = dist_ind[min_dist_ind]
    return pt_ind

def internal_landmarks(vein, tip_indices):
    if len(tip_indices) < MIN_TIPS_REQUIRED:
        raise ValueError(f"Not enough tips ({len(tip_indices)}) to calculate the full 40 internal landmarks (Required: {MIN_TIPS_REQUIRED}).")

    # Hardcoded sequence of internal landmarks (40 points)
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

        if end_ind > beg_ind:
            interval_xvals = new_xvals[beg_ind:end_ind+1]
            interval_yvals = new_yvals[beg_ind:end_ind+1]
        else:
            raise ValueError(f"Interval end index ({end_ind}) is before start index ({beg_ind}). Check landmark sequence.")

        if len(interval_xvals) < 2:
            raise ValueError(f"Interval is too short (beg:{beg_ind}, end:{end_ind}) for interpolation.")

        curr_inter_xvals, curr_inter_yvals = interpolation(interval_xvals, interval_yvals, num_land)

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

    if len(land_indices) > 0:
        last_ind = land_indices[-1]
        inter_points_x.append(new_xvals[last_ind])
        inter_points_y.append(new_yvals[last_ind])

    return inter_points_x, inter_points_y

def rotate_and_normalize(vein_xvals, vein_yvals, blade_xvals, blade_yvals, base_ind, tip_ind, end_ind):
    vein_arr = np.column_stack((vein_xvals, vein_yvals)) 
    blade_arr = np.column_stack((blade_xvals, blade_yvals)) 
    vein_len = np.shape(vein_arr)[0]
    blade_len = np.shape(blade_arr)[0]
    overall_len = vein_len + blade_len
    overall_arr = np.row_stack((vein_arr, blade_arr)) 

    pet_junc = np.mean(overall_arr[[base_ind, end_ind], :], axis=0)
    centered_arr = overall_arr - pet_junc

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(centered_arr)
    df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

    p1 = (df["pc1"].loc[tip_ind,], df["pc2"].loc[tip_ind,])
    p2 = (0,0)
    p3 = (10,0)
    angle = angle_between(p1, p2, p3)

    rotated_xvals, rotated_yvals = rotate_points(centered_arr[:, 0], centered_arr[:, 1], angle)
    rotated_arr = np.column_stack((rotated_xvals, rotated_yvals))

    if rotated_arr[10,0] < 0:
        rotated_arr[:,0] = -rotated_arr[:,0]

    centroid_size = np.sqrt(np.sum(rotated_arr**2))

    normalized_arr = rotated_arr / centroid_size

    normalized_vein = normalized_arr[0:vein_len,]
    normalized_blade = normalized_arr[vein_len:(vein_len+blade_len),]

    return normalized_vein, normalized_blade

def scan_for_leaf_files(data_dir, data_label):
    print(f"\n--- Scanning directory: {data_dir} ---")
    vein_files = glob.glob(join(data_dir, '**', '*_veins.txt'), recursive=True)
    mision_metadata_rows = []
    for vein_path in vein_files:
        blade_path = vein_path.replace('_veins.txt', '_blade.txt')
        if exists(blade_path):
            original_leaf_name = basename(vein_path).replace('_veins.txt', '')
            parent_folder = basename(dirname(vein_path))
            if parent_folder and parent_folder != basename(data_dir):
                unique_id = f"{parent_folder}_{original_leaf_name}"
            else:
                unique_id = original_leaf_name
            mision_metadata_rows.append({
                'leaf_id': unique_id,
                'genotype_label': data_label,
                'file_path_prefix': original_leaf_name, 
                'base_path': dirname(vein_path) 
            })
    return pd.DataFrame(mision_metadata_rows)

def plot_and_save_leaf(leaf_id, vein_coords, blade_coords, output_dir):
    fig, ax = plt.subplots(figsize=(4, 6))
    face_color = 'lightgray'
    edge_color = 'black'
    title = f"SUCCESS: {leaf_id}"
    
    # Plot processed shape
    ax.fill(blade_coords[:, 0], blade_coords[:, 1], facecolor=face_color, edgecolor='none')
    ax.plot(vein_coords[:, 0], vein_coords[:, 1], color=edge_color, linewidth=1, zorder=4)
    ax.scatter(vein_coords[:, 0], vein_coords[:, 1], color='darkred', s=5, alpha=0.5, zorder=5) 
    
    ax.set_title(title, fontsize=6)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off') 
    plt.tight_layout()

    plot_path = join(output_dir, f"{leaf_id}_processed_success.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig) 
    return plot_path

def plot_original_leaf_fail(leaf_id, vein_trace, blade_trace, error_message, output_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot raw traces if available
    if blade_trace.ndim == 2 and blade_trace.shape[0] > 1:
        ax.plot(blade_trace[:, 0], blade_trace[:, 1], color='lightcoral', linewidth=0.5, label='Raw Blade')
        ax.scatter(blade_trace[:, 0], blade_trace[:, 1], color='lightcoral', s=5, alpha=0.3)
    
    if vein_trace.ndim == 2 and vein_trace.shape[0] > 1:
        ax.plot(vein_trace[:, 0], vein_trace[:, 1], color='darkred', linewidth=1, label='Raw Vein')
        ax.scatter(vein_trace[:, 0], vein_trace[:, 1], color='darkred', s=8, alpha=0.8)

    ax.set_title(f"FAILURE (RAW TRACE): {leaf_id}\nError: {error_message[:70]}...", fontsize=7, color='red')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=6)
    plt.tight_layout()

    plot_path = join(output_dir, f"{leaf_id}_RAW_FAILED.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


# ==============================================================================
# 3. DATA PREPARATION
# ==============================================================================

# Scan the Misión data directory
mision_metadata = scan_for_leaf_files(MISION_DATA_DIR, data_label=TARGET_GENOTYPE)
valid_leaf_list = mision_metadata['leaf_id'].tolist()

print(f"\nStarting verification analysis on a total of {len(valid_leaf_list)} Misión leaves.")

successful_leaves_metadata = []

# Ensure output directory exists for the verification files
makedirs(VERIFICATION_DIR, exist_ok=True)


# ==============================================================================
# 4. MAIN LOOP (Processing, Plotting, and Recording)
# ==============================================================================
for i, curr_leaf_id in enumerate(valid_leaf_list):

    leaf_row = mision_metadata[mision_metadata['leaf_id'] == curr_leaf_id].iloc[0]
    base_path = leaf_row['base_path']
    file_prefix = leaf_row['file_path_prefix']
    
    vein_trace = np.array([])
    blade_trace = np.array([])
    
    print(f"Attempting to process leaf {i+1}/{len(valid_leaf_list)}: {curr_leaf_id}")

    try:
        # --- 1. Read Raw Data ---
        vein_filepath = join(base_path, file_prefix + "_veins.txt")
        blade_filepath = join(base_path, file_prefix + "_blade.txt")
        
        vein_trace = np.loadtxt(vein_filepath)
        blade_trace = np.loadtxt(blade_filepath)
        
        # --- 2. Interpolation and Landmark Detection ---
        inter_vein_x, inter_vein_y = interpolation(vein_trace[:, 0], vein_trace[:, 1], res)
        inter_blade_x, inter_blade_y = interpolation(blade_trace[:, 0], blade_trace[:, 1], res)
        
        origin = np.mean((vein_trace[0], vein_trace[-1]), axis=0)
        dist_ori = [euclid_dist(origin[0], origin[1], inter_vein_x[j], inter_vein_y[j]) for j in range(res)]
        peaks, _ = find_peaks(dist_ori, height=0, distance=dist)
        peaks = np.insert(peaks, 0, 0)
        peaks = np.append(peaks, res - 1)
        
        inter_vein = np.column_stack((inter_vein_x, inter_vein_y))
        landmark_indices = internal_landmarks(inter_vein, peaks)
        
        blade_pts = []
        for k in range(len(peaks)):
            blade_dists = [euclid_dist(inter_vein_x[peaks[k]], inter_vein_y[peaks[k]], inter_blade_x[l], inter_blade_y[l]) for l in range(res)]
            blade_pts.append(blade_dists.index(min(blade_dists)))

        # --- 3. Pseudo-Landmark Interpolation and Scaling ---
        curr_tip_ind = peaks
        curr_int_ind = landmark_indices
        curr_bla_ind = blade_pts

        if len(curr_tip_ind) != MIN_TIPS_REQUIRED or len(curr_int_ind) != 40:
            raise ValueError(f"LANDMARK MISMATCH: Tips: {len(curr_tip_ind)} (Req: {MIN_TIPS_REQUIRED}), Internal: {len(curr_int_ind)} (Req: 40).")

        # Hardcoded sequence of indices to extract pseudo-landmarks
        curr_vei_ind = [curr_tip_ind[0], curr_int_ind[0], curr_tip_ind[1], curr_int_ind[1], curr_int_ind[2], curr_tip_ind[2], curr_int_ind[3], curr_int_ind[4], curr_tip_ind[3], curr_int_ind[5], curr_tip_ind[4], curr_int_ind[6], curr_int_ind[7], curr_tip_ind[5], curr_int_ind[8], curr_int_ind[9], curr_tip_ind[6], curr_int_ind[10], curr_int_ind[11], curr_tip_ind[7], curr_int_ind[12], curr_tip_ind[8], curr_int_ind[13], curr_int_ind[14], curr_tip_ind[9], curr_int_ind[15], curr_int_ind[16], curr_tip_ind[10], curr_int_ind[17], curr_int_ind[18], curr_tip_ind[11], curr_int_ind[19], curr_tip_ind[12], curr_int_ind[20], curr_tip_ind[13], curr_int_ind[21], curr_int_ind[22], curr_tip_ind[14], curr_int_ind[23], curr_int_ind[24], curr_tip_ind[15], curr_int_ind[25], curr_int_ind[26], curr_tip_ind[16], curr_int_ind[27], curr_tip_ind[17], curr_int_ind[28], curr_int_ind[29], curr_tip_ind[18], curr_int_ind[30], curr_int_ind[31], curr_tip_ind[19], curr_int_ind[32], curr_int_ind[33], curr_tip_ind[20], curr_int_ind[34], curr_tip_ind[21], curr_int_ind[35], curr_int_ind[36], curr_tip_ind[22], curr_int_ind[37], curr_int_ind[38], curr_tip_ind[23], curr_int_ind[39], curr_tip_ind[24]]

        vein_pseudx, vein_psuedy = interpolated_intervals(curr_vei_ind, inter_vein_x, inter_vein_y, num_land)
        blade_pseudx, blade_psuedy = interpolated_intervals(curr_bla_ind, inter_blade_x, inter_blade_y, num_land)

        vein_len = len(vein_pseudx)
        total_len = len(vein_pseudx) + len(blade_pseudx)
        tip_ind = int(vein_len / 2)
        base_ind = 0
        end_ind = total_len - 1

        scaled_vein, scaled_blade = rotate_and_normalize(vein_pseudx, vein_psuedy,
                                                         blade_pseudx, blade_psuedy,
                                                         base_ind=base_ind, tip_ind=tip_ind, end_ind=end_ind)

        # --- SUCCESS ACTION: Save plot and record leaf ---
        plot_path = plot_and_save_leaf(curr_leaf_id, scaled_vein, scaled_blade, VERIFICATION_DIR)
        
        # Record the successful Misión leaf for CSV
        successful_leaves_metadata.append({
            'leaf_id': curr_leaf_id,
            'genotype_label': TARGET_GENOTYPE,
            'file_path_prefix': file_prefix, 
            'base_path': base_path,
            'plot_path': plot_path
        })
        
        print(f"-> MISION SUCCESS: Verification plot saved for {curr_leaf_id}.")

    except Exception as e:
        error_msg = str(e)
        
        # --- FAILURE ACTION: Save raw plot ---
        
        # Attempt to load raw data for plotting if not already loaded (for failed trace visualization)
        try:
            if vein_trace.size == 0 and exists(vein_filepath):
                vein_trace = np.loadtxt(vein_filepath)
            if blade_trace.size == 0 and exists(blade_filepath):
                blade_trace = np.loadtxt(blade_filepath)
        except Exception:
            # Handle cases where even loading the raw file fails
            vein_trace = np.array([[0,0], [1,1]]) 
            blade_trace = np.array([[0,0], [1,1]])

        plot_original_leaf_fail(curr_leaf_id, vein_trace, blade_trace, error_msg, VERIFICATION_DIR)
        print(f"-> MISION FAILURE: Raw debug plot saved. Error: {error_msg}")
        
        continue # Skip to the next leaf


# ==============================================================================
# 5. FINAL REPORT AND CSV SAVING
# ==============================================================================

if successful_leaves_metadata:
    print("\n============================================================")
    print(f"✅ Successfully Processed Misión Leaves ({len(successful_leaves_metadata)} total) and saving CSV.")
    print("============================================================")
    
    # Save the CSV file
    try:
        mision_success_df = pd.DataFrame(successful_leaves_metadata)
        mision_success_df.to_csv(MISION_SUCCESS_CSV, index=False)
        print(f"CSV saved successfully: {MISION_SUCCESS_CSV}")
    except Exception as e:
        print(f"ERROR: Failed to save Misión success CSV: {e}")
        
    for leaf in successful_leaves_metadata:
        print(f"ID: {leaf['leaf_id']} -> Plot: {leaf['plot_path']}")
        
else:
    print("\n❌ No Misión leaves were successfully processed.")

print(f"\nVerification plots (success and failure) are saved in: {VERIFICATION_DIR}")
print("\n--- Script Execution Complete ---")