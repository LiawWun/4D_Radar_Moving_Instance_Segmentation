import os
import sys
import yaml
import time
import argparse
import numpy as np
from pathlib import Path

from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from collections import deque
from scipy.spatial import cKDTree

from dataset.itri_dataset import ITRI_Dataset
from scipy.optimize import linear_sum_assignment

from model.itri_model import FlowNet3D_GlobalAttention, count_parameters, initialize_weights

from tqdm import tqdm

def iou_calculate(Q, P):

    # Convert the lists of points to sets of tuples for efficient computation
    set_Q = set(map(tuple, Q))
    set_P = set(map(tuple, P))
    
    union_set = set_Q.union(set_P)
    intersection_set = set_Q.intersection(set_P)
    iou = len(intersection_set) / len(union_set)
    
    return iou

def calculate_score(pred_radar_xyz, pred_output_cluster_list, gt_radar_xyz, gt_output_cluster_list, iou_threshold):

    gt_num   = len(gt_output_cluster_list)
    pred_num = len(pred_output_cluster_list)

    iou_matrix = np.zeros((gt_num, pred_num), dtype = np.float32)
    inverse_iou_matrix = np.zeros((gt_num, pred_num), dtype = np.float32)

    for i in range(gt_num):
        for j in range(pred_num):
            
            # Get gt points
            gt_indices = gt_output_cluster_list[i]
            gt_points  = gt_radar_xyz[gt_indices, :]

            # Get pred point
            pred_indices = pred_output_cluster_list[j]
            pred_points  = pred_radar_xyz[pred_indices, :]

            iou = iou_calculate(gt_points, pred_points)

            iou_matrix[i, j] = iou
            inverse_iou_matrix[i, j] = 1 - iou 

    row_ind, col_ind = linear_sum_assignment(inverse_iou_matrix)
    
    tp, fp, fn, sigma_iou = 0, 0, 0, 0
    match_gt_indices = []
    match_pred_indices = []

    for r, c in zip(row_ind, col_ind):
        
        if iou_matrix[r, c] > iou_threshold:
            tp += 1
            sigma_iou += iou_matrix[r, c]
            match_gt_indices.append(r)
            match_pred_indices.append(c)
    
    fp = pred_num - tp
    fn = gt_num   - tp

    all_gt_indices = set(range(gt_num))
    all_pred_indices = set(range(pred_num))
    unmatch_gt_indices = list(all_gt_indices - set(match_gt_indices))
    unmatch_pred_indices = list(all_pred_indices - set(match_pred_indices))

    return tp, fp, fn, sigma_iou, match_gt_indices, unmatch_gt_indices, match_pred_indices, unmatch_pred_indices

def clustering_algorithm(X, S, r, Nθ, stuff_classes):
    """
    Parameters:
    - X: np.ndarray of shape (N, 3), coordinates
    - S: np.ndarray of shape (N,), semantic labels
    - r: float, clustering radius
    - Nθ: int, threshold for cluster point number
    - stuff_classes: set or list of stuff class labels (e.g., walls)
    
    Returns:
    - List of clusters, where each cluster is a list of point indices
    """
    N = len(S)
    v = np.zeros(N, dtype=bool)  # visited flag
    C_all = []  # all valid clusters
    tree = cKDTree(X)

    # Mark stuff classes as visited
    for i in range(N):
        if S[i] in stuff_classes:
            v[i] = True

    for i in range(N):
        if not v[i]:
            Q = deque()
            C = []
            v[i] = True
            Q.append(i)
            C.append(i)

            while Q:
                k = Q.popleft()
                neighbors = tree.query_ball_point(X[k], r)
                for j in neighbors:
                    if S[j] == S[k] and not v[j]:
                        v[j] = True
                        Q.append(j)
                        C.append(j)

            if len(C) > Nθ:
                C_all.append(C)

    return C_all

def get_groundtruth_data(raw_pc, mask):

    # [x, y, z, RCS, v_r, v_r_compensated, moving_mask, cluster_id, xc, yc, zc, time_idx]
    raw_pc = raw_pc[mask]
    raw_pc_xyz = raw_pc[:, [0, 1, 2]]
    cluster_id = raw_pc[:, 7]
    unique_id = np.unique(cluster_id)
    grouped_indices_list = [np.where(cluster_id == val)[0].tolist() for val in unique_id if val != -1 ]

    return raw_pc_xyz, grouped_indices_list

def MS_metrics(output, target):

    output = output.view(-1)
    target = target.view(-1)

    gt_moving_indices = target == 1
    gt_static_indices = target == 0

    pred_moving_indices = output == 1
    pred_static_indices = output == 0

    tp = torch.sum(gt_moving_indices & pred_moving_indices).item()
    fp = torch.sum(gt_static_indices & pred_moving_indices).item()
    tn = torch.sum(gt_static_indices & pred_static_indices).item()
    fn = torch.sum(gt_moving_indices & pred_static_indices).item()

    return tp, fp, tn, fn

def get_unique_points_mask(points):
    points = points.squeeze(0)
    points_tuple = [tuple(p.tolist()) for p in points]

    unique_points_set = set()
    mask = torch.zeros(points.size(0), dtype = torch.bool)

    for i, point in enumerate(points_tuple):

        if point not in unique_points_set:
            unique_points_set.add(point)
            mask[i] = True

    num_unique_points = len(unique_points_set)
    return mask, num_unique_points

def inference(experinment_name = None):

    result_folder = os.path.join("result", experinment_name)
    code_folder = os.path.join(result_folder, "code")
    weight_folder = os.path.join(result_folder, "model_weight")
    yaml_path = os.path.join(result_folder, "config.yaml")
    weight_name_list = sorted(os.listdir(weight_folder))

    if str(Path(code_folder).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(code_folder).resolve()))

    frame_num = 2

    ###### START DATASET ######
    batch_size = 1
    itri_data = ITRI_Dataset(mode = "valid", transform = None, num_frame = frame_num)
    itri_dataloader = DataLoader(itri_data, batch_size = batch_size, shuffle = False, drop_last = True)
    print(f"Valid data length: {len(itri_data)}")
    ###### END OF DATASET ######

    model = FlowNet3D_GlobalAttention()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Moving the model to the GPU.")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")
    count_parameters(model)
    model.to(device)
    ###### END OF MODEL SETTING ######

    ###### TEST SECTION ######
    score = []
    process_time = []
    for weight_name in weight_name_list:

        print(f"Inference using : {weight_name}")
        weight_path = os.path.join(weight_folder, weight_name)
        model.load_state_dict(torch.load(weight_path), strict = True)

        model = model.eval()
        test_iou, total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0, 0
        total_instance_iou, total_instance_tp, total_instance_fp, total_instance_fn = 0, 0, 0, 0 
        with torch.no_grad():


            for idx, data in enumerate(tqdm(itri_dataloader)):

                start_time = time.time()
                # Unpack the batch
                coords, feats, semantic_labels, pc_shift_labels = data

                # Move data to GPU and ensure memory continuity
                coords = coords.to(device).contiguous()
                feats = feats.to(device).contiguous()

                # Get ground truth for the current frame (index 0)
                semantic_gt = semantic_labels[:, 0, :, :].to(device).long()
                pc_shift_gt = pc_shift_labels[:, 0, :, :].to(device)

                # Forward pass through the model
                semantic_pred, pc_shift_pred = model(coords, feats)

                # Select the first sample in batch and reshape
                current_pc = coords[0, 0].permute(1, 0)              # Shape: (N, 3)
                semantic_pred = semantic_pred[0]                     # Shape: (N, num_classes)
                semantic_gt = semantic_gt[0].view(-1)                # Shape: (N,)
                pc_shift_gt = pc_shift_gt[0]                         # Shape: (N, 3)
                pc_shift_pred = pc_shift_pred[0]                     # Shape: (N, 3)

                # Remove duplicated points
                mask, _ = get_unique_points_mask(current_pc)
                current_pc = current_pc[mask]
                semantic_pred = semantic_pred[mask]
                semantic_gt = semantic_gt[mask]
                pc_shift_pred = pc_shift_pred[mask]
                pc_shift_gt = pc_shift_gt[mask]

                # Convert logits to predicted class labels
                semantic_pred = torch.argmax(semantic_pred, dim=1)   # Shape: (N,)

                # Compute TP, FP, TN, FN
                tp, fp, tn, fn = MS_metrics(output=semantic_pred, target=semantic_gt)

                # Accumulate metrics
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

                # Calculate panoptic score
                raw_pc = itri_data.get_frame_data(idx)
                raw_pc_xyz, grouped_indices_list = get_groundtruth_data(raw_pc, mask)

                pc_shift_pred[semantic_pred == 0.0] = 0
                centroid_point = current_pc + pc_shift_pred
                pc_shift_pred = pc_shift_pred.cpu()
                centroid_point = centroid_point.cpu()
                pred_radar_point = current_pc.cpu().numpy()
                output_cluster_list = clustering_algorithm(X = centroid_point, S = semantic_pred, r = 2.0, Nθ = 0.0, stuff_classes = [0])
                instance_tp, instance_fp, instance_fn, sigma_iou, _, _, _, _= calculate_score(pred_radar_point, output_cluster_list, raw_pc_xyz, grouped_indices_list, iou_threshold = 0.5)

                total_instance_iou += sigma_iou
                total_instance_tp += instance_tp
                total_instance_fp += instance_fp
                total_instance_fn += instance_fn
                process_time.append(time.time() - start_time)


            semantic_iou = total_tp / (total_tp + total_fn + total_fp)
            RQ = total_instance_tp / (total_instance_tp + 0.5 * total_instance_fp + 0.5 * total_instance_fn)
            SQ = total_instance_iou / total_instance_tp
            PQ = SQ * RQ
            headers = ["Metric", "Score", "SQ", "RQ", "TP", "FP", "FN"]
            semantic_row = ["Semantic", "{:.4f}".format(semantic_iou), "", "", total_tp, total_fp, total_fn]
            instance_row = ["Instance", "{:.4f}".format(PQ), "{:.4f}".format(SQ), "{:.4f}".format(RQ), total_instance_tp, total_instance_fp, total_instance_fn]
            merged_table = [semantic_row, instance_row]
            print(tabulate(merged_table, headers=headers, tablefmt="grid", stralign="right"))
            score.append([semantic_iou, PQ, SQ, RQ])         
            
    return score, process_time


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <experiment_name>")
        sys.exit(1) # Exit the script if no name is provided

    experiment_name = sys.argv[1]
    
    # The rest of your logic remains the same
    score, process_time = inference(experiment_name)
    score_array = np.array(score)  
    
    avg_semantic_iou = np.max(score_array[:, 0])
    avg_PQ = np.max(score_array[:, 1])
    avg_SQ = np.max(score_array[:, 2])    
    avg_RQ = np.max(score_array[:, 3])                
    
    print(f"\nResults for: {experiment_name}")
    print(f"Semantic IoU: {avg_semantic_iou:.4f}")
    print(f"PQ: {avg_PQ:.4f}")
    print(f"SQ: {avg_SQ:.4f}")
    print(f"RQ: {avg_RQ:.4f}")
    
    average_runtime_s = sum(process_time) / len(process_time)
    average_runtime_ms = average_runtime_s * 1000
    average_frequency = 1 / average_runtime_s
    
    print(f"Average runtime per frame: {average_runtime_ms:.2f} ms")
    print(f"Average frequency: {average_frequency:.2f} Hz")