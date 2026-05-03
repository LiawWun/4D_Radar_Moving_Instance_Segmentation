# === Standard Library ===
import os
import csv
import yaml
import shutil
import datetime
import argparse
from tqdm import tqdm

# === PyTorch Core ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler

# === Project-Specific Imports ===
import transform as t
from dataset.itri_dataset import ITRI_Dataset
from model.itri_model import FlowNet3D_GlobalAttention, count_parameters, initialize_weights
from model_saver import TopKModelSaver


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

class get_loss(nn.Module):

    def __init__(self, class_weight, semantic_weight = 1.0, offset_weight = 1.0):
        super(get_loss, self).__init__()

        self.semantic_weight = semantic_weight
        self.offset_weight   = offset_weight
        self.class_weight    = class_weight 
    
    def forward(self, semantic_pred, semantic_gt, offset_pred, offset_gt):
        
        """
        Computes the total loss as a weighted sum of semantic classification loss
        and offset regression loss.

        Parameters:
        -----------
        semantic_pred : [B, N, 2],
        semantic_gt : [B, N, 1].
        offset_pred : [B, N, 3].
        offset_gt : [B, N, 3].

        Returns:
        --------
        total_loss
        semantic_loss 
        offset_loss
        """

        # Calculate semantic loss
        semantic_pred = semantic_pred.contiguous().view(-1, 2)
        semantic_mask = semantic_gt.view(-1, 1)[:, 0]
        semantic_loss = F.nll_loss(semantic_pred, semantic_mask, weight = self.class_weight)
        
        # Calculate offset loss
        offset_mask = semantic_gt
        offset_pred_masked = offset_pred * offset_mask
        offset_gt_masked   = offset_gt   * offset_mask

        offset_l1 = torch.abs(offset_pred_masked - offset_gt_masked)
        total_valid_points = offset_mask.sum()

        if total_valid_points.item() > 0:
            offset_loss = offset_l1.sum() / total_valid_points
        else:
            offset_loss = torch.tensor(0.0, device=semantic_pred.device)

        total_loss = self.semantic_weight * semantic_loss + self.offset_weight * offset_loss
        return total_loss, semantic_loss, offset_loss

def train():

    ### CHECK DEVICE ###
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.", f"device: {device}")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.", f"device: {device}")

    ###  START DATASET  ###
    print("=" * 100)
    print(f"Dataset info:")

    
    data_transform = t.Compose([t.RandomShift(shift = [0.3, 0.3, 0], prob = 0.9), 
                                t.RandomScale([0.75, 1.25], prob = 0.9), 
                                t.RandomJitter(sigma = 0.075, clip = 0.1 , prob = 0.9),
                                t.RandomJitterOnFeat(prob = 0.9, portion = 0.8),
                                t.RandomFlip(prob = 0.85),
                                t.RandomRotate(angle = [0.0, 0.0, 0.4], prob = 0.85)])
    
    
    itri_data_raw = ITRI_Dataset(mode = "train", transform=None, num_frame = 2)
    itri_data_argumentation = ITRI_Dataset(mode = "train", transform = data_transform, num_frame = 2)
    itri_train_data = ConcatDataset([itri_data_raw, itri_data_argumentation])
    weights = torch.Tensor(itri_data_raw.labelweights).to(device)

    # Train test split
    train_set_size = int(len(itri_train_data) * 0.85)
    valid_set_size = len(itri_train_data) - train_set_size 
    _, itri_valid_data = random_split(itri_train_data, [train_set_size, valid_set_size])

    # DataLoader
    batch_size = 16
    itri_trainloader = DataLoader(itri_train_data, batch_size = batch_size, shuffle = True, drop_last = True)
    itri_validloader = DataLoader(itri_valid_data, batch_size = batch_size, shuffle = True, drop_last = True)

    print(f"Train data length: {len(itri_train_data)}")
    print(f"Valid data length: {len(itri_valid_data)}")
    print("=" * 100)
    ###### END OF DATASET ######

    ###### MODEL SETTING ######
    print("=" * 100)
    print("Model info:")
    model = FlowNet3D_GlobalAttention()
    model.apply(initialize_weights)
    count_parameters(model)
    model.to(device)
    print("=" * 100)
    ###### END OF MODEL SETTING ######

    ###### TRAINING SETTING ######
    init_learning_rate = 0.005
    criterion = get_loss(class_weight = weights , semantic_weight = 3.0, offset_weight = 1.0).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = init_learning_rate, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 5e-5)

    # warmup scheduler
    warmup_steps = 15
    def warmup_lr_lambda(current_step):
        return float(current_step) / float(warmup_steps) if current_step < warmup_steps else 1.0
    
    warmup_scheduler = lr_scheduler.LambdaLR(optimizer, warmup_lr_lambda)

    T_max = 150
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-4)
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    epochs = warmup_steps + T_max
    ###### END OF TRAINING SETTING ######

    ### Create dir to save result
    Output_root = "result"
    experinment_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    Output_folder = os.path.join(Output_root, experinment_name)
    os.makedirs(Output_folder, exist_ok = True)

    # Folder save model
    model_weight_folder = os.path.join(Output_folder, "model_weight")
    os.makedirs(model_weight_folder, exist_ok=True)
    top_k_saver = TopKModelSaver(save_dir = model_weight_folder, k = 5)

    # Save log
    log_folder = os.path.join(Output_folder, "log")
    os.makedirs(log_folder, exist_ok=True)
    train_log_csv_path = os.path.join(log_folder, "train_log.csv")
    valid_log_csv_path = os.path.join(log_folder, "valid_log.csv")


    ## Save this training code and dataset code
    Code_output_folder = os.path.join(Output_folder, "code")
    os.makedirs(Code_output_folder, exist_ok = True)
    training_code_path = "train.py"
    model_code_path    = "model/itri_model.py"
    shutil.copy(training_code_path, Code_output_folder)
    shutil.copy(model_code_path, Code_output_folder)
    print(f"Copied {training_code_path} and {model_code_path} to {Code_output_folder}")

    # Copy the util folder
    util_source_folder = os.path.join("util")
    util_dest_folder = os.path.join(Code_output_folder, "util")
    os.makedirs(util_dest_folder, exist_ok=True)
    for filename in os.listdir(util_source_folder):
        if filename.endswith(".py"):
            full_src_path = os.path.join(util_source_folder, filename)
            full_dst_path = os.path.join(util_dest_folder, filename)
            shutil.copy(full_src_path, full_dst_path)

    print("Start training.....")

    ###### TRAINING SECTION ######
    train_log = {'train_total_loss':[], 'train_semantic_loss':[], 'train_offset_loss':[], 'tp': [], 'fp': [], 'tn': [], 'fn': [], 'IOU': []}
    valid_log = {'valid_total_loss':[], 'valid_semantic_loss':[], 'valid_offset_loss':[], 'tp': [], 'fp': [], 'tn': [], 'fn': [], 'IOU': []}
    best_iou = 0
    for epoch in range(1, epochs + 1):

        print("#" * 100)
        print(f"EPOCH: {epoch}")

        # Training 
        model.train()
        total_loss_sum, semantic_loss_sum, offset_loss_sum, train_iou, total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0, 0, 0, 0, 0
        train_pbar = tqdm(itri_trainloader, desc=f"EPOCH: ({epoch}/{epochs}), LR: {optimizer.param_groups[0]['lr']:.5f}")
        for i, data in enumerate(train_pbar):
           
            optimizer.zero_grad()
            
            # Get data
            coords, feats, semantic_labels, pc_shift_labels = data
            coords = coords.to(device).contiguous()
            feats = feats.to(device).contiguous()
            semantic_gt = semantic_labels[:, 0, :, :].to(device).long()# Only get the current frame's result
            pc_shift_gt = pc_shift_labels[:, 0, :, :].to(device)# Only get the current frame's result

            # Model input
            semantic_pred, offset_pred = model(coords, feats)

            # Loss calculation
            total_loss, semantic_loss, offset_loss = criterion(semantic_pred = semantic_pred, semantic_gt = semantic_gt, offset_pred = offset_pred, offset_gt = pc_shift_gt)
            total_loss.backward()
            total_loss_sum += total_loss.item()
            semantic_loss_sum += semantic_loss.item()
            offset_loss_sum += offset_loss.item()

            # Optimizer setting
            optimizer.step()

            gt   = semantic_gt.view(-1)
            pred = torch.argmax(semantic_pred, dim = 2).view(-1)
            tp, fp, tn, fn = MS_metrics(output = pred, target = gt)

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
        
        train_avg_total_loss = total_loss_sum / i
        train_avg_semantic_loss = semantic_loss_sum / i
        train_avg_offset_loss = offset_loss_sum / i
        train_iou = total_tp / (total_tp + total_fn + total_fp)
        train_log['train_total_loss'].append(train_avg_total_loss)
        train_log['train_semantic_loss'].append(train_avg_semantic_loss)
        train_log['train_offset_loss'].append(train_avg_offset_loss)
        train_log['tp'].append(total_tp)
        train_log['fp'].append(total_fp)
        train_log['tn'].append(total_tn)
        train_log['fn'].append(total_fn)
        train_log['IOU'].append(train_iou)
        print(  f'Training mean loss: {train_avg_total_loss:.6f} | '
                f'Training semantic loss: {train_avg_semantic_loss:.6f} | '
                f'Training offset loss: {train_avg_offset_loss:.6f} | '
                f'TP: {total_tp:6d} | '
                f'FP: {total_fp:6d} | '
                f'TN: {total_tn:6d} | '
                f'FN: {total_fn:6d} | '
                f'IOU: {train_iou}')
        
        # Validate
        model = model.eval()
        total_loss_sum, semantic_loss_sum, offset_loss_sum, valid_iou, total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():

            valid_pbar = tqdm(itri_validloader, desc=f"EPOCH: ({epoch}/{epochs}), LR: {optimizer.param_groups[0]['lr']:.5f}")
            for i, data in enumerate(valid_pbar):

                # Get data
                coords, feats, semantic_labels, pc_shift_labels = data
                coords = coords.to(device).contiguous()
                feats = feats.to(device).contiguous()
                semantic_gt = semantic_labels[:, 0, :, :].to(device).long()# Only get the current frame's result
                pc_shift_gt = pc_shift_labels[:, 0, :, :].to(device)# Only get the current frame's result

                # Model input
                semantic_pred, offset_pred = model(coords, feats)

                # Loss calculation
                total_loss, semantic_loss, offset_loss = criterion(semantic_pred = semantic_pred, semantic_gt = semantic_gt, offset_pred = offset_pred, offset_gt = pc_shift_gt)
                total_loss_sum += total_loss.item()
                semantic_loss_sum += semantic_loss.item()
                offset_loss_sum += offset_loss.item()

                gt   = semantic_gt.view(-1)
                pred = torch.argmax(semantic_pred, dim = 2).view(-1)
                tp, fp, tn, fn = MS_metrics(output = pred, target = gt)

                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn
            
            valid_avg_total_loss = total_loss_sum / i
            valid_avg_semantic_loss = semantic_loss_sum / i
            valid_avg_offset_loss = offset_loss_sum / i
            valid_iou = total_tp / (total_tp + total_fn + total_fp)
            valid_log['valid_total_loss'].append(valid_avg_total_loss)
            valid_log['valid_semantic_loss'].append(valid_avg_semantic_loss)
            valid_log['valid_offset_loss'].append(valid_avg_offset_loss)
            valid_log['tp'].append(total_tp)
            valid_log['fp'].append(total_fp)
            valid_log['tn'].append(total_tn)
            valid_log['fn'].append(total_fn)
            valid_log['IOU'].append(valid_iou)
            print(  f'Valid    mean loss: {valid_avg_total_loss:.6f} | '
                    f'Valid    semantic loss: {valid_avg_semantic_loss:.6f} | '
                    f'Valid    offset loss: {valid_avg_offset_loss:.6f} | '
                    f'TP: {total_tp:6d} | '
                    f'FP: {total_fp:6d} | '
                    f'TN: {total_tn:6d} | '
                    f'FN: {total_fn:6d} | '
                    f'IOU: {valid_iou}')
            
          
        scheduler.step()

        # Save best model
        top_k_saver.update(validation_metric = valid_iou, model = model, epoch=epoch)
        if valid_iou > best_iou:
            best_iou = valid_iou
            print(f"Best model found! IOU = {valid_iou}")
    
    # Save the train log to a CSV file
    with open(train_log_csv_path, mode='w', newline='') as train_log_file:
        writer = csv.writer(train_log_file)
        # Write header
        writer.writerow(['train_total_loss', 'train_semantic_loss', 'train_offset_loss', 'tp', 'fp', 'tn', 'fn', 'IOU'])
        # Write data
        for i in range(len(train_log['train_total_loss'])):
            writer.writerow([
                train_log['train_total_loss'][i],
                train_log['train_semantic_loss'][i],
                train_log['train_offset_loss'][i],
                train_log['tp'][i],
                train_log['fp'][i],
                train_log['tn'][i],
                train_log['fn'][i],
                train_log['IOU'][i]
            ])
    # Save the validation log to a CSV file
    with open(valid_log_csv_path, mode='w', newline='') as valid_log_file:
        writer = csv.writer(valid_log_file)
        # Write header
        writer.writerow(['valid_total_loss', 'valid_semantic_loss', 'valid_offset_loss', 'tp', 'fp', 'tn', 'fn', 'IOU'])
        # Write data
        for i in range(len(valid_log['valid_total_loss'])):
            writer.writerow([
                valid_log['valid_total_loss'][i],
                valid_log['valid_semantic_loss'][i],
                valid_log['valid_offset_loss'][i],
                valid_log['tp'][i],
                valid_log['fp'][i],
                valid_log['tn'][i],
                valid_log['fn'][i],
                valid_log['IOU'][i]
            ])
    print(f"Training log saved to {train_log_csv_path}")
    print(f"Validation log saved to {valid_log_csv_path}")
    ###### END OF TRAINING ######

    return experinment_name

if __name__ == "__main__":

    train()