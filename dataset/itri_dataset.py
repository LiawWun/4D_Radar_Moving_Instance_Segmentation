import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class ITRI_Dataset(Dataset):

    def __init__(self, mode = 'train', transform = None, num_frame = 2):

        self.mode = mode
        self.num_frame = num_frame
        assert 2 <= self.num_frame <= 5, "num_frame must be between 2 and 5 (inclusive)."
        self.transform = transform

        self.base_folder = "/home/ee904/Desktop/hand_over_draft/experinment_on_itri/itri_dataset"
        self.train_seq = ["seq0", "seq3", "seq4", "seq5", "seq6", "seq7", "seq9", "seq12", "seq13", "seq14", "seq16", "seq17", "seq18", "seq19", "seq20", "seq21", "seq22", "seq23"]
        self.val_seq   = ["seq1", "seq2", "seq8", "seq11", "seq15"]
        
        self.seq = self.train_seq if self.mode == "train" else self.val_seq
        self.total_data = []
        self.frame_idx = []
        self.total_camera_path = []
        for seq_name in self.seq:

            radar_folder = os.path.join(self.base_folder, seq_name)
            radar_name_list = sorted(os.listdir(radar_folder))
            radar_path_list = [os.path.join(radar_folder, radar_name) for radar_name in radar_name_list]
            self.total_data += radar_path_list
            self.frame_idx += [seq_name + "_" + radar_name.split(".")[0] for radar_name in radar_name_list]

            camera_folder = os.path.join(self.base_folder, seq_name, "synchronized_data_" + seq_name.split("seq")[1], "camera", "front")
            camera_path_list = [os.path.join(camera_folder, radar_name.split('.bin')[0] + ".jpg") for radar_name in radar_name_list]
            self.total_camera_path += camera_path_list

        idx = 0
        self.cache = {}
        self.total_points = np.zeros(2, dtype = np.float32)
        self.labelweights = np.zeros(2, dtype = np.float32)
        for radar_pc_path in tqdm(self.total_data, desc = "Preparing data"):
            
            #[x, y, z, RCS, v_r, v_r_compensated, moving_mask, cluster_id, xc, yc, zc, time_idx]
            raw_pc = np.fromfile(radar_pc_path, dtype = np.float32).reshape(-1, 12)

            if self.transform:
                raw_pc_xyz   = raw_pc[:, [0, 1, 2]] # [x, y, z]
                raw_pc_feat  = raw_pc[:, [3, 4, 5]] # [RCS, v_r, v_r_compensated]
                raw_pc_label = raw_pc[:, [6, 7, 11]] # [moving_mask, cluster_id, time_idx]
                raw_pc_center = raw_pc[:, [8, 9, 10]] # [xc, yc, zc]
                raw_pc_xyz_transform, raw_pc_feat_transform, raw_pc_label_transform, raw_pc_center_transform  = self.transform(raw_pc_xyz, raw_pc_feat, raw_pc_label, raw_pc_center)
                raw_pc = np.concatenate((raw_pc_xyz_transform, raw_pc_feat_transform, raw_pc_label_transform[:, :2], raw_pc_center_transform, raw_pc_label_transform[:, [2]]), axis = 1)
            
            # Calculate the semantic label weight
            pc_current_indices  = np.where(raw_pc[:, 11] == 0)[0]
            current_pc = raw_pc[pc_current_indices, :]
            current_pc_moving_mask = current_pc[:, 6]
            self.total_points[0] += np.where(current_pc_moving_mask == 0)[0].size
            self.total_points[1] += np.where(current_pc_moving_mask == 1)[0].size

            self.cache[idx] = (raw_pc)
            idx += 1

        self.labelweights = np.power(np.sum(self.total_points) / self.total_points, 1 / 1.5)

    def __getitem__(self, index):
        
        # [x, y, z, RCS, v_r, v_r_compensated, moving_mask, cluster_id, xc, yc, zc, time_idx]
        # time index 0: current, 
        # time index 1: t - 1, 
        # time index 2: t - 2,
        # time index 3: t - 3,
        # time index 4: t - 4,

        raw_pc = self.cache[index]
        if self.mode == "train":
            shuffled_indices = np.random.permutation(raw_pc.shape[0])
            raw_pc = raw_pc[shuffled_indices]

        coords = []
        feats = []
        semantic_labels = []
        pc_shift_labels = []

        for t in range(self.num_frame):

            raw_pc_t_indices = np.where(raw_pc[:, 11] == t)[0]
            raw_pc_t = raw_pc[raw_pc_t_indices, :]

            # Extract coordinate (x, y, z)
            coord = raw_pc_t[:, [0, 1, 2]]
            coord_t = coord.transpose(1, 0)
            coords.append(coord_t)

            # Extract feature (RCS, v_r_compensated)
            feat = raw_pc_t[:, [3, 5]]
            feat_t = feat.transpose(1, 0)
            feats.append(feat_t)

            # Extarct semantic label
            semantic_label = raw_pc_t[:, [6]]
            semantic_labels.append(semantic_label)

            # Extract centroid label
            pc_centroid = raw_pc_t[:, [8, 9, 10]]
            pc_shift_label = pc_centroid - coord
            pc_shift_labels.append(pc_shift_label)

        coord_3d = np.stack(coords) #[T, 3, N]
        feat_3d  = np.stack(feats) #[T, D, N]
        semantic_label_3d = np.stack(semantic_labels) #[T, N, 1]
        pc_shift_label_3d = np.stack(pc_shift_labels) #[T, N, 3]

        coord_3d = coord_3d.astype(np.float32)
        feat_3d = feat_3d.astype(np.float32)
        semantic_label_3d = semantic_label_3d.astype(np.float32)
        pc_shift_label_3d = pc_shift_label_3d.astype(np.float32)

        return coord_3d, feat_3d, semantic_label_3d, pc_shift_label_3d

    def __len__(self):

        return len(list(self.cache.keys()))
    
    def get_frame_data(self, index):

        raw_pc = self.cache[index]
        raw_pc_current_indices = np.where(raw_pc[:, 11] == 0)[0]
        raw_current_pc = raw_pc[raw_pc_current_indices, :]
        return raw_current_pc



if __name__ == "__main__":

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    frame_num = 3
    vod_data = ITRI_Dataset(mode = "train", num_frame = frame_num)
    coord_3ds, feat_3ds, semantic_label_3ds, pc_shift_label_3ds = vod_data.__getitem__(850)

    print(coord_3ds.shape)
    print(feat_3ds.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    cmap = plt.colormaps['Blues'] 
    color_range = np.linspace(0.8, 0.2, frame_num) # From dark to light
    color_list = [cmap(val) for val in color_range]

    ax1.set_title("Raw data with different time index")
    ax2.set_title("Semantic result")
    ax3.set_title("Centroid point")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim([-25, 25])
        ax.set_ylim([-1, 50])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

    # Draw different time index data
    legend_patches = []
    for t in range(frame_num - 1, -1, -1):
        coord_3d = coord_3ds[t, :, :]
        coord_3d = coord_3d.transpose()
        ax1.scatter(-coord_3d[:, 1], coord_3d[:, 0], color = color_list[t])
        name = f"t - {t}" if t != 0 else "t"
        patch = mpatches.Patch(color=color_list[t], label= name)
        legend_patches.append(patch)

    ax1.legend(handles=legend_patches[::-1], loc = 'upper right')

    # Draw semantic label
    current_coord_3d = coord_3ds[0, :, :].transpose()
    current_semantic_label = semantic_label_3ds[0, :, :].squeeze()
    moving_indices = current_semantic_label == 1.0
    static_indices = current_semantic_label == 0.0
    ax2.scatter(-current_coord_3d[static_indices, 1], current_coord_3d[static_indices, 0], color = 'lightgray', label = "Static")
    ax2.scatter(-current_coord_3d[moving_indices, 1], current_coord_3d[moving_indices, 0], color = 'red', label = "Moving")
    ax2.legend(loc = 'upper right')

    # Draw center prediction
    current_coord_3d = coord_3ds[0, :, :].transpose()
    current_pc_shift_label = pc_shift_label_3ds[0, :, :]

    shifted_coord = current_coord_3d[moving_indices, :]
    shift_vector = current_pc_shift_label[moving_indices, :]
    centroid = shifted_coord + shift_vector

    ax3.scatter(-current_coord_3d[static_indices, 1], current_coord_3d[static_indices, 0], color = 'lightgray', label = "Static")
    ax3.scatter(-centroid[:, 1], centroid[:, 0], color = 'red', label = "Centroid")
    ax3.legend(loc = 'upper right')

    plt.tight_layout()
    plt.show()
