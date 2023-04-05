import torch
import numpy as np
from scipy.ndimage import zoom
from scipy.io import savemat, loadmat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2)-0.5, int(h/2)-0.5)
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

class PairDataset(Dataset):
    def __init__(self, data_path, config, correlation, subset_idx_list):
        self.tof_list = []
        self.ss_list = []
        self.subset_idx_list = subset_idx_list

        for i in range(1, 201):
            tof = loadmat(f'{data_path}/tof_{config}/grf_{correlation}{i}_tof.mat')
            self.tof_list.append(torch.from_numpy(tof['tof_array'].astype(float)))
            ss = loadmat(f'{data_path}/sound_speed/grf_{correlation}{i}_sound_speed.mat')
            self.ss_list.append(torch.from_numpy(ss['sound_speed'].astype(float)))
        
        self.tof_bulk = torch.stack(self.tof_list, 0)
        self.ss_bulk = torch.stack(self.ss_list, 0)
        self.tof_min, self.tof_max = self.tof_bulk.min(), self.tof_bulk.max()
        self.ss_min, self.ss_max = self.ss_bulk.min(), self.ss_bulk.max()
        
        mask = create_circular_mask(ss['sound_speed'].shape[0], ss['sound_speed'].shape[1], radius = 40)
        self.masked_img = np.ones((128,128))
        self.masked_img[~mask] = 0
        
    def __len__(self):
        return len(self.subset_idx_list)
        
    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tof = torch.zeros((3, *self.tof_list[self.subset_idx_list[idx]].shape))
        xx = torch.linspace(0, self.tof_list[self.subset_idx_list[idx]].shape[0]-1, self.tof_list[self.subset_idx_list[idx]].shape[0])/self.tof_list[self.subset_idx_list[idx]].shape[0]
        yy = torch.linspace(0, self.tof_list[self.subset_idx_list[idx]].shape[1]-1, self.tof_list[self.subset_idx_list[idx]].shape[1])/self.tof_list[self.subset_idx_list[idx]].shape[1]
        grid_xx, grid_yy = torch.meshgrid(xx, yy, indexing='ij')
        tof[0] = (self.tof_list[self.subset_idx_list[idx]]-self.tof_min)/(self.tof_max-self.tof_min)
        tof[1] = grid_xx
        tof[2] = grid_yy
        ss = ((self.ss_list[self.subset_idx_list[idx]]-self.ss_min)/(self.ss_max-self.ss_min)).unsqueeze(0)#.unsqueeze(0)

        sample = {  'abs_id'   : self.subset_idx_list[idx],
                    'rel_id'   : idx,
                    'input'    : tof.float(),
                    'output'   : ss.float()*torch.from_numpy(self.masked_img).float()}
        return sample