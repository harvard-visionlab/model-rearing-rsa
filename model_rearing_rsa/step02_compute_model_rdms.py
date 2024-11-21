'''
    Compute neural rdms
    Requires visionlab_datasets to be installed
        pip install git+https://github.com/harvard-visionlab/datasets.git

    Example
    python step02_compute_model_rdms.py compute_rdms --width w1 --task supervised
    
'''
import os
import torch
import numpy as np
import argparse
import fire
from fastprogress import master_bar, progress_bar
from collections import defaultdict
from pathlib import Path

from models.alexnets import load_model

from pdb import set_trace

parser = argparse.ArgumentParser(description='Generate LitData streaming dataset')
FLAGS, FIRE_FLAGS = parser.parse_known_args()

default_layers = [
    'model.base_arch.backbone.conv_block_1',
    'model.base_arch.backbone.conv_block_2',
    'model.base_arch.backbone.conv_block_3',
    'model.base_arch.backbone.conv_block_4',
    'model.base_arch.backbone.conv_block_5',
    'model.base_arch.projector.layers.fc_block_6',
    'model.base_arch.projector.layers.fc_block_7',
    'model.base_arch.projector.layers.fc_block_8',
]

def compute_rdms(width="w1", task="supervised", layer_names=default_layers, recompute=False, device=None):
    
    filename = os.path.join("results", "step02_compute_model_rdms",
                            f"alexnet2023_{width}_{task}_rdms.pth")
    
    if os.path.isfile(filename) and not recompute:
        print(f"==> File exists, skipping: {filename}")
        return 

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    results = defaultdict(list)
    model = load_model(width=width, task=task, device=device)
    
    # mb = master_bar(betas.items())
    # for brain_region, roi_betas in mb:
    #     for sub_num, sub_betas in enumerate(progress_bar(roi_betas, parent=mb)):
    #         if sub_num==1: continue
    #         roi_rel = reliability[brain_region][sub_num]
    #         if threshold is not None:
    #             betas_masked = sub_betas[roi_rel > threshold]
    #         else:
    #             betas_masked = sub_betas
    #         if len(betas_masked) < min_voxels:
    #             print(f"Low voxel warning, {brain_region}: sub_num={sub_num}, betas={sub_betas.shape}, reliability={roi_rel.shape}")

    #         RSM = np.corrcoef(betas_masked.transpose())
    #         RDM = torch.tensor(1 - RSM)
    #         results[brain_region].append(RDM)
    #     results[brain_region] = torch.stack(results[brain_region])
    # torch.save(results, filename)
    
if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)