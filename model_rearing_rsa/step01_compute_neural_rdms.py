'''
    Compute neural rdms
    Requires visionlab_datasets to be installed
        pip install git+https://github.com/harvard-visionlab/datasets.git

    Example
    python step01_compute_neural_rdms.py compute_rdms SECTORS
'''
import os
import torch
import numpy as np
import argparse
import fire
from fastprogress import master_bar, progress_bar
from collections import defaultdict
from pathlib import Path
from visionlab_datasets.neuro.konkle_72objects import (
    NeuralDatasetSectors, NeuralDatasetGradient, StimulusDataset
)

from pdb import set_trace

parser = argparse.ArgumentParser(description='Generate LitData streaming dataset')
FLAGS, FIRE_FLAGS = parser.parse_known_args()

def compute_rdms(split, threshold=.3, min_voxels=50, recompute=False):
    if split=="SECTORS":
        brain_data = NeuralDatasetSectors()
    else:
        brain_data = NeuralDatasetGradient()
    print(brain_data)

    filename = os.path.join("results", "step01_compute_neural_rdms",
                            f"ExploringObjects_{split}_reliability{threshold}_rdms.pth")
    if os.path.isfile(filename) and not recompute:
        print(f"==> File exists, skipping: {filename}")
        return 
        
    results = defaultdict(list)
    betas = brain_data['Betas']
    reliability = brain_data['Reliability']
    mb = master_bar(betas.items())
    for brain_region, roi_betas in mb:
        for sub_num, sub_betas in enumerate(progress_bar(roi_betas, parent=mb)):
            if sub_num==1: continue
            roi_rel = reliability[brain_region][sub_num]
            if threshold is not None:
                betas_masked = sub_betas[roi_rel > threshold]
            else:
                betas_masked = sub_betas
            if len(betas_masked) < min_voxels:
                print(f"Low voxel warning, {brain_region}: sub_num={sub_num}, betas={sub_betas.shape}, reliability={roi_rel.shape}")

            RSM = np.corrcoef(betas_masked.transpose())
            RDM = torch.tensor(1 - RSM)
            results[brain_region].append(RDM)
        results[brain_region] = torch.stack(results[brain_region])
    torch.save(results, filename)
    
if __name__ == '__main__':
    fire.Fire(command=FIRE_FLAGS)