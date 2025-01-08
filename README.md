# PNFS-main
Code implementation of Restabilizing Diffusion Models with Predictive Noise Fusion Strategy for Image Super-Resolution

> **Abstract:** Diffusion models are prominent in image generation for producing detailed and realistic images from Gaussian noises. However, they often encounter instability issues in image restoration tasks, e.g., super-resolution. Existing methods typically rely on multiple runs to find an initial noise that produces a reasonably restored image. Unfortunately, these methods are computationally expensive and time-consuming without guaranteeing stable and consistent performance. To address these challenges, we propose a novel Predictive Noise Fusion Strategy (PNFS) that predicts pixel-wise errors in the restored image and combines different noises to generate a more effective noise. Extensive experiments show that PNFS significantly improves the stability and performance of diffusion models in super-resolution, both quantitatively and qualitatively. Furthermore, PNFS can be flexibly integrated into various diffusion models to enhance their stability.


## Dependencies

- Python 3.8
- PyTorch 1.13.0+cu116

```bash
conda create -n pnfs python=3.8
conda activate pnfs
pip install -r requirements.txt
```

## Quick Start
### Discrepancy Prediction Module
Run the following scripts. The code is partially based on [Segformer](https://github.com/NVlabs/SegFormer.git).
```shell
python tools/train.py local_configs/model/B1/b1.512x512.ade.160k.py
  ```
### Probabilistic Fusion Module

  ```shell
import torch
import torch.nn.functional as F

def sample_index_matrix(softmax_matrix, num_samples=10):
    """
    Perform multiple samplings based on the softmax output matrix to get several index matrices.
    
    Parameters:
        softmax_matrix (torch.Tensor): A softmax output matrix with shape (H, W, 2), representing the probability values [p0, p1] at each position.
        num_samples (int): The number of samplings to be performed, i.e., the number of index matrices to generate.
    
    Returns:
        list of torch.Tensor: A list containing multiple index matrices, each with shape (H, W).
    """
    H, W, _ = softmax_matrix.shape
    p0 = softmax_matrix[..., 0]
    p1 = softmax_matrix[..., 1] 
    samples = torch.multinomial(torch.stack([p0, p1], dim=2).view(H * W, 2), num_samples, replacement=True)
    samples = samples.view(H, W, num_samples)
    index_matrices = samples.permute(2, 0, 1)
    
    return index_matrices

def gen_fused_noise(index_matrices, noiseA, noiseB, fusemask, K=5):
    """
    Generate the fused noise based on the sampled index matrices.
    
    Parameters:
        index_matrices (list of torch.Tensor): A list of index matrices, each with shape (H, W).
        noiseA (torch.Tensor): The input gaussian noiseA.
        noiseB (torch.Tensor): The input gaussian noiseB.
        fusemask (torch.Tensor): The mask matrix M in Eq. (6).
        K (int): The number of matrices with the smallest new mean to select.
    
    Returns:
        torch.Tensor: The best noiseA matrix.
    """

    selected_matrices = sorted(index_matrices, key=lambda m: torch.mean((1. - m) * noiseA + m * noiseB))[:K]
    best_ratio = float('-inf')
    best_xT = None
    for new_mask in selected_matrices:
        new_xT = (1. - new_mask) * noiseA + new_mask * noiseB
        new_ratio = torch.sum(new_mask == fusemask).item() / new_mask.numel()
        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_xT = new_xT
            
    return best_xT

def pfm_runner(A_discrepancy, B_discrepancy, noiseA, noiseB, fusemask):
    """  
    Parameters:
        A_discrepancy (torch.Tensor): The discrepancy matrice of noiseA generated with DPM.
        B_discrepancy (torch.Tensor): The discrepancy matrice of noiseB generated with DPM.
        noiseA (torch.Tensor): The input gaussian noiseA.
        noiseB (torch.Tensor): The input gaussian noiseB.
        fusemask (torch.Tensor): The mask matrix M in Eq. (6).
    
    Returns:
        torch.Tensor: The fused noise matrix.
    """

    combined_matrix = torch.cat(((1.0 / B_discrepancy).unsqueeze(2), (1.0 / A_discrepancy).unsqueeze(2)), dim=2) 
    softmax_matrix = F.softmax(combined_matrix, dim=2)
    index_matrices = sample_index_matrix(softmax_matrix, num_samples=10)
    fused_xT = gen_fused_noise(index_matrices, noiseA, noiseB, fusemask, K=5)
    
    return fused_xT
  ```
