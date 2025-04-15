import random
import warnings
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner
from mmseg.core import DistEvalHook, EvalHook
from mmseg.utils import get_root_logger
from torch.utils.data import dataset, dataloader
import os
from PIL import Image
import yaml

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(x)

def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

class LoadData(dataset.Dataset):
    def __init__(self, noise_folder, perceptual_discrepancy, content_discrepancy, mask_folder, ori_folder, mode, alpha, beta):
            super(LoadData, self).__init__()
            self.mode = mode

            self.noise_folder = noise_folder 
            self.perceptual_discrepancy = perceptual_discrepancy
            self.content_discrepancy = content_discrepancy
            self.mask_folder = mask_folder
            self.ori_folder = ori_folder
            self.alpha = alpha
            self.beta = beta
            with open('sampleval.txt', 'r') as file:
                self.mask_paths = [line.strip() for line in file.readlines()]
            
    def __getitem__(self, index):
            mask_path = self.mask_paths[index]  
            (maskname, maskext) = os.path.splitext(os.path.basename(mask_path)) 

            imgidx = mask_path.split('/')[-2]
            noiseB = maskname.split("_")[-2]  
            noiseA = maskname.split("_")[-1]
            
            LR_path = f'{self.ori_folder}/{imgidx}/LR_{imgidx}.pt' 
            LR_feat = torch.load(LR_path)
            
            x_A_path = f'{self.noise_folder}/{imgidx}/noise_{noiseA}_{imgidx}.pt'    
            x_A = torch.load(x_A_path).squeeze(0)
            x_A = torch.cat([x_A,LR_feat],dim=0)
            x_B_path = f'{self.noise_folder}/{imgidx}/noise_{noiseB}_{imgidx}.pt'    
            x_B = torch.load(x_B_path).squeeze(0)
            x_B = torch.cat([x_B,LR_feat],dim=0)
            
            A_perceptual_path = f'{self.perceptual_discrepancy}/{imgidx}/{noiseA}_{imgidx}.pt'    
            B_perceptual_path = f'{self.perceptual_discrepancy}/{imgidx}/{noiseB}_{imgidx}.pt'    

            A_perceptual = torch.load(A_perceptual_path).cuda()
            B_perceptual = torch.load(B_perceptual_path).cuda()

            A_content_path = f'{self.content_discrepancy}/{imgidx}/{imgidx}_{noiseA}.pt'    
            B_content_path = f'{self.content_discrepancy}/{imgidx}/{imgidx}_{noiseB}.pt'    
            A_content = torch.load(A_content_path).cuda()
            B_content = torch.load(B_content_path).cuda()

            diff_A = self.alpha * A_content + self.beta * A_perceptual
            diff_B = self.alpha * B_content + self.beta * B_perceptual

            mask_base = torch.where(diff_B-diff_A > 0, 1, 0).cuda() 
            shape = torch.tensor([mask_base.shape[0],mask_base.shape[1]])

            idx = torch.tensor([int(imgidx.lstrip('0')), int(noiseB), int(noiseA)])
            return {'x_A': x_A.cuda(), 'mask_base': mask_base.type(torch.long).cuda(), 'x_B': x_B.cuda(), 'diff_A': diff_A.cuda(), 'diff_B': diff_B.cuda(), 'shape':shape.cuda(), 'idx':idx.cuda()}
                       
    def __len__(self):
            return len(self.mask_paths)

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    dataset = dataset
    data_loaders = [dataloader.DataLoader(
    dataset=ds,
    batch_size=100,
    shuffle=False
    ) for ds in dataset]
    
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        # cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        cfg.runner = {'type':'EpochBasedRunner', 'max_epochs':50}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = LoadData(noise_folder="./val_noise_folder", perceptual_discrepancy="./val_perceptual_discrepancy", content_discrepancy="./val_content_discrepancy", mask_folder="./val_mask", ori_folder="./val_LRimg", mode = 'eval', alpha=0.8, beta=0.2)
        val_dataloader = dataloader.DataLoader(
            dataset=val_dataset,
            batch_size=600,
            shuffle=False
            )

        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)