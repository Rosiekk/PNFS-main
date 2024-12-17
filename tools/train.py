import os
import os.path as osp

import argparse
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

from torch.utils.data import dataset
import os

class LoadData(dataset.Dataset):
    def __init__(self, noise_folder, perceptual_discrepancy, content_discrepancy, ori_folder, mode, alpha, beta):
            super(LoadData, self).__init__()
            self.mode = mode
            self.noise_folder = noise_folder 
            self.perceptual_discrepancy = perceptual_discrepancy
            self.content_discrepancy = content_discrepancy
            self.ori_folder = ori_folder
            self.alpha = alpha
            self.beta = beta   
            with open('sample.txt', 'r') as file: # training index list
                self.mask_paths = [line.strip() for line in file.readlines()]              

    def __getitem__(self, index):
            mask_path = self.mask_paths[index]  
            (maskname, maskext) = os.path.splitext(os.path.basename(mask_path)) 
            
            idx = maskname.split("_")[0]+'_'+maskname.split("_")[1] 
            noiseA = maskname.split("_")[-1]

            x_A_path = f'{self.noise_folder}/{idx}/noise_{noiseA}_{idx}.pt'
            x_A = torch.load(x_A_path).squeeze(0)
            LR_path = f'{self.noise_folder}/{idx}/LR_{idx}.pt'
            LR_feat = torch.load(LR_path)
            x_A = torch.cat([x_A,LR_feat],dim=0)
            A_perceptual = torch.load(f'{self.perceptual_discrepancy}/{idx}/{noiseA}_{idx}.pt').squeeze(0).cuda()
            A_content = torch.load(f'{self.content_discrepancy}/{idx}/{idx}_{noiseA}.pt').squeeze(0).cuda()
            GTdiff = self.alpha * A_content + self.beta * A_perceptual
            return {'x_A': x_A.cuda(), 'mask_base': GTdiff.cuda()}
                       
    def __len__(self):
            return len(self.mask_paths)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_folder',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    logger.info(model)

    train_dataset = LoadData(noise_folder="./train_noise_folder", perceptual_discrepancy="./train_perceptual_discrepancy", content_discrepancy="./train_content_discrepancy",ori_folder="./train_LRimg", mode = 'train', alpha = 0.8, beta = 0.2)
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=1,
            PALETTE=2)
    model.CLASSES = 2 
    train_segmentor(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
