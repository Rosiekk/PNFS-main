import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
import argparse
import os
import mmcv
import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader
from mmseg.models import build_segmentor

from torch.utils.data import dataset
import os
import yaml

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

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

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        if cfg.data.test.type == 'CityscapesDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = LoadData(noise_folder="./val_noise_folder", perceptual_discrepancy="./val_perceptual_discrepancy", content_discrepancy="./val_content_discrepancy", mask_folder="./val_mask", ori_folder="./val_LRimg", mode = 'eval', alpha=0.8, beta=0.2)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        pin_memory=False
        )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cuda:0')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = False #False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        results, pred_diff, losses, accuracy  = single_gpu_test(model.cuda(), data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            for i in range(len(results)):
                results = results.squeeze()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
