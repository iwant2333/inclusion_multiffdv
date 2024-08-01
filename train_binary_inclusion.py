import argparse
import torch
import toml
import os
import sys, logging, random, numpy as np
import time
from tqdm import tqdm

from dataset.lavdf import Lavdf, LavdfDataModule #, Metadata
# from dataset.avdf1m import AVDF1M, AVDF1MDataModule #, Metadata
from dataset.avdf1m_binary_inclusion import AVDF1M
from model import Batfd, BatfdPlus
from utils import LrLogger, EarlyStoppingLR, read_json, AverageMeter
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict
from datetime import datetime
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from torch.nn import parallel
import logging
from tqdm import tqdm

parser = argparse.ArgumentParser(description="BATFD training")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=8)

parser.add_argument("--precision", default=32)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=None)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--prefix", type=str, default=None)
parser.add_argument("--gpus", type=str, default=None)
parser.add_argument("--exp_name", type=str, default="", help="experiment name")


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def print_pass(*args, **kwargs):
    pass
    return

def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    return

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_config(cfg):
    cfg.exp_name = "avdf"
    cfg.gpus = "2,3"
    cfg.prefix = "batfdp"
    #cfg.log_path = "/home/zoloz/8T-2/zhanyi/code/LAV-DF_debug/train.log"
    cfg.log_path = "/16T-2/hanyu/shenzhuang/audio_mp4/open/train.log"
    cfg.logger = get_logger(cfg.log_path)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    cfg.prefix = os.path.join(cfg.prefix, cfg.exp_name, current_time)
    os.makedirs(cfg.prefix, exist_ok=True)

    cfg.seed = 1314
    cfg.node_rank = 0
    cfg.node_size = 1
    
    cfg.ngpus_per_node = len(cfg.gpus.split(','))
    cfg.world_size = cfg.ngpus_per_node * cfg.node_size
    cfg.dist_backend = 'nccl'
    cfg.dist_url = 'tcp://localhost:234{}'.format(random.randint(10,99))
    return cfg

def main_worker(gpu, cfg):
    if gpu == 0:
        cfg.debug_print = print_flush
    elif gpu != 0:
        cfg.debug_print = print_pass
    else: pass
    print('current gpu_id: {}'.format(gpu))
    cfg.gpu = gpu
    cfg.map_loc = 'cuda:{}'.format(cfg.gpu)
    cfg.device = torch.device(cfg.map_loc)
    cfg.global_rank = cfg.node_rank * cfg.ngpus_per_node + gpu

    cfg.debug_print('final cfg: \n{}'.format(cfg))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)

    torch.distributed.init_process_group(
        backend=cfg.dist_backend, init_method=cfg.dist_url,
        rank=cfg.global_rank, world_size=cfg.world_size
    )

    torch.cuda.set_device(cfg.gpu)
    device = torch.cuda.current_device()
    print('current gpu device: ', device)
    train_worker(cfg)

def main():
    mycfg = update_config(EasyDict(vars(parser.parse_args())))
    device_ids = mycfg.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids

    ngpus_per_node = mycfg.ngpus_per_node
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(mycfg,))

def train_worker(cfg):
    
    global mycfg
    mycfg = cfg

    config = toml.load(mycfg.config)

    best_acc, best_tpr = 0.0, 0.0
    writer_count, writer_count_test = 0, 0

    writer = SummaryWriter(mycfg.prefix)

    learning_rate = config["optimizer"]["learning_rate"]

    v_encoder_type = config["model"]["video_encoder"]["type"]
    a_encoder_type = config["model"]["audio_encoder"]["type"]

    if config["model_type"] == "batfd_plus":
        model = BatfdPlus(
            v_encoder=v_encoder_type,
            a_encoder=config["model"]["audio_encoder"]["type"],
            frame_classifier=config["model"]["frame_classifier"]["type"],
            ve_features=config["model"]["video_encoder"]["hidden_dims"],
            ae_features=config["model"]["audio_encoder"]["hidden_dims"],
            v_cla_feature_in=config["model"]["video_encoder"]["cla_feature_in"],
            a_cla_feature_in=config["model"]["audio_encoder"]["cla_feature_in"],
            boundary_features=config["model"]["boundary_module"]["hidden_dims"],
            boundary_samples=config["model"]["boundary_module"]["samples"],
            temporal_dim=config["num_frames"],
            max_duration=config["max_duration"],
            weight_frame_loss=config["optimizer"]["frame_loss_weight"],
            weight_modal_bm_loss=config["optimizer"]["modal_bm_loss_weight"],
            weight_contrastive_loss=config["optimizer"]["contrastive_loss_weight"],
            contrast_loss_margin=config["optimizer"]["contrastive_loss_margin"],
            cbg_feature_weight=config["optimizer"]["cbg_feature_weight"],
            prb_weight_forward=config["optimizer"]["prb_weight_forward"],
            weight_decay=config["optimizer"]["weight_decay"],
            learning_rate=learning_rate,
            distributed=mycfg.ngpus_per_node > 1
        )
        require_match_scores = True
        get_meta_attr = BatfdPlus.get_meta_attr

    elif config["model_type"] == "batfd":
        model = Batfd(
            v_encoder=config["model"]["video_encoder"]["type"],
            a_encoder=config["model"]["audio_encoder"]["type"],
            frame_classifier=config["model"]["frame_classifier"]["type"],
            ve_features=config["model"]["video_encoder"]["hidden_dims"],
            ae_features=config["model"]["audio_encoder"]["hidden_dims"],
            v_cla_feature_in=config["model"]["video_encoder"]["cla_feature_in"],
            a_cla_feature_in=config["model"]["audio_encoder"]["cla_feature_in"],
            boundary_features=config["model"]["boundary_module"]["hidden_dims"],
            boundary_samples=config["model"]["boundary_module"]["samples"],
            temporal_dim=config["num_frames"],
            max_duration=config["max_duration"],
            weight_frame_loss=config["optimizer"]["frame_loss_weight"],
            weight_modal_bm_loss=config["optimizer"]["modal_bm_loss_weight"],
            weight_contrastive_loss=config["optimizer"]["contrastive_loss_weight"],
            contrast_loss_margin=config["optimizer"]["contrastive_loss_margin"],
            weight_decay=config["optimizer"]["weight_decay"],
            learning_rate=learning_rate,
            distributed=mycfg.ngpus_per_node > 1
        )
        require_match_scores = False
        get_meta_attr = Batfd.get_meta_attr
    else:
        raise ValueError("Invalid model type")

    dataset = config["dataset"]

    if dataset == "lavdf":
        # metadata_train: List[Metadata] = read_json(os.path.join(mycfg.data_root, "train_metadata.json"), lambda x: Metadata(**x))
        # metadata_val: List[Metadata] = read_json(os.path.join(mycfg.data_root, "val_metadata.json"), lambda x: Metadata(**x))

        train_dataset = Lavdf("train", root=mycfg.data_root, frame_padding=config["num_frames"], max_duration=config["max_duration"], metadata=None, get_meta_attr=get_meta_attr, require_match_scores=require_match_scores)
        val_dataset = Lavdf("val", root=mycfg.data_root, frame_padding=config["num_frames"], max_duration=config["max_duration"], metadata=None, get_meta_attr=get_meta_attr, require_match_scores=require_match_scores)

        train_dataloader = DataLoader(train_dataset, batch_size=mycfg.batch_size, num_workers=mycfg.num_workers, sampler=RandomSampler(train_dataset))
        val_dataloader = DataLoader(val_dataset, batch_size=mycfg.batch_size, num_workers=mycfg.num_workers, sampler=SequentialSampler(val_dataset))
    elif dataset == "avdf1m":
        # metadata_train: List[Metadata] = read_json(os.path.join(mycfg.data_root, "train_metadata.json"), lambda x: Metadata(**x))
        # metadata_val: List[Metadata] = read_json(os.path.join(mycfg.data_root, "val_metadata.json"), lambda x: Metadata(**x))

        train_dataset = AVDF1M("trainset", root=mycfg.data_root, frame_padding=config["num_frames"], max_duration=config["max_duration"], metadata=None, get_meta_attr=get_meta_attr, require_match_scores=require_match_scores)
        val_dataset = AVDF1M("valset", root=mycfg.data_root, frame_padding=config["num_frames"], max_duration=config["max_duration"], metadata=None, get_meta_attr=get_meta_attr, require_match_scores=require_match_scores)

        train_dataloader = DataLoader(train_dataset, batch_size=mycfg.batch_size, num_workers=mycfg.num_workers, sampler=torch.utils.data.distributed.DistributedSampler(train_dataset))
        val_dataloader = DataLoader(val_dataset, batch_size=mycfg.batch_size, num_workers=mycfg.num_workers, sampler=SequentialSampler(val_dataset))
    else:
        raise ValueError("Invalid dataset type")

    cfg.logger.info(len(train_dataloader))
    cfg.logger.info(len(val_dataloader))
    model = model.to(mycfg.gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    mycfg.debug_print("Using torch.nn.SyncBatchNorm.convert_sync_batchnorm")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    mycfg.debug_print("Using torch.nn.parallel.DistributedDataParallel")
    model = parallel.DistributedDataParallel(model, device_ids=[mycfg.gpu], find_unused_parameters=True)

    since = time.time()
    mycfg.debug_print('-' * 10)
    for epoch in range(mycfg.max_epochs):
        train_model(model, train_dataloader, optimizer, scheduler, criterion, epoch, writer_count, writer)
        if (epoch + 1) % 2 != 0 : continue
        validate_model(model, val_dataloader, epoch, writer_count_test, writer)

        if mycfg.global_rank == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }
            torch.save(state, os.path.join(mycfg.prefix, f'checkpoint_{epoch}.pth.tar'))

    time_elapsed = time.time() - since
    mycfg.debug_print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))




def train_model(model, train_dataloader, optimizer, scheduler, criterion, epoch, writer_count, writer):
    model.train(True)
    batch_nums = len(train_dataloader)
    mycfg.logger.info('-' * 10)
    mycfg.logger.info(f'train batch_nums per process/gpu: {batch_nums}')
    mycfg.debug_print('-' * 10)
    mycfg.debug_print(f'train batch_nums per process/gpu: {batch_nums}')
    am_epoch_time = AverageMeter()
    for i, batch in enumerate(train_dataloader):
        _, _, video_level_label = batch
        batch_time = time.time()
        optimizer.zero_grad()
        model.zero_grad()
        # import ipdb;ipdb.set_trace()
        output = model(batch)
        # print(video_level_label)
        label_list = [int(x) for x in video_level_label]
        video_level_label = torch.tensor(label_list).to(output.device)
        loss = criterion(output, video_level_label)

        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_time
        epoch_time = batch_nums * batch_time / 60 / 60
        am_epoch_time.update(epoch_time)

        curr_lr = float(optimizer.param_groups[0]['lr'])
        print_info = f'Epoch: {epoch}/{mycfg.max_epochs} [{i}/{len(train_dataloader)} ({100.*i/len(train_dataloader):.0f}%)], '
        print_info += f'LR: {curr_lr:.12f}, Time: {am_epoch_time.avg:.4f}h, '

        writer.add_scalar('loss', loss.mean(), writer_count)
        print_info += f'Loss: {loss.mean():.6f}, '

        mycfg.debug_print(print_info[:-2])
        mycfg.logger.info(print_info[:-2])
        writer.add_scalar("Training/Training_LR", float(optimizer.param_groups[0]['lr']), writer_count)

        writer_count += 1
        scheduler.step(epoch)

        if mycfg.global_rank == 0:
            if i % 10000 == 0:
                checkpoint_path = os.path.join(mycfg.prefix, f'checkpoint_{epoch}_{i}_loss{loss}.pth')
                torch.save({
                    'epoch': epoch,
                    'iteration': i,
                    'model_state_dict': model.state_dict(),
                }, checkpoint_path)
    
    return

def validate_model(model, val_dataloader, epoch, writer_count_test, writer):
    num_classes = 2
    from sklearn.preprocessing import label_binarize
    model.eval()
    mycfg.debug_print('-' * 10)
    mycfg.logger.info('-' * 10)
    batch_nums = len(val_dataloader)
    softmax_layer = torch.nn.Softmax(dim=1)
    mycfg.debug_print(f'val batch_nums per process/gpu: {batch_nums}')
    mycfg.logger.info(f'val batch_nums per process/gpu: {batch_nums}')

    for i, batch in tqdm(enumerate(val_dataloader)):
        _, _, video_level_label = batch
        output = model(batch)
        output = softmax_layer(output)

        y_labels_onehot = label_binarize(video_level_label, classes=np.arange(num_classes))

        y_score_preds = output
        y_score_labels = y_score_preds.argmax(axis=1)
        y_score_onehot = label_binarize(y_score_labels, classes=np.arange(num_classes))

        from sklearn.metrics import accuracy_score
        accuracy  = accuracy_score(y_labels, y_score_labels)
        mycfg.debug_print(prefix+' accuracy: {}'.format(accuracy))
        print(f'==> Epoch: {epoch:d}, Test Accuracy: {accuracy:.8f}%\n')
        state_info = f'==> Epoch: {epoch:d}, Test Accuracy: {accuracy:.8f}%\n'

        if mycfg.global_rank==0:
            stat_path = os.path.join(mycfg.prefix, 'test.log')
            with open(stat_path, 'a+') as f: f.write(stat_info)
            mycfg.debug_print(stat_info)

        return accuracy


if __name__ == '__main__':
    main()
