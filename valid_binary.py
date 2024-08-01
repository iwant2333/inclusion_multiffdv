import argparse

import toml
import torch
import os

from dataset.avdf1m_test import AVDF1M_test
from dataset.avdf1m_binary_inclusion import AVDF1M
from metrics import AP, AR
# from model import Batfd, BatdfPlus
# from model.batfd_plus_binary import BatdfPlus
from model.batfd import Batfd
from model.batfd_plus_binary import BatfdPlus
from post_process import post_process
from utils import read_json
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser(description="BATFD evaluation")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--modalities", type=str, nargs="+", default=["fusion"])
parser.add_argument("--subset", type=str, nargs="+", default=["full"])
parser.add_argument("--gpus", type=int, default=1)


def evaluate_avdf1m(config, args):
    model_name = config["name"]
    model_type = config["model_type"]

    if config["model_type"] == "batfd_plus":
        model = BatfdPlus(
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
            cbg_feature_weight=config["optimizer"]["cbg_feature_weight"],
        )
        checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda')

        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    else:
        raise ValueError("Invalid model type")

    test_dataset = AVDF1M(
        "valset",
        root=args.data_root,
        frame_padding=config["num_frames"],
        max_duration=config["max_duration"],
        return_file_name=True,
    )

    device = torch.device("cuda:6" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    
    model.to(device)

    results_path = './prediction.txt'

    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=8)
    softmax_layer = torch.nn.Softmax(dim=1)
    with open(results_path, 'w') as result_file:
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_dataloader)):
                # import ipdb;ipdb.set_trace()
                video, audio, video_level_label, filenames = batch
                # batch = video, audio, video_level_label
                video = video.to(device)
                audio = audio.to(device)
                batch_res = video, audio, None
                # batch_res = batch_res.to(device)
                output = model.forward(batch_res)
                output = softmax_layer(output)
                probs_label_0 = output[:, 0]
                probs_label_1 = output[:, 1:].sum(dim=1)
                for filename, prob_label_1 in zip(filenames, probs_label_1):
                    result_file.write(f'{filename};{prob_label_1.item()}\n')
                # import ipdb;ipdb.set_trace()


if __name__ == '__main__':
    args = parser.parse_args()
    config = toml.load(args.config)
    torch.backends.cudnn.benchmark = True
    if config["dataset"] == "avdf1m":
        evaluate_avdf1m(config, args)
    else:
        raise NotImplementedError