from typing import Dict, Optional, Union, Sequence, Tuple
import torch.nn.functional as F


import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# from dataset.lavdf import Metadata
from dataset.avdf1m import Metadata
from loss import MaskedFrameLoss, MaskedContrastLoss, MaskedBsnppLoss
from .audio_encoder import get_audio_encoder
from .boundary_module_plus import BoundaryModulePlus, NestedUNet
from .frame_classifier import FrameLogisticRegression
from .fusion_module import ModalFeatureAttnBoundaryMapFusion, ModalFeatureAttnCfgFusion
from .video_encoder import get_video_encoder
from utils import write_loss_to_file
from .cross_attn import MultiModalCrossAttention


class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=2):
        super(BinaryClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BatfdPlus(nn.Module):

    def __init__(self,
        v_encoder: str = "c3d", a_encoder: str = "cnn", frame_classifier: str = "lr",
        ve_features=(64, 96, 128, 128), ae_features=(32, 64, 64), v_cla_feature_in=256, a_cla_feature_in=256,
        boundary_features=(512, 128), boundary_samples=10, temporal_dim=512, max_duration=40,
        weight_frame_loss=2., weight_modal_bm_loss=1., weight_contrastive_loss=0.1, contrast_loss_margin=0.99,
        cbg_feature_weight=0.01, prb_weight_forward=1.,
        weight_decay=0.0001, learning_rate=0.0002, distributed=False
    ):
        super().__init__()
        # self.save_hyperparameters()

        self.cla_feature_in = v_cla_feature_in
        self.temporal_dim = temporal_dim

        self.video_encoder = get_video_encoder(v_cla_feature_in, temporal_dim, v_encoder, ve_features)
        self.audio_encoder = get_audio_encoder(a_cla_feature_in, temporal_dim, a_encoder, ae_features)

        if frame_classifier == "lr":
            self.video_frame_classifier = FrameLogisticRegression(n_features=v_cla_feature_in)
            self.audio_frame_classifier = FrameLogisticRegression(n_features=a_cla_feature_in)

        assert self.video_encoder and self.audio_encoder and self.video_frame_classifier and self.audio_frame_classifier

        assert v_cla_feature_in == a_cla_feature_in

        v_bm_in = v_cla_feature_in + 1
        a_bm_in = a_cla_feature_in + 1

        # Complementary Boundary Generator in BSN++ mechanism
        self.video_comp_boundary_generator = NestedUNet(in_ch=v_bm_in, out_ch=2)
        self.audio_comp_boundary_generator = NestedUNet(in_ch=a_bm_in, out_ch=2)

        # Proposal Relation Block in BSN++ mechanism
        self.video_boundary_module = BoundaryModulePlus(v_bm_in, boundary_features, boundary_samples, temporal_dim,
            max_duration
        )
        self.audio_boundary_module = BoundaryModulePlus(a_bm_in, boundary_features, boundary_samples, temporal_dim,
            max_duration
        )

        if cbg_feature_weight > 0:
            self.cbg_fusion_start = ModalFeatureAttnCfgFusion(v_bm_in, a_bm_in)
            self.cbg_fusion_end = ModalFeatureAttnCfgFusion(v_bm_in, a_bm_in)
        else:
            self.cbg_fusion_start = None
            self.cbg_fusion_end = None

        self.prb_fusion_p = ModalFeatureAttnBoundaryMapFusion(v_bm_in, a_bm_in, max_duration)
        self.prb_fusion_c = ModalFeatureAttnBoundaryMapFusion(v_bm_in, a_bm_in, max_duration)
        self.prb_fusion_p_c = ModalFeatureAttnBoundaryMapFusion(v_bm_in, a_bm_in, max_duration)

        self.frame_loss = MaskedFrameLoss(BCEWithLogitsLoss())
        self.contrast_loss = MaskedContrastLoss(margin=contrast_loss_margin)
        self.bm_loss = MaskedBsnppLoss(cbg_feature_weight, prb_weight_forward)
        self.weight_frame_loss = weight_frame_loss
        self.weight_modal_bm_loss = weight_modal_bm_loss
        self.weight_contrastive_loss = weight_contrastive_loss / (v_cla_feature_in * temporal_dim)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed

        # self.cross_attn = MultiModalCrossAttention(512, 8)

        self.input_size = 512*512
        self.fc_binary = BinaryClassifier(self.input_size)
        # self.class_loss = nn.CrossEntropyLoss()

    def scaled_dot_product_attention(self, query, key, value):
        d_k = query.size(-1)
        d_k_tensor = torch.tensor(d_k, dtype=torch.float)
        #print(d_k.shape)
        #print(query.shape)
        #print(key.shape)
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k_tensor)
        attn_weights = F.softmax(attn_logits, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights
    
    def forward_all(self, video: Tensor, audio: Tensor) -> Sequence[Tensor]:
        a_bm_in, a_features, a_frame_cla, v_bm_in, v_features, v_frame_cla = self.forward_features(audio, video)

        # a_v_features = torch.cat((a_features, v_features), dim=2)

        # modal boundary module
        v_bm_map_p, v_bm_map_c, v_bm_map_p_c = self.video_boundary_module(v_bm_in)
        a_bm_map_p, a_bm_map_c, a_bm_map_p_c = self.audio_boundary_module(a_bm_in)

        # complementary boundary generator
        if self.cbg_fusion_start is not None:
            v_cbg_feature, v_cbg_start, v_cbg_end = self.forward_video_cbg(v_bm_in)
            a_cbg_feature, a_cbg_start, a_cbg_end = self.forward_audio_cbg(a_bm_in)
        else:
            v_cbg_feature, v_cbg_start, v_cbg_end = None, None, None
            a_cbg_feature, a_cbg_start, a_cbg_end = None, None, None

        # boundary map modal attention fusion
        fusion_bm_map_p = self.prb_fusion_p(v_bm_in, a_bm_in, v_bm_map_p, a_bm_map_p)
        fusion_bm_map_c = self.prb_fusion_c(v_bm_in, a_bm_in, v_bm_map_c, a_bm_map_c)
        fusion_bm_map_p_c = self.prb_fusion_p_c(v_bm_in, a_bm_in, v_bm_map_p_c, a_bm_map_p_c)

        # complementary boundary generator modal attention fusion
        if self.cbg_fusion_start is not None:
            fusion_cbg_start = self.cbg_fusion_start(v_bm_in, a_bm_in, v_cbg_start, a_cbg_start)
            fusion_cbg_end = self.cbg_fusion_end(v_bm_in, a_bm_in, v_cbg_end, a_cbg_end)
        else:
            fusion_cbg_start = None
            fusion_cbg_end = None

        return (
            fusion_bm_map_p, fusion_bm_map_c, fusion_bm_map_p_c, fusion_cbg_start, fusion_cbg_end,
            v_bm_map_p, v_bm_map_c, v_bm_map_p_c, v_cbg_start, v_cbg_end,
            a_bm_map_p, a_bm_map_c, a_bm_map_p_c, a_cbg_start, a_cbg_end,
            v_frame_cla, a_frame_cla, v_features, a_features, v_cbg_feature, a_cbg_feature,
        )

    def forward_back(self, video: Tensor, audio: Tensor) -> Sequence[Optional[Tensor]]:
        if self.cbg_fusion_start is not None:
            a_bm_in, _, _, v_bm_in, _, _ = self.forward_features(audio, video)

            # complementary boundary generator
            v_cbg_feature, v_cbg_start, v_cbg_end = self.forward_video_cbg(v_bm_in)
            a_cbg_feature, a_cbg_start, a_cbg_end = self.forward_audio_cbg(a_bm_in)

            # complementary boundary generator modal attention fusion
            fusion_cbg_start = self.cbg_fusion_start(v_bm_in, a_bm_in, v_cbg_start, a_cbg_start)
            fusion_cbg_end = self.cbg_fusion_end(v_bm_in, a_bm_in, v_cbg_end, a_cbg_end)

            return (
                fusion_cbg_start, fusion_cbg_end, v_cbg_start, v_cbg_end, a_cbg_start, a_cbg_end,
                v_cbg_feature, a_cbg_feature
            )
        else:
            return None, None, None, None, None, None, None, None

    def forward_features(self, audio, video):
        # encoders
        v_features = self.video_encoder(video)
        a_features = self.audio_encoder(audio)
        # frame classifiers
        v_frame_cla = self.video_frame_classifier(v_features)
        a_frame_cla = self.audio_frame_classifier(a_features)
        # concat classification result to features
        v_bm_in = torch.column_stack([v_features, v_frame_cla])
        a_bm_in = torch.column_stack([a_features, a_frame_cla])
        return a_bm_in, a_features, a_frame_cla, v_bm_in, v_features, v_frame_cla

    def forward_video_cbg(self, feature: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cbg_prob, cbg_feature = self.video_comp_boundary_generator(feature)
        start = cbg_prob[:, 0, :].squeeze(1)
        end = cbg_prob[:, 1, :].squeeze(1)
        return cbg_feature, end, start

    def forward_audio_cbg(self, feature: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cbg_prob, cbg_feature = self.audio_comp_boundary_generator(feature)
        start = cbg_prob[:, 0, :].squeeze(1)
        end = cbg_prob[:, 1, :].squeeze(1)
        return cbg_feature, end, start

    def loss_fn(self,
        fusion_bm_map_p: Tensor, fusion_bm_map_c: Tensor, fusion_bm_map_p_c: Tensor,
        fusion_cbg_start: Tensor, fusion_cbg_end: Tensor,
        fusion_cbg_start_back: Tensor, fusion_cbg_end_back: Tensor,
        v_bm_map_p: Tensor, v_bm_map_c: Tensor, v_bm_map_p_c: Tensor,
        v_cbg_start: Tensor, v_cbg_end: Tensor, v_cbg_feature: Tensor,
        v_cbg_start_back: Tensor, v_cbg_end_back: Tensor, v_cbg_feature_back: Tensor,
        a_bm_map_p: Tensor, a_bm_map_c: Tensor, a_bm_map_p_c: Tensor,
        a_cbg_start: Tensor, a_cbg_end: Tensor, a_cbg_feature: Tensor,
        a_cbg_start_back: Tensor, a_cbg_end_back: Tensor, a_cbg_feature_back: Tensor,
        v_frame_cla: Tensor, a_frame_cla: Tensor, n_frames: Tensor,
        fusion_bm_label: Tensor, fusion_start_label: Tensor, fusion_end_label: Tensor,
        v_bm_label, a_bm_label, v_start_label, a_start_label, v_end_label, a_end_label,
        v_frame_label, a_frame_label, contrast_label, v_features, a_features, a_v_features,
    ) -> Dict[str, Tensor]:
        (
            fusion_bm_loss, fusion_cbg_loss, fusion_prb_loss, fusion_cbg_loss_forward, fusion_cbg_loss_backward, _
        ) = self.bm_loss(
            fusion_bm_map_p, fusion_bm_map_c, fusion_bm_map_p_c,
            fusion_cbg_start, fusion_cbg_end, fusion_cbg_start_back, fusion_cbg_end_back,
            fusion_bm_label, fusion_start_label, fusion_end_label, n_frames
        )

        (
            v_bm_loss, v_cbg_loss, v_prb_loss, v_cbg_loss_forward, v_cbg_loss_backward, v_cbg_feature_loss
        ) = self.bm_loss(
            v_bm_map_p, v_bm_map_c, v_bm_map_p_c,
            v_cbg_start, v_cbg_end, v_cbg_start_back, v_cbg_end_back,
            v_bm_label, v_start_label, v_end_label, n_frames,
            v_cbg_feature, v_cbg_feature_back
        )

        (
            a_bm_loss, a_cbg_loss, a_prb_loss, a_cbg_loss_forward, a_cbg_loss_backward, a_cbg_feature_loss
        ) = self.bm_loss(
            a_bm_map_p, a_bm_map_c, a_bm_map_p_c,
            a_cbg_start, a_cbg_end, a_cbg_start_back, a_cbg_end_back,
            a_bm_label, a_start_label, a_end_label, n_frames,
            a_cbg_feature, a_cbg_feature_back
        )

        v_frame_loss = self.frame_loss(v_frame_cla.squeeze(1), v_frame_label, n_frames)
        a_frame_loss = self.frame_loss(a_frame_cla.squeeze(1), a_frame_label, n_frames)

        contrast_loss = torch.clip(self.contrast_loss(v_features, a_features, contrast_label, n_frames)
                                   / (self.cla_feature_in * self.temporal_dim), max=1.)

        a_v_features = self.fc(a_v_features)
        classify_loss = self.class_loss(a_v_features, 1-contrast_label)

        loss = fusion_bm_loss + \
               classify_loss + \
               self.weight_modal_bm_loss * (a_bm_loss + v_bm_loss) / 2 + \
               self.weight_frame_loss * (a_frame_loss + v_frame_loss) / 2 + \
               self.weight_contrastive_loss * contrast_loss

        loss_dict = {
            "loss": loss, "fusion_bm_loss": fusion_bm_loss, "v_bm_loss": v_bm_loss, "a_bm_loss": a_bm_loss,
            "v_frame_loss": v_frame_loss, "a_frame_loss": a_frame_loss, "contrast_loss": contrast_loss,
            "fusion_cbg_loss": fusion_cbg_loss, "v_cbg_loss": v_cbg_loss, "a_cbg_loss": a_cbg_loss,
            "fusion_prb_loss": fusion_prb_loss, "v_prb_loss": v_prb_loss, "a_prb_loss": a_prb_loss,
            "fusion_cbg_loss_forward": fusion_cbg_loss_forward, "v_cbg_loss_forward": v_cbg_loss_forward,
            "a_cbg_loss_forward": a_cbg_loss_forward, "fusion_cbg_loss_backward": fusion_cbg_loss_backward,
            "v_cbg_loss_backward": v_cbg_loss_backward, "a_cbg_loss_backward": a_cbg_loss_backward,
            "v_cbg_feature_loss": v_cbg_feature_loss, "a_cbg_feature_loss": a_cbg_feature_loss,
            "classify_loss": classify_loss,
        }
        # write_loss_to_file(loss)
        return {k: v for k, v in loss_dict.items() if v is not None}

    def forward(self, batch: Sequence[Tensor]) -> Dict[str, Tensor]:
        (
            video, audio, video_level_label
        ) = batch
        # forward
        (
            fusion_bm_map_p, fusion_bm_map_c, fusion_bm_map_p_c, fusion_cbg_start, fusion_cbg_end,
            v_bm_map_p, v_bm_map_c, v_bm_map_p_c, v_cbg_start, v_cbg_end,
            a_bm_map_p, a_bm_map_c, a_bm_map_p_c, a_cbg_start, a_cbg_end,
            v_frame_cla, a_frame_cla, v_features, a_features, v_cbg_feature, a_cbg_feature
        ) = self.forward_all(video, audio)

        video_attention_output, video_attn_weights = self.scaled_dot_product_attention(query=v_features, key=a_features, value=a_features)
        audio_attention_output, audio_attn_weights = self.scaled_dot_product_attention(query=a_features, key=v_features, value=v_features)
        output = torch.cat((video_attention_output, audio_attention_output), dim=2)
        output = self.fc_binary(output)
        return output

    def post_process_predict(self,
        bm_map_p: Tensor, bm_map_c: Tensor, bm_map_p_c: Tensor,
        cbg_start: Tensor, cbg_end: Tensor,
        cbg_start_back: Tensor, cbg_end_back: Tensor
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:

        bm_map = (bm_map_p + bm_map_c + bm_map_p_c) / 3
        if self.cbg_fusion_start is not None:
            start = torch.sqrt(cbg_start * torch.flip(cbg_end_back, dims=(1,)))
            end = torch.sqrt(cbg_end * torch.flip(cbg_start_back, dims=(1,)))
        else:
            start = None
            end = None

        return bm_map, start, end

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optimizer, gamma=0.992),
                "monitor": "val_loss"
            }
        }

    
    # for AVDF1M
    @classmethod
    def get_meta_attr(cls, meta: Metadata, video: Tensor, audio: Tensor, label: Tuple[Tensor, Tensor, Tensor]):
        fusion_bm_label, fusion_start_label, fusion_end_label = label

        a_bm_label, v_bm_label = cls.gen_audio_video_labels(fusion_bm_label, meta)
        a_start_label, v_start_label = cls.gen_audio_video_labels(fusion_start_label, meta)
        a_end_label, v_end_label = cls.gen_audio_video_labels(fusion_end_label, meta)

        frame_label_real = torch.zeros(512)
        frame_label_fake = torch.zeros(512)
        for begin, end in meta.fake_segments:
            begin = int(begin * 25)
            end = int(end * 25)
            frame_label_fake[begin: end] = 1

        if meta.modify_type == 'real':
            v_frame_label = frame_label_real
            a_frame_label = frame_label_real

            contrast_label = 1

        elif meta.modify_type == "visual_modified":
            v_frame_label = frame_label_fake
            a_frame_label = frame_label_real

            contrast_label = 0

        elif meta.modify_type == "audio_modified":
            v_frame_label = frame_label_real
            a_frame_label = frame_label_fake

            contrast_label = 0

        elif meta.modify_type == "both_modified":
            v_frame_label = frame_label_fake
            a_frame_label = frame_label_fake

            contrast_label = 0

        return [
            meta.video_frames, v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label,
            a_start_label, v_start_label, a_end_label, v_end_label
        ]

    @classmethod
    def gen_audio_video_labels(cls, label_fake: Tensor, meta: Metadata):
        label_real = torch.zeros(label_fake.size(), dtype=label_fake.dtype, device=label_fake.device)

        if meta['modify_type'] == 'real_audio_real_visual':
            v_label = label_real
            a_label = label_real

        elif meta['modify_type'] == "real_audio_fake_visual":
            v_label = label_fake
            a_label = label_real

        elif meta['modify_type'] == "fake_audio_real_visual":
            v_label = label_real
            a_label = label_fake

        elif meta['modify_type'] == "fake_audio_fake_visual":
            v_label = label_fake 
            a_label = label_fake 

        return a_label, v_label

