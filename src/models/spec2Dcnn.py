from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from scipy.signal import find_peaks
import numpy as np

class Spec2DCNN(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        encoder_weights: Optional[str] = None,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
        unet_class: str = "Unet",
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
        model_sigmod: bool = False,
        stage_two=None,
        is_stage_two=False,
        post_process_cfg=None, 
        downsample_rate=None,
        output_sigmod=None,
        output_clip=None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.unet_class = getattr(smp, unet_class)
        self.encoder = self.unet_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.decoder = decoder
        self.stage_two = stage_two
        self.is_stage_two = is_stage_two
        if is_stage_two:
            assert post_process_cfg is not None and downsample_rate is not None and output_sigmod is not None and output_clip is not None
            self.score_th = post_process_cfg.score_th
            self.distance = post_process_cfg.distance
            self.downsample_rate = downsample_rate
            self.output_sigmod = output_sigmod
            self.output_clip = output_clip
                    
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = loss_fn
        if model_sigmod:
            self.sigmod = nn.Sigmoid()
        else:
            self.sigmod = None

    def get_stage_two_batch(self, logits):
        if self.output_sigmod:
            logits = logits.sigmod()
        if self.output_clip:
            logits = torch.clamp(logits, min=0,max=1)
        logits = logits.detach().cpu().numpy()
        batch = []
        for i in range(logits.shape[0]):
            clip_steps_pair = []
            for dim in range(2):
                peeks = find_peaks(logits[i, :, dim+1], height=0.02, threshold=self.score_th, distance=self.distance)[0]
                if len(peeks) ==0:
                    peeks = [1]
                for peeks_idx in peeks:
                    peeks_idx = peeks_idx * self.downsample_rate
                    
                    if peeks_idx < 1440:
                        clip_start = 0
                        clip_end = 2880
                    elif peeks_idx > 14400:
                        clip_start = 14400
                        clip_end = 17280
                    else:
                        clip_start = peeks_idx - 1440
                        clip_end = peeks_idx + 1440
                    clip_steps_pair.append((clip_start, clip_end))
            batch.append(list(set(clip_steps_pair)))
        return batch

                
                
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
        logits: Optional[torch.Tensor] = None,
        raw_label: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        if self.is_stage_two:
            loss = None
            assert logits is not None
            batch_steps = self.get_stage_two_batch(logits)
            batch = []
            batch_label = []
            # print(x.shape)
            # print(len(batch_steps))
            batch_steps_one_list = []
            for i, b in enumerate(batch_steps):
                for steps_pair in b:
                    batch_steps_one_list.append((i, steps_pair))
                    batch.append(x[i, :, steps_pair[0]:steps_pair[1]])
                    if labels is not None:
                        batch_label.append(raw_label[i, steps_pair[0]:steps_pair[1], :])
            # random select 64 on a batch
            random_select_index = list(np.random.choice(len(batch), 64)) if len(batch) > 64 else list(range(len(batch)))
            ret = torch.zeros((len(batch_steps), 17280, 3))
            x = torch.stack([batch[i] for i in random_select_index])
            if raw_label is not None:
                y = torch.stack([batch_label[i] for i in random_select_index])
            x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)
            x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
            logits = self.decoder(x)  # (batch_size, n_timesteps, n_classes)
            if raw_label is not None:
                loss = self.loss_fn(logits, y)
            # TODO overwrite
            selected_batch_steps = [batch_steps_one_list[i] for i in random_select_index]
            for i, (b_idx, steps_pair) in enumerate(selected_batch_steps):
                # print(ret.shape, steps_pair, logits.shape)
                ret[b_idx, steps_pair[0]:steps_pair[1], :] = logits[i, :, :]
            return ret, loss
        raw_x = x.clone()
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
        logits = self.decoder(x)  # (batch_size, n_timesteps, n_classes)
        
        if self.sigmod is not None:
            logits = self.sigmod(logits)
        output = {}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            output["loss"] = loss
        if self.stage_two is not None:
            logits, stage_two_loss = self.stage_two(raw_x, labels, logits=logits, raw_label=raw_label)
            if labels is not None:
                output["loss"] = output["loss"]+stage_two_loss
        output["logits"] = logits

        return output
