from typing import Union

import torch.nn as nn
from omegaconf import DictConfig

from src.models.decoder.informer import Informer
from src.models.decoder.itransformer import ITransformer
from src.models.decoder.lstmdecoder import LSTMDecoder
from src.models.decoder.enlstm import enLSTMDecoder
from src.models.decoder.mlpdecoder import MLPDecoder
from src.models.decoder.transformerdecoder import TransformerDecoder
from src.models.decoder.unet1ddecoder import UNet1DDecoder
from src.models.decoder.reslstm import ResLSTMDecoder
from src.models.feature_extractor.cnn import CNNSpectrogram
from src.models.feature_extractor.lstm import LSTMFeatureExtractor
from src.models.feature_extractor.panns import PANNsFeatureExtractor
from src.models.feature_extractor.spectrogram import SpecFeatureExtractor
from src.models.feature_extractor.transformer import TransformerFeatureExtractor
from src.models.feature_extractor.mix import MixedFeatureExtractor
from src.models.loss.focal import FocalLoss
from src.models.loss.mix import CombinedLoss
from src.models.spec1D import Spec1D
from src.models.spec2Dcnn import Spec2DCNN

FEATURE_EXTRACTORS = Union[
    CNNSpectrogram, PANNsFeatureExtractor, LSTMFeatureExtractor, SpecFeatureExtractor
]
DECODERS = Union[UNet1DDecoder, LSTMDecoder, TransformerDecoder, MLPDecoder]
MODELS = Union[Spec1D, Spec2DCNN]


def get_feature_extractor(
    cfg: DictConfig, feature_dim: int, num_timesteps: int
) -> FEATURE_EXTRACTORS:
    feature_extractor: FEATURE_EXTRACTORS
    if cfg.name == "CNNSpectrogram":
        feature_extractor = CNNSpectrogram(
            in_channels=feature_dim,
            base_filters=cfg.base_filters,
            kernel_sizes=cfg.kernel_sizes,
            stride=cfg.stride,
            sigmoid=cfg.sigmoid,
            output_size=num_timesteps,
            conv=nn.Conv1d,
            reinit=cfg.reinit,
        )
    elif cfg.name == "PANNsFeatureExtractor":
        feature_extractor = PANNsFeatureExtractor(
            in_channels=feature_dim,
            base_filters=cfg.base_filters,
            kernel_sizes=cfg.kernel_sizes,
            stride=cfg.stride,
            sigmoid=cfg.sigmoid,
            output_size=num_timesteps,
            conv=nn.Conv1d,
            reinit=cfg.reinit,
            win_length=cfg.win_length,
        )
    elif cfg.name == "LSTMFeatureExtractor":
        feature_extractor = LSTMFeatureExtractor(
            in_channels=feature_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            bidirectional=cfg.bidirectional,
            out_size=num_timesteps,
        )
    elif cfg.name == "SpecFeatureExtractor":
        feature_extractor = SpecFeatureExtractor(
            in_channels=feature_dim,
            height=cfg.height,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            out_size=num_timesteps,
        )
    elif cfg.name == "TransformerExtractor":
        feature_extractor = TransformerFeatureExtractor(
            in_channels=feature_dim,
            height=cfg.height,
            num_layers=cfg.num_layers,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            out_size=num_timesteps,
        )
    elif cfg.name == "MixedFeatureExtractor":
        feature_extractor = MixedFeatureExtractor(extractors=[get_feature_extractor(extractor, feature_dim, num_timesteps) for extractor in cfg.extractors], transformer_weight=cfg.transformer_weight, attention=cfg.attention)
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg.name}")

    return feature_extractor


def get_decoder(cfg: DictConfig, n_channels: int, n_classes: int, num_timesteps: int) -> DECODERS:
    decoder: DECODERS
    if cfg.decoder.name == "UNet1DDecoder":
        decoder = UNet1DDecoder(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_timesteps,
            bilinear=cfg.decoder.bilinear,
            se=cfg.decoder.se,
            res=cfg.decoder.res,
            scale_factor=cfg.decoder.scale_factor,
            dropout=cfg.decoder.dropout,
        )
    elif cfg.decoder.name == "LSTMDecoder":
        decoder = LSTMDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            bidirectional=cfg.decoder.bidirectional,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "enLSTMDecoder":
        decoder = enLSTMDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            bidirectional=cfg.decoder.bidirectional,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "ResLSTMDecoder":
        decoder = ResLSTMDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            bidirectional=cfg.decoder.bidirectional,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "TransformerDecoder":
        decoder = TransformerDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            nhead=cfg.decoder.nhead,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "MLPDecoder":
        decoder = MLPDecoder(n_channels=n_channels, n_classes=n_classes)
    elif cfg.decoder.name == "Informer":
        decoder = Informer(
            enc_in=n_channels, 
            dec_in=n_channels, 
            c_out=n_classes, 
            factor=cfg.decoder.factor, 
            d_model=cfg.decoder.d_model,
            n_heads=cfg.decoder.n_heads,
            e_layers=cfg.decoder.e_layers,
            d_layers=cfg.decoder.d_layers,
            d_ff=cfg.decoder.d_ff,
            max_len=cfg.duration,
            dropout=cfg.decoder.dropout,
            attn=cfg.decoder.attn,
            attn_layer=cfg.decoder.attn_layer, 
            embed=cfg.decoder.embed,
            freq=cfg.decoder.freq,
            activation=cfg.decoder.activation,
            distil=cfg.decoder.distil,
            mix=cfg.decoder.mix,
        )
    elif cfg.decoder.name == "ITransformer":
        decoder = ITransformer(
            c_in=cfg.feature_extractor.base_filters,
            c_out=n_classes,
            seq_len=cfg.duration//cfg.downsample_rate, 
            pred_len=cfg.duration//cfg.downsample_rate, 
            e_layers=cfg.decoder.e_layers, 
            d_model=cfg.decoder.d_model, 
            d_ff=cfg.decoder.d_ff, 
            factor=cfg.decoder.factor, 
            n_heads=cfg.decoder.n_heads, 
            activation=cfg.decoder.activation, 
            dropout=0.1, 
        )
    else:
        raise ValueError(f"Invalid decoder name: {cfg.decoder.name}")

    return decoder

def get_loss_fn(loss_cfg: DictConfig, sigmod: bool) -> nn.Module:
    if loss_cfg.name == "BCE":
        if sigmod:
            loss_fn = nn.BCELoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss()
    elif loss_cfg.name == "MSE":
        loss_fn = nn.MSELoss()
    elif loss_cfg.name == "NLL":
        loss_fn = nn.NLLLoss()
    elif loss_cfg.name == "Focal":
        loss_fn = FocalLoss(alpha=loss_cfg.alpha, gamma=loss_cfg.gamma)
    elif loss_cfg.name == "MIX":
        loss_fn = CombinedLoss(loss1=get_loss_fn(loss_cfg.loss1), loss2=get_loss_fn(loss_cfg.loss2))
    else:
        raise ValueError(f"Invalid loss name: {loss_cfg.name}")
    return loss_fn

def get_model(cfg: DictConfig, feature_dim: int, n_classes: int, num_timesteps: int, sigmod:bool) -> MODELS:
    model: MODELS
    if cfg.model.name == "Spec2DCNN":
        feature_extractor = get_feature_extractor(cfg.feature_extractor, feature_dim, num_timesteps)
        decoder = get_decoder(cfg, feature_extractor.height, n_classes, num_timesteps)
        loss_fn = get_loss_fn(cfg.loss, sigmod)
        model = Spec2DCNN(
            feature_extractor=feature_extractor,
            decoder=decoder,
            encoder_name=cfg.model.encoder_name,
            in_channels=feature_extractor.out_chans,
            encoder_weights=cfg.model.encoder_weights,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
            unet_class=cfg.model.unet_class,
            loss_fn=loss_fn,
            sigmod=sigmod,
        )
    elif cfg.model.name == "Spec1D":
        feature_extractor = get_feature_extractor(cfg, feature_dim, num_timesteps)
        decoder = get_decoder(cfg, feature_extractor.height, n_classes, num_timesteps)
        model = Spec1D(
            feature_extractor=feature_extractor,
            decoder=decoder,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
        )
    else:
        raise NotImplementedError

    return model
