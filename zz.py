import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
import numpy as np
from VocGAN.model.generator import ModifiedGenerator
from VocGAN.utils.hparams import HParam, load_hparam_str
from VocGAN.denoiser import Denoiser

MAX_WAV_VALUE = 32768.0


def hi(checkpoint_path,input,d=True):
    checkpoint = torch.load(checkpoint_path)
    hp = load_hparam_str(checkpoint['hp_str'])

    model = ModifiedGenerator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=True)

    with torch.no_grad():
        mel = torch.from_numpy(input)
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()
        audio = model.inference(mel)

        audio = audio.squeeze(0)  # collapse all dimension except time axis
        if d:
            denoiser = Denoiser(model).cuda()
            audio = denoiser(audio, 0.01)
        audio = audio.squeeze()
        audio = audio[:-(hp.audio.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        audio = audio.cpu().detach().numpy()

        out_path = "hi.wav"
        write(out_path, hp.audio.sampling_rate, audio)
        return
