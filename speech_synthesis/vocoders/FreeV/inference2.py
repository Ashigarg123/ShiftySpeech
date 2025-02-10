from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from utils import AttrDict
from dataset import mel_spectrogram, load_wav
from models import Generator
import soundfile as sf
import librosa
import numpy as np
import time
h = None
device = None


def load_checkpoint(filepath, device):
    filepath = "path/to/APNet2/checkpoint" # replace this by actual checkpoint path 
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def infer(h):
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(h.checkpoint_file_load, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    l=0
    with torch.no_grad():
        starttime = time.time()
        inp_pth = h.wav_scp
        with open(inp_pth, 'r') as f:
            lines = f.readlines()
            for line in lines:
                utt_name, utt_pth = line.strip().split(" ")
                raw_wav, _ = librosa.load(utt_pth, sr=h.sampling_rate, mono=True)
                raw_wav = torch.FloatTensor(raw_wav).to(device)
                x = get_mel(raw_wav.unsqueeze(0))
                
                logamp_g, pha_g, _, _, y_g = generator(x)
                audio = y_g.squeeze()
                # logamp = logamp_g.squeeze()
                # pha = pha_g.squeeze()
                audio = audio.cpu().numpy()
                # logamp = logamp.cpu().numpy()
                # pha = pha.cpu().numpy()
                audiolen=len(audio)
                sf.write(os.path.join(h.test_output_dir, f"{utt_name}.wav"), audio, h.sampling_rate,'PCM_16')

                # print(pp)
                l+=audiolen
       

        end=time.time()
        print(end-starttime)
        print(l/24000)
        print(l/24000/(end-starttime))           

def main():
    print('Initializing Inference Process..')

    config_file = 'config.json'

    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    # inference(h)
    infer(h)


if __name__ == '__main__':
    main()

