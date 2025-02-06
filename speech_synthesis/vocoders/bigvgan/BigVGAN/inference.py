# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import json
import torch
import librosa
from utils import load_checkpoint
from meldataset import get_mel_spectrogram
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from bigvgan import BigVGAN as Generator

h = None
device = None
torch.backends.cudnn.benchmark = False


def inference(a, h):
    f = a.input_wavs_file
    generator = Generator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])
    os.makedirs(a.output_dir, exist_ok=True)
    generator.eval()
    generator.remove_weight_norm()

    with torch.no_grad():
        with open(f, 'r') as wav_file:
            lines = wav_file.readlines()
            for line in lines:
                utt_id, wav_pth = line.strip().split(" ")
                wav, sr = librosa.load(wav_pth, sr=h.sampling_rate, mono=True
            ) 
                wav = torch.FloatTensor(wav).to(device)

                x = get_mel_spectrogram(wav.unsqueeze(0), generator.h)

                y_g_hat = generator(x)

                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype("int16")

                output_file = os.path.join(
                    a.output_dir, f"{utt_id}" + os.path.basename(wav_pth).split(".wav")[0] + "_generated.wav"
                )
                write(output_file, h.sampling_rate, audio)

        


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wavs_dir", default="test_files")
    parser.add_argument("--input_wavs_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_file",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False)

    a = parser.parse_args()

    config_file = args.config
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(a, h)


if __name__ == "__main__":
    main()
