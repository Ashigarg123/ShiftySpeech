from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from mel_data import MAX_WAV_VALUE, mel_spectrogram, load_wav
from models import Generator
from stft import TorchSTFT
import torchaudio
import librosa
from scipy.signal import resample
from librosa.filters import mel as librosa_mel_fn
h = None
device = None


def load_checkpoint(filepath, device):
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


def inference(a):
    generator = Generator(h).to(device)
    stft = TorchSTFT(filter_length=h.gen_istft_n_fft, hop_length=h.gen_istft_hop_size, win_length=h.gen_istft_n_fft).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    os.makedirs(a.output_dir, exist_ok=True)
    generator.eval()
    generator.remove_weight_norm()
    wav_scp = a.input_wavs_dir
    with torch.no_grad():
        for root, dirs, files in os.walk(wav_scp):
            for file in files:
                if file.endswith(".wav"):
                    utt_pth = os.path.join(root, file)
                    utt_name = os.path.basename(file).split(".wav")[0]
                    wav, sr = load_wav(utt_pth)
                    if sr !=24000:
                        target_sr = 24000
                        num_samples = int(len(wav) * target_sr / sr)
                        wav = resample(wav, num_samples)
                        sr = target_sr
                    wav = wav / MAX_WAV_VALUE
                    wav = torch.FloatTensor(wav).to(device)
                    x = get_mel(wav.unsqueeze(0))
                    spec, phase = generator(x)
                    stft.window = stft.window.to(spec.device)
                    y_g_hat = stft.inverse(spec, phase)
                    audio = y_g_hat.squeeze()
                    audio = audio * MAX_WAV_VALUE
                    audio = audio.cpu().numpy().astype('int16')
                    output_file = os.path.join(a.output_dir, utt_name + '_generated.wav')
                    write(output_file, h.sampling_rate, audio)
                    


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
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

    inference(a)


if __name__ == '__main__':
    main()

