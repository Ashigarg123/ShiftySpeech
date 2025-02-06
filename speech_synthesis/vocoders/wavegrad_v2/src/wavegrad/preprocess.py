# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT
import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from params import params


def transform(data):
  utt_name, filename = data
  save_dir = "path/to/save/mel-spectrograms"
  audio, sr = T.load(filename)
  if params.sample_rate != sr:
    print(f'Invalid sample rate {sr}.')
    resampler = TT.Resample(sr, params.sample_rate)
    audio = resampler(audio)
    sr = params.sample_rate
  audio = torch.clamp(audio[0], -1.0, 1.0)

  hop = params.hop_samples
  win = hop * 4
  n_fft = 2**((win-1).bit_length())
  f_max = sr / 2.0
  mel_spec_transform = TT.MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win, hop_length=hop, f_min=20.0, f_max=f_max, power=1.0, normalized=True)

  with torch.no_grad():
    spectrogram = mel_spec_transform(audio)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    np.save(f'{save_dir}/{utt_name}.spec.npy', spectrogram.cpu().numpy())


def main(args):
  with open(args.dir, 'r') as f:
    data = [(line.strip().split()[0], line.strip().split()[1]) for line in f if line.strip()]
  with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(transform, data), desc='Preprocessing', total=len(data)))


if __name__ == '__main__':
  parser = ArgumentParser(description='prepares a dataset to train WaveGrad')
  parser.add_argument('--dir',
      help='directory containing .wav files for training')
  parser.add_argument('--save_dir', help='directory to save npy')
  main(parser.parse_args())
