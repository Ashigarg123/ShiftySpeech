import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
from omegaconf import OmegaConf
from model.generator import Generator
import pdb
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import librosa
import torchaudio
# helper functions derived from -- https://github.com/jik876/hifi-gan/blob/master/meldataset.py 
# also -- https://github.com/maum-ai/univnet/tree/9bb2b54838bb6d7ce767131cc7b8b61198bc7558 original

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_wav(full_path):
    # wav, sr = librosa.load(full_path, sr=24000)
    wav, sr = torchaudio.load(full_path)
    print(f"Min value: {wav.min()}, Max value: {wav.max()}")
    target_sr = 24000
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    new_wav = resampler(wav)
    # clamp 
    new_wav = torch.clamp(new_wav, min=-1, max=1)
    new_wav = new_wav.numpy()
   
    return new_wav, target_sr

# generate mel spectrogram on the fly 

def mel_spectrogram(y):
    """Computes mel-spectrograms from a batch of waves
    PARAMS
    ------
    y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

    RETURNS
    -------
    mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
    """
    print(y.shape)
    y = torch.from_numpy(y).unsqueeze(0).to(device)
    assert(torch.min(y) >= -1)
    assert(torch.max(y) <= 1)
    n_mel_channels = 100
    n_fft=  1024
    hop_size =  256 # WARNING: this can't be changed.
    win_size = 1024
    sampling_rate = 24000
    fmin = 0.0
    fmax = 12000.0
    center = False
    mel = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, fmin, fmax)
    
    mel_basis = torch.from_numpy(mel).float().to(device)
    
    hann_window = torch.hann_window(win_size).to(device)
    y = torch.nn.functional.pad(y,
                                (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                 mode='reflect')
   

    y = y.squeeze(1).to(device) 
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                        center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)
    
    return spec

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = OmegaConf.load(args.config)
    else:
        print(checkpoint['hp_str'])
        hp = OmegaConf.create(checkpoint['hp_str'])

    model = Generator(hp).cuda()
    saved_state_dict = checkpoint['model_g']
    new_state_dict = {}
    
    for k, v in saved_state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict['module.' + k]
        except:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval(inference=True)
    
    with torch.no_grad():
        wav_scp = args.wav_scp
        for root, dirs, files in os.walk(wav_scp):
            for file in files:
                if file.endswith(".wav"):
                    utt_pth = os.path.join(root, file)
                    utt_name = os.path.basename(file).split(".wav")[0]
     
                    audio, sr = load_wav(utt_pth)
                    # generate mel spectrogram 
                    mel = mel_spectrogram(audio)
                    mel = mel.squeeze(0)

            
                    if len(mel.shape) == 2:
                        mel = mel.unsqueeze(0)
                    
                    mel = mel.cuda()
                
                    audio = model.inference(mel)
                    audio = audio.cpu().detach().numpy()
                    out_path = os.path.join(args.output_folder, f"{utt_name}.wav")

                    write(out_path, hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str,
                        help="directory of mel-spectrograms to invert into raw audio.")
    parser.add_argument('-o', '--output_folder', type=str, default='.',
                        help="directory which generated raw audio is saved.")
    parser.add_argument('--wav_scp', '--wav_scp', type=str,required=True, default=None,
                        help="file containing wav paths.")
    args = parser.parse_args()

    main(args)
