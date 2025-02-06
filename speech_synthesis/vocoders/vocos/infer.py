import torch
from vocos import Vocos
import torchaudio
import os 

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
print('model loaded!')
wav_scp = "/path/to/eval_file"
save_pth = "/path/to/output_dir"
with open(wav_scp, 'r') as f:
    lines = f.readlines()
    for line in lines:
        uttname, utt_pth = line.strip().split()
        y, sr = torchaudio.load(utt_pth)
        if y.size(0) > 1:  # mix to mono
            y = y.mean(dim=0, keepdim=True)
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
        y_hat = vocos(y)
        # save audio 
        torchaudio.save(os.path.join(save_pth, f"{uttname}.wav"), y_hat, 24000)