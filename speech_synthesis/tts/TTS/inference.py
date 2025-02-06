from TTS.api import TTS
import os 
import random 
import pdb
import inspect 

# Glow-TTS 
tts = TTS("tts_models/en/ljspeech/glow-tts", gpu=True)

# Neural-HMM 
# tts = TTS("tts_models/en/ljspeech/neural_hmm", gpu=True)

# VITS 
# tts = TTS("tts_models/en/ljspeech/vits", gpu=True)

tr_f = []
with open('transcripts.txt', 'r') as transcript_file:
    lines = transcript_file.readlines()
    for line in lines:
        tr, fname = line.split("|")
        tr_f.append((tr, fname))
 
save_dir =  "./results" # change this to directory of your choice 
for i in range(len(tr_f)):
    txt, fname = tr_f[i][0], tr_f[i][1]
    save_pth = os.path.join(save_dir, f"{fname}.wav")
    tts.tts_to_file(text=txt,
                    file_path=save_pth,
                    # language="en",
                    split_sentences=False
                    )
   