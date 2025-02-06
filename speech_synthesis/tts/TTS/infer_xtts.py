from TTS.api import TTS
import os 
import random 
import pdb
import inspect 
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
spk_f = "./xtts_spks.txt"
tr_f = []
# get all speakers 
with open(spk_f, 'r') as speaker_file:
    speakers = [line.strip() for line in speaker_file if line.strip()]

with open('transcripts.txt', 'r') as transcript_file:
    lines = transcript_file.readlines()
    for line in lines:
        tr, fname = line.split("|")
        tr_f.append((tr, fname))


save_dir = "./results" # change this to directory of your choice 
for i in range(len(tr_f)):
    txt, fname = tr_f[i][0], tr_f[i][1]
    speaker = random.choice(speakers)
    save_pth = os.path.join(save_dir, f"{fname}.wav")
    tts.tts_to_file(text=txt,
                    file_path=save_pth,
                    speaker= speaker,
                    language="en",
                    split_sentences=False
                    )
   