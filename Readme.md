<p align="center">
  <a href="https://github.com/Ashigarg123/ShiftySpeech">
    <img src="https://github.com/Ashigarg123/ShiftySpeech/blob/main/Images/shiftyspeech_logo.png" alt="ShiftySpeech Logo" width="500" height="auto">
  </a>
</p>

# 🌀 *ShiftySpeech*: A Large-Scale Synthetic Speech Dataset with Distribution Shifts
This is the official repository of *ShiftySpeech* – a diverse and extensive dataset containing **3000+ hours** of synthetic speech generated using various **TTS systems** and **vocoders**, while covering multiple **distribution shifts**. 

## 🔥 Key Features
- 3000+ hours of synthetic speech
- **Diverse Distribution Shifts**: The dataset spans **7 key distribution shifts**, including:  
  - 📖 **Reading Style**  
  - 🎙️ **Podcast**  
  - 🎥 **YouTube**  
  - 🗣️ **Languages (Three different languages)**  
  - 🌎 **Demographics (including variations in age, accent, and gender)**  
- **Multiple Speech Generation Systems**: Includes data synthesized from various **TTS models** and **vocoders**.

## 💡 Why We Built This Dataset
> Driven by advances in self-supervised learning for speech, state-of-the-art synthetic speech detectors have achieved low error rates on popular benchmarks such as ASVspoof. However, prior benchmarks do not address the wide range of real-world variability in speech. Are reported error rates realistic in real-world conditions? To assess detector failure modes and robustness under controlled distribution shifts, we introduce **ShiftySpeech**, a benchmark with more than 3000 hours of synthetic speech from 7 domains, 6 TTS systems, 12 vocoders, and 3 languages.

## Downloading the dataset 

### ShiftySpeech
Dataset can be downloaded from [HuggingFace](https://huggingface.co/datasets/ash56/ShiftySpeech/tree/main)

##### Dataset Structure
The dataset is structured as follows:
```plaintext
/ShiftySpeech
    ├── Vocoders/ 
    │  ├── vocoder-1/
    │  │   ├── vocoder-1_aishell_flac.tar.gz
    │  │   ├── vocoder-1_jsut_flac.tar.gz
    │  │   ├── vocoder-1_youtube_flac.tar.gz
    │  │   ├──vocoder-1_audiobook_flac.tar.gz
    │  │   ├── vocoder-1_podcast_flac.tar.gz
    │  │   ├── vocoder-1_voxceleb_test_flac.tar.gz
    │  │   ├── vocoder-1_commonvoice_flac.tar.gz
    │  │   ├── vocoder-1_ljspeech_flac.tar.gz
    │  │   ├── vocoder-1_train-clean-360_flac.tar.gz
    │  │   
    │  ├── vocoder-2/
    │       ├── ...
    │      
    │     
    │ 
    ├── TTS/  # Contains multiple TTS-based generated speech
    │   ├── TTS_Grad-TTS.tar.gz
    │   ├── TTS_Glow-TTS.tar.gz  
    │   ├── TTS_FastPitch.tar.gz   
    │   ├── TTS_VITS.tar.gz
    │   ├── TTS_XTTS.tar.gz   
    │   ├── TTS_YourTTS.tar.gz   
    │   ├── TTS_hfg_vocoded.tar.gz #Note: ``TTS_hfg_vocoded.tar.gz`` contains synthetic speech generated using HiFiGAN vocoder trained on LJSpeech
    │

```
The source datasets covered by different TTS and Vocoder systems are listed in [tts.yaml](https://github.com/Ashigarg123/ShiftySpeech/tree/main/dataset/tts.yaml) and [vocoders.yaml](https://github.com/Ashigarg123/ShiftySpeech/tree/main/dataset/vocoders.yaml)

Source dataset for above synthetic speech can be download using the following links: 
- [JSUT Basic5000](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
- [AISHELL-1](https://www.openslr.org/33/)
- [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
- [Audiobook](https://www.openslr.org/12)
- [MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
- [CommonVoice 19.0](https://commonvoice.mozilla.org/tw/datasets)


Wav list corresponding to the source datasets can be found here : 
```sh 
   dataset/wav_lists
```

We utilize WaveFake dataset for training. It can be downloaded from [here](https://zenodo.org/records/5642694)
Train, dev and test split used for WaveFake can be found in ``dataset/train_wav_list``
## Downloading the pre-trained models 

Pre-trained models can be downloaded from [here](https://huggingface.co/ash56/ShiftySpeech/tree/main/detection-systems) 

Individual models based on the below folder structure
Example, to download ``hifigan.pt``:
```bash 
wget https://huggingface.co/ash56/ShiftySpeech/resolve/main/detection-systems/single_vocoder/hifigan.pth
```
Folder structure:
```plaintext
detection-system/
│
├──single_vocoder/         # Selecting synthetic speech to train on       
│   ├──hifigan.pth         # Detectors trained on synthetic speech from a single vocoder  
│   ├──melgan.pth          # Synthetic speech derived from MelGAN vocoder
│   ├── ...                
│   
├──leave_one_out/          # Detectors trained on synthetic speech from multiple vocoders       
│   ├──leave_hifigan.pth   # Synthetic speech derived from vocoders other than HiFiGAN      
│   ├──leave_melgan.pth         
│   ├── ...
│   
├──num_spks/               # Detectors trained on synthetic speech from ``n`` number of speakers       
│   ├──exp-1/              # Round one of selecting different set of speakers to train on
|   |   ├──spk1.pt         # Synthetic speech derived from single speaker
│   │   ├──spks4.pt        # Synthetic speech derived from four speakers
│   │                           
│   ├──... 
│   │   
│   │     
│   ├──exp-5/               
|   |  ├──spk1.pt/
│   │  ├──spks4.pt/
|
├──augmentations/          # Detectors trained on synthetic speech from HiFiGAN vocoder 
|      ├──hfg_aug_1_2.pt   # Augmentations applied during training:
│                            (1) Linear and non-linear convolutive noise                                    
|                            (2) Impulsive signal-dependent additive noise
│                          
```

## Reproducing Experiments 
To reproduce the experiments, please follow the following instructions:

---
**Environment Setup**
```sh
conda create -n SSL python=3.10.14
```
```sh
conda activate SSL 
```

```sh
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Download and install fairseq from [here](https://github.com/facebookresearch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)

```sh
pip install -r requirements.txt
```
Download xlsr model from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr) and add in /synthetic_speech_detection/SSL_Anti-spoofing/model.py


**Prepare the data**

- Create evaluation file of the format :
```sh 
   <real_audio_path> bonafide 
   <synthetic_audio_path> spoof
```
- `bonafide` represents genuine audio.
- `spoof` represents synthetic or fake audio.

The helper script for creating the evaluation file is ``prepare_test.py``. See below for an example usage:
```sh
dataset/prepare_test.py --bona_dir dataset/real-data/jsut --spoof_dir dataset/Shiftyspeech/Vocoders/bigvgan/jsut_flac --save_path dataset/test_files/bigvgan_jsut_test.txt
```
🚀**Evaluating In-the-Wild**
- Download In-the-Wild dataset from [here](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa) and create the evaluation file in the above format
- Evaluate the model trained on HiFiGAN generated utterances with and without augmentations
Download pre-trained models:
```bash 
wget https://huggingface.co/ash56/ShiftySpeech/resolve/main/detection-systems/single_vocoder/hifigan.pth
```

```bash 
wget https://huggingface.co/ash56/ShiftySpeech/resolve/main/detection-systems/augmentations/hfg_aug_1_2.pt
```

Evaluate using SSL-AASIST model:
```sh
python train.py --eval \
    --test_score_dir <path_to_save_scores> \
    --model_name SSL-AASIST \
    --test_list_path <path_to_test_file_with_utterances_and_labels> \
    --model_path <path_to_downloaded_model>
```
You can also train the models on WaveFake dataset using the following commands:
```sh
python train.py --trn_list_path <path_to_train_file> \ 
                --dev_list_path <path_to_dev_file> \
                --save_path <path_to_save_checkpoint> 
```
For the following experiments, create the evaluation files for all distribution shifts as mentioned in ``Prepare the data`` section 

**Impact of Synthetic Speech Selection** 

Here, we study the impact of training on single vocoder vs multiple-vocoder systems. In addition, we analyze the effect of training on one vocoder vs the other. 
- Load the pre-trained model saved in ``models/pre-trained``

Example evaluation:
```bash
python train.py --eval \
    --test_score_dir dataset/test_scores \
    --model_name SSL-AASIST \
    --test_list_path dataset/test_files/bigvgan_jsut.txt \
    --model_path models/pre-trained/hifigan.pt>
```
🗣️**Training on more speakers** 

We analyze the impact of training a detector on single speaker vs training on multiple speakers. Number of speakers in training vary from 1 to 10. 
We release pre-trained models for models trained on single-speaker and four speakers. Training data used is LibriTTS. Five different models are trained for both single and multi-speaker experiment. Speakers are randomly selected. 

You can download the pre-trained models as follows:

```sh 
wget https://huggingface.co/ash56/ShiftySpeech/resolve/main/detection-systems/num_spks/exp-1/spk1.pt
```
Similarly pre-trained model for experiment 2 and single speaker model can be downloaded from -- 
```sh 
wget https://huggingface.co/ash56/ShiftySpeech/resolve/main/detection-systems/num_spks/exp-2/spk1.pt
```
Example evaluation:
```bash
python train.py --eval \
    --test_score_dir dataset/test_scores \
    --model_name SSL-AASIST \
    --test_list_path dataset/test_files/bigvgan_jsut.txt \
    --model_path models/pre-trained/spk1.pt>
```

**Newly released vocoder** 

Now, we include new vocoders in training in chronological order of release. 
For vocoder systems not included in WaveFake dataset, we release the generated samples for training and can be downloaded from folder -- 
```bash

wget https://huggingface.co/datasets/ash56/ShiftySpeech/tree/main/Vocoders/<vocoder_of_choice>/ljspeech_flac.tar.gz

```
Trained models can then be evaluated on distribution shifts similar to above experiments

**Training on Vocoded speech vs training on TTS speech**

In previous experiments we train detection systems by using vocoders as the only source of synthetic speech generation. Here, we train the detection system with synthetic speech generated using end-to-end TTS systems. For this experiment, we train detectors using following systems as the source of generation:
- Grad-TTS 
- VITS
- Glow-TTS
- HFG Vocoded 
- Combination of above

Note: Vocoder utilized during synthesis for TTS systems is HiFiGAN (HFG). All the systems utilized are trained on LJSpeech.
Above TTS systems can be downloaded as part of coqui-ai repository. Please follow coqui-ai [Readme](https://github.com/Ashigarg123/ShiftySpeech/tree/main/speech_synthesis/tts/TTS#readme) for more instructions. Below we list the models used for synthesis:

```sh
tts = TTS("tts_models/en/ljspeech/glow-tts", gpu=True,  vocoder_path="vocoder_models/en/ljspeech/hifigan_v2")
```
```sh
tts = TTS("tts_models/en/ljspeech/vits", gpu=True, vocoder_path="vocoder_models/en/ljspeech/hifigan_v2")
```
Grad-TTS can be used to synthesize speech using the following command:
```sh
python inference.py -f <path_to_transcripts> -c speech_synthesis/tts/Grad-TTS/checkpts/grad-tts.pt
```
Example format for transcripts:
```sh
Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition|LJ001-0001.wav
```
HiFiGAN model utilized is derived from [Hugging Face](https://huggingface.co/speechbrain/tts-hifigan-ljspeech)

Train, test and dev splits for LJSpeech are same as defined above for previous experiments. Trained models can then be evaluated on synthesized test set of above TTS and vocoder systems

**Language distribution shift**

Here, we train detection systems on English corpus and evaluate on Chinese corpus and vice-versa. We utilize VCTK as the source corpus for English and AISHELL-1 corpus for Chinese. Multi-lingual XTTS system is used for generation. It can be loaded from coqui-ai as:
```sh
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
```

Wav list for training can be found in : ``dataset/train_wav_list``
Trained detectors can then be evaluated on test set of AISHELL-1 corpus and JSUT basic5000 corpus generated using XTTS

Please note that for a given sample audio, source speaker from original audio is used as reference, example: 
```sh
  tts.tts_to_file(text=txt,
                    file_path=save_pth,
                    speaker= source_speaker,
                    language="ja",
                    split_sentences=False
                    )
```

## **Synthetic speech detection system**
This repository utilizes [SSL-AASIST](https://arxiv.org/abs/2202.12233) model for spoof speech detection. Implementation is derived from [official](https://github.com/TakHemlata/SSL_Anti-spoofing) GitHub repository. 

## **Speech synthesis systems** 
This section lists TTS and vocoder models used in this repository
#### **Text-to-Speech (TTS) Systems**
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/README.md)
- [Coqui-ai](https://github.com/coqui-ai/TTS/tree/dev)

#### **Vocoder Systems**
- [ParalleWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/tree/master)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [BigVSAN](https://github.com/sony/bigvsan)
- [UnivNet](https://github.com/maum-ai/univnet/tree/master)
- [Vocos](https://github.com/gemelo-ai/vocos)
- [WaveGrad](https://github.com/lmnt-com/wavegrad)
- [iSTFTNET](https://github.com/rishikksh20/iSTFTNet-pytorch/tree/master)
- [APNet2](https://github.com/BakerBunker/FreeV/tree/main)

### **FAQ ❓** 

Please feel free to reach out if you have any questions or comments about the resource using GitHub issues or contacting us via email at [agarg22@jhu.edu](mailto:agarg22@jhu.edu) or [noa@jhu.edu](mailto:noa@jhu.edu).

### **Citation**

If you find this dataset or repository useful, please cite our work:
```bibtex
@misc{garg2025syntheticspeechdetectionwild,
      title={Less is More for Synthetic Speech Detection in the Wild}, 
      author={Ashi Garg and Zexin Cai and Henry Li Xinyuan and Leibny Paola García-Perera and Kevin Duh and Sanjeev Khudanpur and Matthew Wiesner and Nicholas Andrews},
      year={2025},
      eprint={2502.05674},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2502.05674}, 
}
```
## Acknowledgment
This work was supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the ARTS Program under contract D2023-2308110001. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

We thank Rafael Rivera-Soto, Rachel Wicks and Andrew Wang for their valuable discussions and feedback, which contributed to this work.


