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

## Downloading the dataset 

### ShiftySpeech
To download the dataset, you need to install `gsutil`. Follow the official [Installation Guide](https://cloud.google.com/storage/docs/gsutil_install#linux) for setup instructions.

```bash 
gsutil cp -r https://storage.cloud.google.com/ssd_in_the_wild/ShiftySpeech/flac_version/ShiftySpeech dataset/ShiftySpeech
```
##### Dataset Structure
The dataset is structured as follows:
```plaintext
/ShiftySpeech
    ├── Vocoders/ 
        ├── vocoder-1/
        │   ├── aishell/
        │   ├── jsut/
            ├── youtube/
            ├── audiobook/
            ├── podcast/
            ├── voxceleb/
        ├── vocoder-2/
        │   ├── aishell/
        │   ├── jsut/
            ├── youtube/
            ├── audiobook/
            ├── podcast/
            ├── voxceleb/
        ├── ... 
    ├── TTS/  # Contains multiple TTS-based generated speech
    │   ├── Grad-TTS/
    │   │   ├── LJSpeech/
    │   │   
    │   ├── glow-tts/
    │   │   ├── ljspeech/
    │   │   
    │   ├── ... 
    │

```
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
We also provide the corresponding real data for convenience. It can be downloaded using the following command:

```bash 
gsutil cp -r https://storage.cloud.google.com/ssd_in_the_wild/real-data dataset/
```
We utilize WaveFake dataset for training. It can be downloaded from [here](https://zenodo.org/records/5642694)
Train, dev and test split used for WaveFake can be found in ``dataset/train_wav_list``
## Downloading the pre-trained models 
Pre-trained models can be downloaded as follows:
```bash 
gsutil cp -r https://storage.cloud.google.com/ssd_in_the_wild/pre-trained-models/ models/pre-trained/
```
You can also download individual models based on the below folder structure
Example, to download ``hifigan.pt``:
```bash 
gsutil cp https://storage.cloud.google.com/ssd_in_the_wild/pre-trained-models/5_experiments/5.1_sst_train/single_vocoder/hifigan.pt models/pre-trained/hifigan.pt
```
Folder structure:
```plaintext
pre-trained-models/
│
├── 5_experiments/                 
│   ├── 5.1_sst-train/        # Selecting synthetic speech to train on
│   │   ├── single_vocoder/   # Detectors trained on synthetic speech from a single vocoder       
│   │   │   ├── hifigan.pt    # Synthetic speech derived from HifiGAN vocoder
│   │   │   ├── melgan.pt     
│   │   │   ├── ...           
│   │   │
│   │   ├── multi_vocoder/    # Detectors trained on synthetic speech from multiple vocoders
│   │       ├── leave_hfg.pt  # Synthetic speech derived from vocoders other than HifiGAN
│   │       ├── leave_melgan.pt    
│   │       ├── ...

```

## Reproducing Experiments 
To reproduce the experiments, please follow the following instructions:

---
**Environment Setup**
```sh
conda create -n SSL python=3.10.14
```
```sh
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Download and install fairseq from [here](https://github.com/facebookresearch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)

```sh
pip install -r requirements.txt
```
Download xlsr model from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr) and add in /synthetic_speech_detection/SSL_Anti-spoofing/model.py

```sh
conda activate SSL 
```
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
gsutil cp -r https://storage.cloud.google.com/ssd_in_the_wild/pre-trained-models/5_experiments/5.1_sst_train/single_vocoder/hifigan.pth synthetic_speech_detection/pre-trained
```

```bash 
gsutil cp -r https://storage.cloud.google.com/ssd_in_the_wild/pre-trained-models/5_experiments/5.2_data-aug/hfg_aug2.pth synthetic_speech_detection/pre-trained
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
gsutil cp https://storage.cloud.google.com/ssd_in_the_wild/pre-trained-models/5_experiments/5.3_more_spks/exp-1/spk1.pt models/pre-trained/spk1.pt
```
Similarly pre-trained model for experiment 2 and single speaker model can be downloaded from -- 
```sh 
https://storage.cloud.google.com/ssd_in_the_wild/pre-trained-models/5_experiments/5.3_more_spks/exp-2/spk1.pt
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
For vocoder systems not included in WaveFake dataset, we release the generated samples for training and can be downloaded from folder -- ``https://storage.cloud.google.com/ssd_in_the_wild/flac_version/ShiftySpeech/Vocoders/<vocoder_of_choice>/ljspeech_flac
``
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
