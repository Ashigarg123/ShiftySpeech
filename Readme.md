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

## Downloading the pre-trained models 

```bash 
gsutil cp -r https://storage.cloud.google.com/ssd_in_the_wild/pre-trained-models/ models/pre-trained/
```
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


**Evaluation:**

```sh
python train.py --eval \
    --test_score_dir <path_to_save_scores> \
    --model_name SSL-AASIST \
    --test_list_path <path_to_test_file_with_utterances_and_labels> \
    --model_path <path_to_model_checkpoint>
```
### Section 3 

Prepare the data 
- Download In-the-Wild dataset from [here](https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa) 
- Create evaluation file of the format :
```sh 
   <real_audio_path> bonafide 
   <synthetic_audio_path> spoof
```

- `bonafide` represents genuine audio.
- `spoof` represents synthetic or fake audio.


Download the pre-trained model 

```bash 
gsutil cp -r https://storage.cloud.google.com/ssd_in_the_wild/pre-trained-models/5_experiments/5.1_sst_train/single_vocoder/hifigan.pth synthetic_speech_detection/pre-trained
```

```bash 
gsutil cp -r https://storage.cloud.google.com/ssd_in_the_wild/pre-trained-models/5_experiments/5.2_data-aug/hfg_aug2.pth synthetic_speech_detection/pre-trained
```


Evaluate the above downloaded models

```sh
conda activate SSL
```
```sh
python train.py --eval \
    --test_score_dir <path_to_save_scores> \
    --model_name SSL-AASIST \
    --test_list_path <path_to_test_file_with_utterances_and_labels> \
    --model_path <path_to_downloaded_model>
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
