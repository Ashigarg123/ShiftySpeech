Environment set-up:
```sh
conda create -n SSL python=3.10.14
```
```sh
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
```sh
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
```
```sh
pip install --editable ./
```
```sh
pip install -r requirements.txt
```
Download xlsr model from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr) and add in baseline_systems/SSL_Anti-spoofing/models


**Evaluation:**

```sh
python train.py --eval \
    --test_score_dir <path_to_save_scores> \
    --model_name SSL-AASIST \
    --test_list_path <path_to_test_file_with_utterances_and_labels> \
    --model_path <path_to_model_checkpoint>
```
