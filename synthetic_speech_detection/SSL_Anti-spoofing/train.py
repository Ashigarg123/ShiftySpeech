import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval, pad, process_Rawboost_feature
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
import wandb
from tqdm import tqdm
import librosa 
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index], frr, far


def calculate_tDCF_EER(cm_scores_file,
                       output_file,
                       printout=True):
    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 2].astype(float)
     # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    all_scores = np.concatenate([bona_cm, spoof_cm])
    all_true_labels = np.concatenate([np.ones_like(bona_cm), np.zeros_like(spoof_cm)])
    
    auc = roc_auc_score(all_true_labels, all_scores, max_fpr=0.05)
    eer_cm, eer_threshold, frr, far = compute_eer(bona_cm, spoof_cm)
    
    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(Equal error rate for countermeasure)\n'.format(
                            eer_cm * 100))
            f_res.write('\t pAUC with max fpr - 0.05 is :{}'.format(auc))


def evaluate_accuracy(dev_loader, model, device, args):
    val_loss = 0.0
    num_total = 0.0
    algo = args.algo
    cut = 64600
    model.eval()
    
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    progress_bar = tqdm(dev_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    for current_step, (batch_pths, batch_y) in  enumerate(progress_bar):
        batch_x = batch_pths
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path, trial_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()

    fname_list = []
    score_list = []
    
    for batch_x,utt_id in data_loader:
        
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        batch_out = model(batch_x)
        
        batch_score = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel() 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
    assert len(trial_lines) == len(fname_list) == len(score_list)
       
    with open(save_path, 'a+') as fh:
        for fname, cm, trl in zip(fname_list,score_list, trial_lines):
            utt_id,key = trl.strip().split(' ')
            assert fname == utt_id
            fh.write('{} {} {}\n'.format(fname, key, cm))
    fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr,optim, device, args):
    running_loss = 0
    
    num_total = 0.0
    algo = args.algo
    model.train()
    cut = 64600
    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    for current_step, (batch_pths, batch_y) in  enumerate(progress_bar):
        batch_x = batch_pths
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SSL-AASIST baseline system')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--model_name', type=str, default="SSL-AASIST")
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--trn_list_path',  default=None, help="path to train file")
    parser.add_argument('--dev_list_path', default=None, help="path to validation file")
    parser.add_argument('--test_list_path', default=None, help="path to test file")
    parser.add_argument('--test_score_dir', default=None, help="path to save test scores")
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--save_path', type=str, default=".", help="Model save path")
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
   

    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project_name = os.getenv("WANDB_PROJECT_NAME")

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    wandb.login(key=wandb_api_key)
    wandb.init(project=wandb_project_name , config={
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay
    })

    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}'.format(
        args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join(args.save_path,model_tag)
    
    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))


    #evaluation 

    if args.eval:
        file_eval = genSpoof_list(dir_meta = args.test_list_path,is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2021_eval(list_IDs=file_eval)
        eval_output = os.path.join(args.test_score_dir, f"{args.model_name}_model_score.txt")
        produce_evaluation_file(eval_set, model, device, eval_output, args.test_list_path)
        output_file = os.path.join(args.test_score_dir, f"{args.model_name}_model_eer.txt")
        eval_eer = calculate_tDCF_EER(
            cm_scores_file=eval_output,output_file=output_file)

        sys.exit(0)
   
    trn_list_path = args.trn_list_path
    dev_trial_path  = args.dev_list_path
    train_set=Dataset_ASVspoof2019_train(args,metafile=trn_list_path,algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=16, shuffle=True,drop_last=True)
    del train_set
    
    dev_set = Dataset_ASVspoof2019_train(args,metafile=dev_trial_path,algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=16, shuffle=False)
    del dev_set
    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    
    for epoch in range(num_epochs):
        
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device, args)
        val_loss = evaluate_accuracy(dev_loader, model, device, args)
        wandb.log({"epoch": epoch, "train_loss": running_loss, "val_loss": val_loss})
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch,
                                                   running_loss,val_loss))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
