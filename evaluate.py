from models import models_vit
import sys
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.Generate_Model import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
from sklearn.metrics import confusion_matrix
import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random

seed = 1
random.seed(seed)  
np.random.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DFEW')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=224)

    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='evaluate several folds and average, e.g. --folds 1 2 3 4 5. '
                             'The published figures are 5-fold means, so a single fold is '
                             'not directly comparable to them.')

    # --- Unbalanced Optimal Transport fusion ---
    # These must match the flags used at training time, and load_state_dict will
    # NOT catch it if they do not. mode/tau/eps/iters are plain attributes on
    # UOTFusion, absent from the state_dict, so every arm loads into every other
    # arm with zero missing or unexpected keys -- and then computes the wrong
    # thing. That is why they are read back from the checkpoint below instead of
    # being trusted from the command line.
    parser.add_argument('--eval-num-classes', type=int, default=None,
                        help='score using only the first N logits (MAFW model -> 7 classes)')
    parser.add_argument('--test-annotation', type=str, default=None,
                        help='score on a different corpus than the checkpoint was trained '
                             "on, e.g. './annotation/MAFW_set_{fold}_test_faces7.txt'")
    parser.add_argument('--zero-audio', action='store_true',
                        help='replace every spectrogram with the constant a missing .wav '
                             'produces, to measure what the audio branch actually contributes')
    parser.add_argument('--use-uot', action='store_true')
    parser.add_argument('--uot-eps', type=float, default=None)
    parser.add_argument('--uot-tau', type=float, default=None)
    # All four default to None on purpose, so we can tell "user asked for this" from
    # "argparse filled it in". None of mode/tau/eps/iters lives in the state_dict --
    # they are plain Python attributes on UOTFusion -- so every arm loads cleanly into
    # every other arm and the mismatch is invisible. tau in particular is the ONLY
    # thing separating the balanced arm (1e6) from the unbalanced one (1.0).
    parser.add_argument('--uot-iters', type=int, default=None)
    parser.add_argument('--uot-mode', type=str, default=None, choices=['uot', 'attn'])
    parser.add_argument('--uot-detach', action='store_true', default=None)

    args = parser.parse_args()
    return args

def main(set, args):
    
    data_set = set
    
    if args.dataset == "DFEW":
        print("*********** DFEW Dataset Fold  " + str(data_set) + " ***********")
        test_annotation_file_path = "./annotation/DFEW_set_"+str(data_set)+"_test.txt"
        args.number_class = 7     
    elif args.dataset == "MAFW":
        print("*********** MAFW Dataset Fold  " + str(data_set) + " ***********")
        test_annotation_file_path = "./annotation/MAFW_set_"+str(data_set)+"_test_faces.txt"
        args.number_class = 11      
    if args.test_annotation:
        test_annotation_file_path = args.test_annotation.format(fold=data_set)
        print('Cross-corpus evaluation on: ' + test_annotation_file_path)

    # Doc uot_mode TU CHECKPOINT, dung tin dong lenh. Nhanh 'attn' va 'uot' co
    # ten va so tham so GIONG HET nhau, nen nap nham che do van thanh cong,
    # khong mot canh bao nao, va tinh sai toan bo. Day dung loai loi am tham da
    # tra gia ba lan trong du an nay -- lan nay chan truoc.
    if args.use_uot:
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        old = ck.get('args')
        ck_fold = ck.get('fold')
        del ck
        # MAC DINH cua ban truoc khi co viec luu args -- chi dung khi checkpoint cu.
        MD = {'uot_mode': 'uot', 'uot_tau': 1.0, 'uot_eps': 0.05,
              'uot_iters': 10, 'uot_detach': False}
        for k, mac_dinh in MD.items():
            saved = getattr(old, k, None) if old is not None else None
            truyen = getattr(args, k)
            if truyen is None:
                setattr(args, k, saved if saved is not None else mac_dinh)
            elif saved is not None and saved != truyen:
                raise SystemExit(
                    f"DUNG LAI: checkpoint train bang {k}={saved!r} nhung dong lenh "
                    f"truyen {truyen!r}.\n  Cac nhanh co cung ten va shape tham so nen "
                    f"nap van sach, chi la tinh sai. Bo co do di de lay tu checkpoint.")
        if old is None:
            print('CANH BAO: checkpoint khong luu args -- dung mac dinh ban cu. '
                  'Neu day la nhanh attn hoac tau=1e6 thi so bao cao SE SAI.')
        print('uot: ' + '  '.join(f'{k}={getattr(args, k)}' for k in MD)
              + f'  fold={ck_fold}')

    model = GenerateModel(args=args)
    model = torch.nn.DataParallel(model).cuda()
    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=args.img_size)

    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=1, #args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
 
    uar, war = computer_uar_war(val_loader, model, args.checkpoint, data_set,
                                zero_audio=args.zero_audio)
  
    return uar, war

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

# What dataloader.video_dataloader produces for a clip whose .wav is missing:
# torch.zeros(512,128) pushed through (x + 4.2677393) / (4.5689974 * 2).
DEAD_AUDIO_VALUE = 4.2677393 / (4.5689974 * 2)


def computer_uar_war(val_loader, model, checkpoint_path, data_set, zero_audio=False):
    
    pre_trained_dict = torch.load(checkpoint_path, weights_only=False)['state_dict']
    model.load_state_dict(pre_trained_dict)
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images, target, audio) in enumerate(tqdm.tqdm(val_loader)):
            
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            if zero_audio:
                audio = torch.full_like(audio, DEAD_AUDIO_VALUE)
            output = model(images, audio)
            if getattr(args, 'eval_num_classes', None):
                output = output[:, :args.eval_num_classes]

            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(val_loader.dataset)
    
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()
        
    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)
    if zero_audio:
        print("(audio was zeroed -- compare against the normal run to size the audio branch)")
    return uar, war


if __name__ == '__main__':
    args = parse_args() 
    print('************************')
    for k, v in vars(args).items():
        print(k,'=',v)
    print('************************')
    folds = args.folds if args.folds else [args.fold]
    template = args.checkpoint
    results = []

    for fold in folds:
        # {fold} lets one invocation walk the per-fold checkpoints of a release.
        args.checkpoint = template.format(fold=fold) if '{fold}' in template else template
        uar, war = main(fold, args)
        results.append((fold, uar, war))

    print('********* Final Results *********')
    for fold, uar, war in results:
        print("fold %d   UAR: %0.2f   WAR: %0.2f" % (fold, uar, war))
    if len(results) > 1:
        print('---')
        print("mean over %d folds   UAR: %0.2f   WAR: %0.2f" % (
            len(results),
            sum(r[1] for r in results) / len(results),
            sum(r[2] for r in results) / len(results)))
    print('*********************************')
