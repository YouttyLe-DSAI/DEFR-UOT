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
from tracking import Tracker, git_sha, gpu_name
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


def set_seed(s):
    """Reseed every generator that touches this run.

    Seeding once at import is not enough for an ablation. Building the model
    draws from the global RNG, and --use-uot builds 12 UOTFusion blocks with two
    nn.Linear(128,128) each -- 24 extra draws that the baseline never makes.
    DataLoader(shuffle=True) then takes its RandomSampler seed and its worker
    base seed from that same global RNG, so the baseline arm ends up with a
    different batch order, different sampled frames and different augmentation
    than the three UOT arms. That is a second variable inside an experiment whose
    whole premise is that only one thing changes, and nothing in the logs shows
    it: they all still say seed=1.

    Calling this at the top of each fold, and again before each epoch, makes the
    data stream a pure function of (seed, fold, epoch) -- independent of how many
    modules were constructed, and identical whether a run is resumed or not.
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _worker_init(worker_id):
    """Give each worker a seed derived from the loader generator, not the clock.

    The frame picked per segment (dataloader/video_dataloader.py, np.random.randint)
    and the augmentation both run inside workers, so they need seeding too.
    """
    s = (torch.initial_seed() + worker_id) % (2 ** 31)
    random.seed(s)
    np.random.seed(s)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int)

    parser.add_argument('--exper-name', type=str)
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=224)

    # --- Unbalanced Optimal Transport fusion ---
    parser.add_argument('--use-uot', action='store_true',
                        help='replace the mean-pool broadcast fusion with UOT-aligned fusion')
    parser.add_argument('--uot-eps', type=float, default=0.05,
                        help='entropic regularization; cost is 1-cosine so it lives in [0,2]')
    parser.add_argument('--uot-tau', type=float, default=1.0,
                        help='marginal relaxation; large (e.g. 1e6) recovers balanced OT')
    parser.add_argument('--uot-iters', type=int, default=10)
    parser.add_argument('--uot-mode', type=str, default='uot', choices=['uot', 'attn'],
                        help="how the cost matrix becomes routing weights. 'uot' = "
                             "Sinkhorn scaling (balanced when --uot-tau is large); "
                             "'attn' = row-wise softmax, the control arm that isolates "
                             "whether transport structure matters or just the added "
                             "capacity. Identical parameter count either way.")
    parser.add_argument('--no-prev-ckpt', action='store_true',
                        help='dung giu model.prev.pth. Tiet kiem 780 MB moi run '
                             '(15,6 GB ca loat) nhung mat duong lui khi checkpoint '
                             'moi nhat con nguyen ma sai.')

    # --- Theo doi (tuy chon, chiu loi, khong anh huong so hoc) ---
    parser.add_argument('--wandb', action='store_true',
                        help='gui metric len wandb. Gom hai may vao mot dashboard. '
                             'Hong thi bo qua, log.txt van la nguon su that.')
    parser.add_argument('--wandb-project', type=str, default='defr-uot')
    parser.add_argument('--tb', action='store_true',
                        help='ghi TensorBoard vao <log_dir>/tb')

    parser.add_argument('--uot-detach', action='store_true',
                        help='treat the transport plan as fixed routing weights (no backprop through the solver)')

    parser.add_argument('--num-classes', type=int, default=None,
                        help='override the class count. MAFW defaults to 11; pass 7 to '
                             'train it in the label space it shares with DFEW.')
    parser.add_argument('--train-annotation', type=str, default=None,
                        help="training split, e.g. './annotation/MAFW_set_{fold}_train_faces7.txt'")
    parser.add_argument('--eval-num-classes', type=int, default=None,
                        help='score using only the first N logits. An MAFW model emits 11; '
                             "restricting to 7 puts it on the same footing as a DFEW model, "
                             'which is valid because the first 7 classes match in order.')
    parser.add_argument('--test-annotation', type=str, default=None,
                        help='score on a different corpus than the one trained on, e.g. '
                             "'./annotation/MAFW_set_{fold}_test_faces7.txt'. {fold} is "
                             'expanded per fold. The label spaces must line up: DFEW 7 '
                             "classes == MAFW's first 7, same order.")
    parser.add_argument('--resume-training', type=str, default=None,
                        help='continue an interrupted run: restores weights, optimizer, '
                             'LR schedule and epoch counter from a checkpoint written by '
                             'this script. {fold} is expanded. Different from --resume, '
                             'which loads weights only.')
    parser.add_argument('--resume', type=str, default=None,
                        help="warm-start from a trained checkpoint, e.g. the authors' "
                             "release. May contain {fold}, expanded per fold. Weights only "
                             "-- the optimizer and epoch counter start fresh.")
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='1-based folds to run, e.g. --folds 1. Default: all 5. '
                             'Needed when the session has a wall-clock limit (Kaggle).')

    args = parser.parse_args()
    return args

def main(set, args):

    data_set = set+1

    # Reseed per fold, BEFORE anything draws from the RNG. Without this the fold
    # loop below just continues whatever state the previous fold left behind, so
    # fold 3 of a --folds 1 2 3 4 5 run differs from fold 3 of a --folds 3 run.
    set_seed(seed * 100 + data_set)

    if args.dataset == "DFEW":
        print("*********** DFEW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.txt'
        log_curve_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.png'
        log_confusion_matrix_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/' + 'cn.png'
        checkpoint_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model.pth'
        train_annotation_file_path = "./annotation/DFEW_set_"+str(data_set)+"_train.txt"
        test_annotation_file_path = "./annotation/DFEW_set_"+str(data_set)+"_test.txt"
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/checkpoint/')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/models')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/AudioMAE')
        os.makedirs('./log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/dataloader')

        for filename in ['main.py', 'train_DFEW.sh', 'train_MAFW.sh', 'models/Generate_Model.py', 'models/Temporal_Model.py', 'dataloader/video_dataloader.py', 'dataloader/video_transform.py', 'models/models_vit.py', 'models/uot.py', 'AudioMAE/audio_models_vit.py']:
            shutil.copyfile(filename, './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log/code/'+filename)

    elif args.dataset == "MAFW":
        print("*********** MAFW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.txt'
        log_curve_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/' + 'log.png'
        log_confusion_matrix_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/' + 'cn.png'
        checkpoint_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/'+'checkpoint/'+'model.pth'

        train_annotation_file_path = "./annotation/MAFW_set_"+str(data_set)+"_train_faces.txt"
        test_annotation_file_path = "./annotation/MAFW_set_"+str(data_set)+"_test_faces.txt"
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/checkpoint/')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/models')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/AudioMAE')
        os.makedirs('./log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/dataloader')

        for filename in ['main.py', 'train_DFEW.sh', 'train_MAFW.sh', 'models/Generate_Model.py', 'models/Temporal_Model.py', 'dataloader/video_dataloader.py', 'dataloader/video_transform.py', 'models/models_vit.py', 'models/uot.py', 'AudioMAE/audio_models_vit.py']:
            shutil.copyfile(filename, './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log/code/'+filename)

    # Cross-corpus: train on one corpus, score on another. DFEW's 7 classes are the
    # first 7 of MAFW's in the same order, so a DFEW model can be scored on MAFW-7
    # directly -- see tools/make_mafw7.py, which writes those files.
    if args.train_annotation:
        train_annotation_file_path = args.train_annotation.format(fold=data_set)
        print('Training split overridden: ' + train_annotation_file_path)
    if args.test_annotation:
        test_annotation_file_path = args.test_annotation.format(fold=data_set)
        print('Cross-corpus evaluation on: ' + test_annotation_file_path)
        with open(log_txt_path, 'a') as f:
            f.write('test_annotation=' + test_annotation_file_path + '\n')

    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training name: ' + time_str)
       
    model = GenerateModel(args=args)
  
    # only open learnable part
    for name, param in model.named_parameters():
        param.requires_grad = True #False

    for name, param in model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = False 
        if "audio_model" in name:
            param.requires_grad = False

        if "our_classifier" in name:
            param.requires_grad = True
        if "positional_embedding" in name:
            param.requires_grad = True
        if "learnable_prompts" in name:
            param.requires_grad = True
        if "pos_embed" in name:
            param.requires_grad = True
        if "audio_proj" in name:
            param.requires_grad = True
        if "temporal" in name:
            param.requires_grad = True
        if "gate" in name:
            param.requires_grad = True
        if "context_att" in name:
            param.requires_grad = True
        if "learnable_q" in name:
            param.requires_grad = True
        if "audio_att" in name:
            param.requires_grad = True
        if "norm_xt" in name:
            param.requires_grad = True
        if "norm_xt_2" in name:
            param.requires_grad = True
        if "norm_qs" in name:
            param.requires_grad = True
        if "uot_fusion" in name:
            param.requires_grad = True

    model_parameters = model.parameters()
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        # Both arms of a comparison must warm-start from the same weights for the
        # result to mean anything. strict=False on purpose: a --use-uot run has
        # uot_fusion.* keys the source checkpoint cannot have, and those must stay
        # at their zero-init values so the model still starts equal to the baseline.
        resume_path = args.resume.format(fold=data_set)
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        state = state.get('state_dict', state) if isinstance(state, dict) else state
        # An MAFW checkpoint carries an 11-class head; training MAFW in the 7 shared
        # classes needs its first 7 rows. Slicing is valid only because the class order
        # matches -- DFEW's 7 are MAFW's first 7. Any other mismatch is left to fail.
        model_state = model.state_dict()
        sliced = []
        for k, v in list(state.items()):
            if k in model_state and v.shape != model_state[k].shape:
                want, have = model_state[k].shape, v.shape
                if len(want) == len(have) and want[0] < have[0] and want[1:] == have[1:]:
                    state[k] = v[:want[0]]
                    sliced.append('{} {}->{}'.format(k, tuple(have), tuple(want)))
        if sliced:
            print('  head sliced to the shared label space: ' + ', '.join(sliced))

        msg = model.load_state_dict(state, strict=False)
        unexpected = list(msg.unexpected_keys)
        non_uot_missing = [k for k in msg.missing_keys if 'uot_fusion' not in k]
        print('Resumed from {}'.format(resume_path))
        print('  missing (uot_fusion, expected): {}'.format(len(msg.missing_keys) - len(non_uot_missing)))
        print('  missing (OTHER -- should be 0): {} {}'.format(len(non_uot_missing), non_uot_missing[:5]))
        print('  unexpected (should be 0)      : {} {}'.format(len(unexpected), unexpected[:5]))
        with open(log_txt_path, 'a') as f:
            f.write('resumed_from=' + resume_path + '\n')

    # print params   
    print('************************')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('************************')
    
    with open(log_txt_path, 'a') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '=' + str(v) + '\n')
    
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()
    
    # define optimizer
    # Gate cua UOT khong chiu weight decay. Chung khoi tao 0 va phai di ra xa 0
    # de nhanh UOT co tac dung, nen weight decay la mot luc keo nguoc chieu voi
    # dung co che dang duoc do. Anh huong dinh luong nho -- lr trung binh theo
    # cosine ~5e-5, wd 1e-2, qua 29.225 buoc chi co lai ~1,5% -- nhung no la
    # thien lech mot chieu chong lai gia thuyet, va bo di khong ton gi.
    #
    # Chi ap cho gate. Moi tham so khac giu nguyen wd=1e-2 dung nhu paper, nen
    # nhanh baseline khong doi mot chut nao.
    gate_params = [p for n, p in model.named_parameters()
                   if 'uot_fusion' in n and 'gate' in n and p.requires_grad]
    gate_ids = {id(p) for p in gate_params}
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in gate_ids]
    if gate_params:
        optimizer = torch.optim.AdamW(
            [{'params': other_params, 'weight_decay': args.weight_decay},
             {'params': gate_params, 'weight_decay': 0.0}],
            lr=args.lr)
        print('optimizer: {} tham so thuong (wd={}), {} gate (wd=0)'.format(
            len(other_params), args.weight_decay, len(gate_params)))
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) 
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(list_file=train_annotation_file_path,
                                   num_segments=16,
                                   duration=1,
                                   image_size=args.img_size,
                                   args=args)

    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=args.img_size)

    # Own generator, so the shuffle order comes from (seed, fold, epoch) and not
    # from a global RNG whose state depends on how many modules the arm built.
    train_gen = torch.Generator()
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               generator=train_gen,
                                               worker_init_fn=_worker_init)

    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    start_epoch = 0
    if args.resume_training:
        # Full restore, unlike --resume which takes weights only. Needed when a
        # session dies partway: without the optimizer and the cosine schedule the
        # run would restart its learning-rate cycle and stop being the same
        # experiment.
        # --folds vang mat nghia la chay TAT CA fold -- dung truong hop nguy hiem
        # nhat, nen phai tinh no la nhieu fold chu khong phai mot.
        nhieu_fold = (not args.folds) or len(args.folds) > 1
        if '{fold}' not in args.resume_training and nhieu_fold:
            raise SystemExit(
                'DUNG LAI: resume nhieu fold nhung --resume-training khong chua {fold}.\n'
                '  str.format tren chuoi khong co {fold} tra ve NGUYEN chuoi, nen fold 4\n'
                '  va 5 se nap trong so cua fold 3 -- da train tren du lieu chong lan\n'
                '  test cua chung. So se dep gia tao, khong mot canh bao nao.')
        rt = pick_checkpoint(args.resume_training.format(fold=data_set))
        ck = torch.load(rt, map_location='cpu', weights_only=False)

        # Doi chieu cau hinh TRUOC khi nap. Ba nhanh attn / tau=1e6 / tau=1.0 co
        # ten va shape state_dict GIONG HET nhau (mode/tau/eps chi la thuoc tinh
        # Python, khong nam trong state_dict), nen load_state_dict strict=True van
        # nap sach 100%. Quen mot co la duoc checkpoint lai: 14 epoch nhanh nay,
        # 11 epoch nhanh kia, log tu mau thuan voi chinh no.
        old = ck.get('args')
        if old is None:
            raise SystemExit(
                'DUNG LAI: checkpoint khong luu args nen khong kiem duoc cau hinh.\n'
                '  Day la ban truoc khi co co --uot-mode. Train lai tu dau.')
        for k in ('use_uot', 'uot_mode', 'uot_tau', 'uot_eps', 'uot_iters', 'uot_detach',
                  'num_classes', 'img_size', 'temporal_layers', 'lr', 'epochs',
                  'batch_size', 'weight_decay', 'dataset',
                  'train_annotation', 'test_annotation'):
            a, b = getattr(old, k, None), getattr(args, k, None)
            if a != b:
                raise SystemExit(
                    f'DUNG LAI: {k} lech nhau. checkpoint={a!r} vs dong lenh={b!r}\n'
                    f'  Nap tiep se tao checkpoint lai giua hai cau hinh.')
        ck_fold = ck.get('fold')
        if ck_fold is not None and ck_fold != data_set:
            raise SystemExit(
                f'DUNG LAI: checkpoint thuoc fold {ck_fold} nhung dang chay fold {data_set}.\n'
                f'  Nap tiep la ro ri tap test.')
        print(f"resume: khop cau hinh (uot_mode={getattr(old, 'uot_mode', None)}, "
              f"uot_tau={getattr(old, 'uot_tau', None)}, fold={ck_fold})")

        model.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        if 'scheduler' in ck:
            scheduler.load_state_dict(ck['scheduler'])
        else:
            for _ in range(ck['epoch']):
                scheduler.step()
        start_epoch = ck['epoch']
        best_acc = ck.get('best_acc', 0)
        if 'recorder' in ck:
            recorder = ck['recorder']
        print('Resumed TRAINING from {} at epoch {}/{}'.format(rt, start_epoch, args.epochs))
        with open(log_txt_path, 'a') as f:
            f.write('resumed_training_from={} epoch={}\n'.format(rt, start_epoch))
        if start_epoch >= args.epochs:
            print('  already finished -- nothing to do')

    # Cua so theo doi. Chiu loi hoan toan: mat mang hay chua cai goi cung khong
    # lam do run. log.txt van la nguon su that.
    tracker = Tracker(args, fold=data_set, log_dir=os.path.dirname(log_txt_path),
                      sha=git_sha(), gpu=gpu_name())

    for epoch in range(start_epoch, args.epochs):

        # Pin the data stream to (seed, fold, epoch). Two consequences that both
        # matter here: the four arms see identical batches despite building
        # different numbers of modules, and a run resumed at epoch 14 replays
        # exactly what an uninterrupted run would have seen -- so no RNG state
        # has to be carried in the checkpoint.
        epoch_seed = seed * 100000 + data_set * 1000 + epoch
        set_seed(epoch_seed)
        train_gen.manual_seed(epoch_seed)

        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate_0 = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            f.write('Current learning rate: ' + str(current_learning_rate_0) + '\n')
            print('Current learning rate: ', current_learning_rate_0)        
            
        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model, criterion, args, log_txt_path)
        
        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        # Luu ca args: cac nhanh ablation ('uot' vs 'attn') co ten va so tham so
        # GIONG HET nhau, nen khong ghi lai che do thi khong cach nao biet
        # checkpoint nay train bang gi -- va nap nham se chay im lang, sai ket qua.
        # evaluate.py doc lai truong nay de tu chon dung che do.
        save_checkpoint({'epoch': epoch + 1,
                         'scheduler': scheduler.state_dict(),
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'args': args,
                         'fold': data_set,
                         'recorder': recorder}, is_best,
                        checkpoint_path, keep_prev=not args.no_prev_ckpt)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.2f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: ' + str(epoch_time) + 's' + '\n')

        tracker.epoch(epoch,
                      train_acc=float(train_acc), train_loss=float(train_los),
                      val_acc=float(val_acc), val_loss=float(val_los),
                      lr=current_learning_rate_0, epoch_time=epoch_time)
        if args.use_uot:
            g = uot_gate_values(model)
            tracker.gates(epoch, g)
            # Canh bao som, KHONG tu dung. Quyet dinh dung hay chay tiep la cua
            # nguoi, vi ca hai ket cuc deu co the la ket qua dang bao cao.
            if epoch == 4:
                vals = [abs(v) for k, v in g.items() if not k.startswith('grad_')]
                grads = [abs(v) for k, v in g.items() if k.startswith('grad_')]
                gm = sum(vals) / len(vals) if vals else 0.0
                gg = sum(grads) / len(grads) if grads else None
                if gm < 1e-3:
                    msg = ('CANH BAO epoch 5: |gate| trung binh = {:.2e}, van gan 0.\n'
                           '  Nhanh UOT chua dong gop gi -- run nay dang do lai baseline.'
                           .format(gm))
                    if gg is not None:
                        msg += ('\n  |grad| tren gate = {:.2e} -> {}'.format(
                            gg, 'KHONG co tin hieu: optimizer tu choi dung nhanh nay, '
                                'day la KET QUA' if gg < 1e-6 else
                                'CO tin hieu ma gate khong di theo: van de TOI UU, '
                                'can xem lai lr rieng cho gate hoac khoi tao khac 0'))
                    print(msg)
                    with open(log_txt_path, 'a') as f:
                        f.write(msg + '\n')

    if args.use_uot:
        report_uot_gates(model, log_txt_path)

    last_uar, last_war = computer_uar_war(val_loader, model, checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set, args.class_names)

    tracker.final(last_uar, last_war)
    tracker.close()
    return last_uar, last_war


def uot_gate_values(model):
    """The gate scalars AND the gradient sitting on them.

    Watching the values alone is the cheapest early warning there is -- still at
    zero by epoch 5 and the run is measuring the baseline a second time. But the
    value on its own cannot say WHY, and the two reasons call for opposite
    responses:

      grad ~ 0 too      the branch output does not help the loss. The optimiser
                        is correctly declining to use it. That is a finding, and
                        arguably the most interesting one available here: given
                        a free choice, the model refuses transport-based fusion.

      grad NOT ~ 0      a signal exists and the parameter is not following it.
                        That is an optimisation problem -- and then a higher
                        learning rate on the gates, or a non-zero init, is worth
                        trying, because the negative result would be an artefact.

    Call this AFTER backward() and BEFORE optimizer.zero_grad(), or the grads
    read as None.
    """
    out = {}
    for n, p in model.named_parameters():
        if 'uot_fusion' in n and 'gate' in n:
            k = n.split('uot_fusion.')[-1]
            out[k] = float(p)
            if p.grad is not None:
                out['grad_' + k] = float(p.grad)
    return out


def report_uot_gates(model, log_txt_path):
    """How far the UOT gates moved off their zero initialisation.

    Wiring the module in correctly is not the same as the module doing
    anything. Both gates pass through tanh, so a value still at zero means the
    branch contributed nothing and the run measured the baseline twice --
    which is a result about the training budget, not about UOT, and the two
    read identically in the accuracy numbers alone.
    """
    gates = {n: float(p) for n, p in model.named_parameters()
             if 'uot_fusion' in n and 'gate' in n}
    if not gates:
        return

    import math
    vals = list(gates.values())
    a2v = [v for n, v in gates.items() if 'a2v' in n]
    v2a = [v for n, v in gates.items() if 'v2a' in n]
    lines = [
        '*** UOT gates after training ***',
        '  {} gates   |g| mean {:.4f}   max {:.4f}'.format(
            len(vals), sum(abs(v) for v in vals) / len(vals), max(abs(v) for v in vals)),
        '  tanh(g), i.e. the fraction of its output the branch actually adds:',
        '    a2v  mean {:+.4f}   min {:+.4f}   max {:+.4f}'.format(
            sum(math.tanh(v) for v in a2v) / len(a2v),
            min(math.tanh(v) for v in a2v), max(math.tanh(v) for v in a2v)),
        '    v2a  mean {:+.4f}   min {:+.4f}   max {:+.4f}'.format(
            sum(math.tanh(v) for v in v2a) / len(v2a),
            min(math.tanh(v) for v in v2a), max(math.tanh(v) for v in v2a)),
    ]
    if max(abs(v) for v in vals) < 1e-3:
        lines.append('  !! still at zero -- UOT contributed nothing; this run is the '
                     'baseline again')
    print('\n'.join(lines))
    with open(log_txt_path, 'a') as f:
        f.write('\n'.join(lines) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)

    # switch to train mode
    model.train()

    for i, (images, target, audio) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()
        audio = audio.cuda()
        # compute output
        output = model(images, audio)        
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def _restrict(output, args):
    """Keep only the first N logits when scoring in a smaller label space."""
    n = getattr(args, 'eval_num_classes', None)
    return output[:, :n] if n else output


def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target, audio) in enumerate(val_loader):
            
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            # compute output
            output = _restrict(model(images, audio), args)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, checkpoint_path, keep_prev=True):
    """Write the latest epoch. One file, deliberately -- there is no best-epoch copy.

    The protocol we are reproducing is explicit (MMA-DFER, CVPR 2024 Workshops, 4.2):
    "We train the models on the train set and report the result of final checkpoint,
    i.e., at 25th epoch, on the test set." No early stopping, no best-checkpoint
    selection. Deviating forfeits the right to compare against the published numbers.

    An earlier version also wrote model_best.pth. That is worse than a protocol
    deviation: this codebase has no validation split, so `is_best` is decided by
    val_loader, which is built from the *test* annotation (see the two call sites of
    test_data_loader). A figure taken from that file is the maximum over 25 test
    evaluations -- an optimistically biased estimate obtained by peeking at the test
    set. Writing the file at all leaves a loaded gun: `evaluate.py --checkpoint
    .../model_best.pth` runs perfectly happily and reports the leaked number.

    `is_best` is still tracked and written to the log ("The best accuracy: ..."), so
    the training curve and its peak remain visible for diagnosing instability. What
    is gone is the weights -- and with them, 780 MB per run and the chance of
    reporting them by accident.

    Written atomically. With model_best.pth gone there is exactly one restore point
    per run, and it is rewritten ~780 MB at a time, 25 times per run, 20 runs -- 500
    chances for a power cut, an OOM kill or a full disk to land mid-write. Writing
    straight to the final path would leave a truncated file and lose the whole run.
    os.replace is atomic within a filesystem, so the file on disk is always either
    the previous epoch intact or the new one intact, never half of either.
    """
    tmp = checkpoint_path + '.tmp'
    with open(tmp, 'wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())          # bytes on the platter, not just in page cache

    # Keep the previous epoch beside the current one, by RENAME rather than copy.
    # A copy would add 780 MB of writes per epoch -- 390 GB across the 20-run
    # sweep -- for a file that is about to be overwritten anyway. A rename is a
    # single metadata operation.
    #
    # Atomic write already rules out a truncated file. What this adds is a way
    # back when the newest checkpoint is intact but wrong: a fold that diverged,
    # a bad resume, a disk that returned garbage after a successful write.
    if keep_prev and os.path.exists(checkpoint_path):
        os.replace(checkpoint_path, checkpoint_path.replace('model.pth',
                                                            'model.prev.pth'))
    os.replace(tmp, checkpoint_path)


def pick_checkpoint(path):
    """Return the newest usable checkpoint, preferring model.pth.

    There is a window of one rename between moving the old file to
    model.prev.pth and moving the new one into place. It is a single syscall
    wide, but a run that dies inside it would leave model.pth missing and
    model.prev.pth holding the last good epoch -- so resume looks there rather
    than reporting the run unrecoverable.
    """
    if os.path.exists(path):
        return path
    prev = path.replace('model.pth', 'model.prev.pth')
    if os.path.exists(prev):
        print('CANH BAO: khong thay {} -- dung {} (bi ngat dung luc doi ten). '
              'Ban nay la epoch TRUOC do.'.format(path, prev))
        return prev
    raise SystemExit('DUNG LAI: khong thay checkpoint nao tai {}'.format(path))

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
        # Mot diem don le khong ve ra gi neu chi co linestyle -- anh chi con luoi
        # va chu giai, trong y het nhu run bi hong. Da gay nham lan that khi
        # chay thu 1 epoch de kiem tra duong ong.
        mk = 'o' if self.total_epoch == 1 else None
        plt.plot(x_axis, y_axis, color='g', linestyle='-', marker=mk, label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', marker=mk, label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', marker=mk, label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', marker=mk, label='valid-loss-x50', lw=2)
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

def computer_uar_war(val_loader, model, checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set, class_names):
    
    pre_trained_dict = torch.load(checkpoint_path, weights_only=False)['state_dict']
    model.load_state_dict(pre_trained_dict)
    
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images, target, audio) in enumerate(tqdm.tqdm(val_loader)):
            
            images = images.cuda()
            target = target.cuda()
            audio = audio.cuda()
            output = model(images, audio)
            n_eval = getattr(args, 'eval_num_classes', None)
            if n_eval:
                output = output[:, :n_eval]

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

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))

    if args.dataset == "DFEW":
        title_ = "Confusion Matrix on DFEW fold "+str(data_set)
    elif args.dataset == "MAFW":
        title_ = "Confusion Matrix on MAFW fold "+str(data_set)

    # class_names comes from the dataset name and may be longer than the label space
    # actually in use (MAFW lists 11 while --num-classes 7 trains on the shared 7).
    class_names = class_names[:normalized_cm.shape[0]]
    plot_confusion_matrix(normalized_cm, classes=class_names, normalize=True, title=title_)
    plt.savefig(os.path.join(log_confusion_matrix_path))
    plt.close()
    
    with open(log_txt_path, 'a') as f:
        f.write('************************' + '\n')
        f.write(checkpoint_path)
        f.write("Confusion Matrix Diag:" + '\n')
        f.write(str(list_diag.tolist()) + '\n')
        f.write('UAR: {:.2f}'.format(uar) + '\n')        
        f.write('WAR: {:.2f}'.format(war) + '\n')
        f.write('************************' + '\n')
    
    return uar, war


if __name__ == '__main__':
    args = parse_args() 
    UAR = 0.0
    WAR = 0.0
    now = datetime.datetime.now()
    time_str = now.strftime("%y%m%d%H%M")
    time_str = time_str + args.exper_name

    print('************************')
    for k, v in vars(args).items():
        print(k,'=',v)
    print('************************')

    if args.dataset == "DFEW":
        args.number_class = args.num_classes or 7
        args.class_names = [
	'happiness.',
	'sadness.',
	'neutral.',
	'anger.',
	'surprise.',
	'disgust.',
	'fear.'
        ]

        all_fold = 5
    elif args.dataset == "MAFW":
        all_fold = 5
        args.number_class = args.num_classes or 11
        # First 7 match DFEW's order, which is what makes the shared label space work.
        args.class_names = ['happiness', 'sadness', 'neutral', 'anger', 'surprise',
                            'disgust', 'fear', 'contempt', 'anxiety', 'helplessness',
                            'disappointment']

    folds = list(range(all_fold)) if args.folds is None else [f - 1 for f in args.folds]
    assert all(0 <= f < all_fold for f in folds), '--folds must be within 1..{}'.format(all_fold)

    for set in folds:
        uar, war = main(set, args)
        UAR += float(uar)
        WAR += float(war)

    print('********* Final Results *********')
    print("Averaged over folds: {}".format([f + 1 for f in folds]))
    print("UAR: %0.2f" % (UAR/len(folds)))
    print("WAR: %0.2f" % (WAR/len(folds)))
    print('*********************************')
