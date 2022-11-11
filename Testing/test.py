from __future__ import division
import time
import os
import argparse
import sys
import torch
import csv
import re

# python test.py -dataset TSU -mode rgb -split_setting CS -model PDAN -train True -num_channel 512 -lr 0.0002
# -kernelsize 2 -APtype map -epoch 140 -batch_size 1 -comp_info TSU_CS_RGB_PDAN -load_model ./Testing/dataset/


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-test', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='50')
parser.add_argument('-model', type=str, default='')
parser.add_argument('-APtype', type=str, default='wap')
parser.add_argument('-randomseed', type=str, default='False')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-num_channel', type=str, default='False')
parser.add_argument('-batch_size', type=str, default='False')
parser.add_argument('-kernelsize', type=str, default='False')
parser.add_argument('-feat', type=str, default='False')
parser.add_argument('-split_setting', type=str, default='CS')
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# set random seed
if args.randomseed == "False":
    SEED = 0
elif args.randomseed == "True":
    SEED = random.randint(1, 100000)
else:
    SEED = int(args.randomseed)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# print('Random_SEED!!!:', SEED)

from torch.optim import lr_scheduler
from torch.autograd import Variable

import json

import pickle
import math

# dirpath = os.getcwd()
# print("current directory is : " + dirpath)

if str(args.APtype) == 'map':
    from apmeter import APMeter

batch_size = int(args.batch_size)

if args.dataset == 'TSU':
    split_setting = str(args.split_setting)

    from Testing.smarthome_i3d_per_video import TSU as Dataset
    from smarthome_i3d_per_video import TSU_collate_fn as collate_fn

    classes = 51

    if split_setting == 'CS':
        test_split = './Testing/data/smarthome_CS_51.json'

    elif split_setting == 'CV':
        test_split = './Testing/data/smarthome_CV_51.json'

    # rgb_root = 'C:/Users/angyi/Documents/SIT Year 3/3104/Project/TSU_RGB_i3d_feat/RGB_i3d_16frames_64000_SSD'
    rgb_root = "C:/Users/angyi/Documents/SIT Year 3/3104/Project/TSU_RGB_i3d_feat/RGB_i3d_16frames_64000_SSD"
    # skeleton_root='C:/Users/angyi/Documents/SIT Year 3/3104/Project/TSU_3DPose_AGCN_feat/2sAGCN_16frames_64000'
    # rgb_root = './Testing/RGB_i3d_16frames_64000_SSD'
    skeleton_root = './Testing/TSU_3DPose_AGCN_feat/2sAGCN_16frames_64000'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data_rgb_skeleton(val_split, root_skeleton, root_rgb):
    # Load Data

    val_dataset = Dataset(val_split, 'testing', root_skeleton, root_rgb, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)  # 2

    dataloaders = {'val': val_dataloader}
    datasets = {'val': val_dataset}
    return dataloaders, datasets



def load_data(val_split, root):
    # Load Data
    # print(val_split, root, batch_size, classes)

    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'val': val_dataloader}
    datasets = {'val': val_dataset}
    return dataloaders, datasets


def load_labels():
    with open("./Testing/data/all_labels.txt") as file_in:
        lines = []
        for line in file_in:
            line = re.sub(r'\d+', '', line)
            line = line.strip()
            lines.append(line)
        return lines


# test the model
def run(models, criterion, num_classes):
    since = time.time()

    probs = []
    for model, gpu, dataloader, optimizer, sched, model_file in models:
        prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], num_classes)
        probs.append(prob_val)
        sched.step(val_loss)

    print("Testing has concluded")


def run_network(model, data, gpu, epoch=0, baseline=False):
    inputs, mask, labels, other = data
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    mask_list = torch.sum(mask, 1)
    mask_new = np.zeros((mask.size()[0], classes, mask.size()[1]))
    for i in range(mask.size()[0]):
        mask_new[i, :, :int(mask_list[i])] = np.ones((classes, int(mask_list[i])))
    mask_new = torch.from_numpy(mask_new).float()
    mask_new = Variable(mask_new.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)
    activation = model(inputs, mask_new)

    outputs_final = activation

    if args.model == "PDAN":
        # print('outputs_final1', outputs_final.size())
        outputs_final = outputs_final[:, 0, :, :]
    # print('outputs_final',outputs_final.size())
    outputs_final = outputs_final.permute(0, 2, 1)
    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)

    loss = loss_f

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def val_step(model, gpu, dataloader, classes):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0
    # Save list of activity label from text file
    event_list = load_labels()
    full_probs = {}
    print("Testing in progress")

    # Iterate over data.
    for data in dataloader:
        num_iter += 1

        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, classes)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probs = probs.squeeze()

        full_probs[other[0][0]] = probs.data.cpu().numpy().T

    epoch_loss = tot_loss / num_iter

    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    print('val-map:', val_map)
    print(100 * apm.value())

    fields = ['val-map', 'apm']
    rows = []
    init_flag = False
    tempApm = apm.value().tolist()
    for value in tempApm:
        if not init_flag:
            rows.append([float(val_map), value])
            init_flag = True
        else:
            rows.append(['', value])

    filename = "./Testing/results/results.csv"
    with open(filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        csv_writer.writerows(rows)
        print('Results saved into ./Testing/results/')

    apm.reset()
    return full_probs, epoch_loss, val_map


if __name__ == '__main__':
    print(str(args.model))
    # print('batch_size:', batch_size)
    # print('cuda_avail', torch.cuda.is_available())

    if args.mode == 'flow':
        print('flow mode')
        dataloaders, datasets = load_data(test_split, flow_root)
    elif args.mode == 'skeleton':
        print('Pose mode')
        dataloaders, datasets = load_data(test_split, skeleton_root)
    elif args.mode == 'rgb':
        print('RGB mode')
        dataloaders, datasets = load_data(test_split, rgb_root)

    if args.test:
        num_channel = args.num_channel
        if args.mode == 'skeleton':
            input_channnel = 256
        else:
            input_channnel = 1024

        num_classes = classes
        mid_channel = int(args.num_channel)

        if args.model == "PDAN":
            print("you are processing PDAN")
            from models import PDAN as Net

            model = Net(num_stages=1, num_layers=5, num_f_maps=mid_channel, dim=input_channnel, num_classes=classes)

        model = torch.nn.DataParallel(model)

        if args.load_model != "False":
            # entire model
            args.load_model = args.load_model.strip("'")
            model = torch.load(args.load_model)
            # weight
            # model.load_state_dict(torch.load(str(args.load_model)))
            # print("loaded", args.load_model)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print('pytorch_total_params', pytorch_total_params)
        # print('num_channel:', num_channel, 'input_channnel:', input_channnel, 'num_classes:', num_classes)
        model.cuda()

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        # print(lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)
        run([(model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion, num_classes=classes)
