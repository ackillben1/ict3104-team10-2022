import os
import sys
import argparse
from multiprocessing import cpu_count
import torch
import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:

    def _rebuild_tensor_v2(
        storage, storage_offset, size, stride, requires_grad, backward_hooks
    ):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor

    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import numpy as np
from pytorch_i3d import InceptionI3d
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-mode", type=str, help="rgb or flow")
parser.add_argument("-load_model", type=str)
parser.add_argument("-root", type=str)
parser.add_argument("-gpu", type=str)
parser.add_argument("-save_dir", type=str)
parser.add_argument("-split", type=str)
parser.add_argument("-window_size", type=int)
args = parser.parse_args()


def get_frames(vid_name, root):
    frames = []
    count = 0
    vidCap = cv2.VideoCapture(os.path.join(root, vid_name))
    success, img = vidCap.read()
    while success:
        frames.append(img)
        success, img = vidCap.read()
        if success:
            count += 1

    return count, frames


def make_dataset(vid_name, root):
    frames = []
    count = 0
    vidCap = cv2.VideoCapture(os.path.join(root, vid_name))
    success, img = vidCap.read()
    while success:
        if count == 40:
            break
        w, h, c = img.shape
        if w < 224 or h < 224:
            d = 226.0 - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.0) * 2 - 1
        frames.append(img)
        success, img = vidCap.read()
        if success:
            count += 1

    print(f"Read {count} frames")
    frames = np.asarray(frames, dtype=np.float32)
    frames = frames[np.newaxis, :, :, :, :]

    return count, video_to_tensor(frames)


def video_to_tensor(pic):
    return torch.from_numpy(pic.transpose([0, 4, 1, 2, 3]))


def run(mode="rgb", root="", vid_name="", batch_size=1, load_model="", save_dir=""):
    # print(root)
    # if os.path.exists(os.path.join(save_dir, vid_name + '.npy')):
    #    print('feature exist!')
    #    exit()

    # setup the model
    if mode == "flow":
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(48, in_channels=3)

    i3d.load_state_dict(torch.load(load_model))
    i3d = i3d.cuda()
    i3d_parallel = torch.nn.DataParallel(i3d)

    frameCount, frames = get_frames(vid_name + ".mp4", root)
    print(f"Frame Count: {frameCount+1}")

    features = []
    np_frames = []

    i3d_parallel.train(False)
    # if os.path.exists(os.path.join(save_dir, vid_name + '.npy')):
    #    exit()

    print("Processing frame(s)")
    for count in tqdm(range(frameCount)):

        if count % 10 != 0 or count == 0:

            img = frames[count]
            w, h, c = img.shape
            if w < 224 or h < 22:
                d = 226.0 - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            img = (img / 255.0) * 2 - 1
            np_frames.append(img)
            count += 1
        else:
            np_frames = video_to_tensor(
                np.asarray(np_frames, dtype=np.float32)[np.newaxis, :, :, :, :]
            )
            np_frames_processed = Variable(np_frames.cuda())
            temp_features = i3d_parallel.module.extract_features(np_frames_processed)
            features.append(
                temp_features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
            )
            np_frames = []

            img = frames[count]
            w, h, c = img.shape
            if w < 224 or h < 224:
                d = 226.0 - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            img = (img / 255.0) * 2 - 1
            np_frames.append(img)
            count += 1

    np.save(os.path.join(save_dir, vid_name), features)
    print("save_path: ", os.path.join(save_dir, vid_name) + ".npy")


if __name__ == "__main__":
    # need to add argparse
    print("cuda_avail", torch.cuda.is_available())
    torch.cuda.empty_cache()
    run(
        mode=args.mode,
        root=args.root,
        load_model=args.load_model,
        save_dir=args.save_dir,
        vid_name=args.split,
    )
