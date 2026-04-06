"""
Retinal dataset for paired CFP-FFA multi-modal learning.
Loads paired Color Fundus Photography (CFP) and Fluorescein Fundus Angiography (FFA)
images with multi-label disease annotations.
"""

import os
import logging

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

logger = logging.getLogger(__name__)


class RetinalDataset(Dataset):
    def __init__(self, diseasesDict, root, modals, modal_format, fold, lengthFA,
                 back_ratio=0.7, imgSize=224, isTraining=True, isRotate=False):
        self.disease_dict = diseasesDict
        self.isTraining = isTraining
        self.root = root
        self.modal = modals
        self.modalFormat = modal_format
        self.name = None
        self.imgsize = imgSize
        self.fold = fold
        self.isRotate = isRotate
        self.lengthFA = lengthFA
        self.allItems = self.getAllPath(root, fold, isTraining)
        self.ratio = back_ratio

    def __len__(self):
        return len(self.allItems)

    def dataTransform(self, data, crop_size, isRotate=False, isTraining=True):
        """Apply Resize + Padding + RandomCrop (train) or Resize (eval) + ToTensor."""
        data_processed = []
        trans_resizeAndPad = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.Pad(15)
        ])
        trans_resize = transforms.Resize((crop_size, crop_size))
        trans_tensor = transforms.ToTensor()

        rotate_angle = random.randint(-15, 15)
        i, j, h, w = transforms.RandomCrop.get_params(
            trans_resizeAndPad(data[0]), output_size=(crop_size, crop_size))

        for item in data:
            if isTraining == 'train':
                dataTmp = trans_resizeAndPad(item)
                dataTmp = TF.crop(dataTmp, i, j, h, w)
                if isRotate:
                    dataTmp = dataTmp.rotate(rotate_angle)
            else:
                dataTmp = trans_resize(item)
            data_processed.append(trans_tensor(dataTmp))
        return data_processed

    def __getitem__(self, index):
        path, diseases_tensor, name = self.allItems[index]
        imgFA = []
        imgCFP = []
        numberFA = 0

        for modal in self.modal:
            format_f = self.modalFormat[modal]
            if 'FFA' in modal:
                nowPath = os.path.join(path, modal)
                listAll = os.listdir(nowPath)
                listAll.sort()

                if format_f[2] == 'one':
                    # Load the middle single frame
                    idx = -2 if len(listAll) > 1 else -1
                    img = Image.open(os.path.join(nowPath, listAll[idx]))
                    imgFA.append(img)

                elif format_f[2] == 'three':
                    # Load early, middle, and late phase frames
                    listNum = [0, len(listAll) // 2, -1]
                    for i in listNum:
                        try:
                            img = Image.open(os.path.join(nowPath, listAll[i])).convert('L')
                            # Filter frames with excessive black background
                            img_arr = np.array(img)
                            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
                            _, thresh = cv2.threshold(img_bgr, 15, 1, cv2.THRESH_BINARY)
                            ratio = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])
                            if ratio > self.ratio:
                                imgFA.append(img)
                        except OSError:
                            print('Error loading FFA frame:', os.path.join(nowPath, listAll[i]))

                    # Pad to 3 frames if some were filtered out
                    if len(imgFA) < 3:
                        try:
                            padImg = imgFA[-1]
                            for _ in range(3 - len(imgFA)):
                                imgFA.append(padImg)
                        except IndexError:
                            print('Error: no valid FFA frames for', path, name)

                else:
                    # Load all frames (up to lengthFA), centered
                    if len(listAll) > self.lengthFA:
                        numberFA = self.lengthFA
                        start = int((len(listAll) - self.lengthFA) / 2)
                        listAll = listAll[start:start + self.lengthFA]
                    else:
                        numberFA = len(listAll)
                    for i, nameTemp in enumerate(listAll):
                        if i >= self.lengthFA:
                            break
                        try:
                            img = Image.open(os.path.join(nowPath, nameTemp)).convert('L')
                            imgFA.append(img)
                        except OSError:
                            print('Error:', os.path.join(nowPath, nameTemp))
                            numberFA -= 1
                    # Pad with last frame if sequence is shorter than lengthFA
                    if len(listAll) < self.lengthFA:
                        padImg = imgFA[-1]
                        for _ in range(self.lengthFA - len(listAll)):
                            imgFA.append(padImg)

            else:
                # Load CFP image
                nowPath = os.path.join(path, modal)
                listAll = os.listdir(nowPath)
                img = Image.open(os.path.join(nowPath, listAll[0]))
                imgCFP.append(img)

        # Apply transforms
        items = []
        if len(imgFA) != 0:
            try:
                items.append(self.dataTransform(imgFA, self.imgsize,
                             isTraining=self.isTraining, isRotate=self.isRotate))
            except OSError:
                print("FFA image file is truncated:", path)
        else:
            print(name, 'FFA not exist')

        if len(imgCFP) != 0:
            try:
                items.append(self.dataTransform(imgCFP, self.imgsize,
                             isTraining=self.isTraining, isRotate=self.isRotate))
            except OSError:
                print("CFP image file is truncated:", path)
                return
        else:
            print(name, 'CFP not exist')
            items.append(None)

        # Sparse label representation (padded to fixed length for batching)
        label_sparse = torch.nonzero(diseases_tensor).squeeze(-1)
        pad_width = 5 - len(label_sparse)
        pad_value = label_sparse[-1]
        label_sparse = F.pad(label_sparse, (0, pad_width), 'constant', pad_value)

        if diseases_tensor.sum() == 0:
            print(name, 'has all-zero labels!')

        return items, diseases_tensor, name, numberFA, label_sparse

    def convertLabel(self, label):
        """Map fine-grained disease subtypes to canonical categories."""
        if 'DryAMD' in label or 'WetAMD' in label:
            label.append('AMD')
        if 'BRVO' in label or 'CRVO' in label:
            label.append('RVO')
        if 'NPDR' in label or 'SNPDR' in label or 'PDR' in label:
            label.append('DR')
        return label

    def getAllPath(self, root, fold, isTraining):
        """Load sample paths and labels from the Excel label file."""
        items = []
        if 'train' in isTraining:
            filePath = os.path.join(root, 'label', fold[0], fold[1], 'train.xlsx')
        elif isTraining == 'validation':
            filePath = os.path.join(root, 'label', fold[0], fold[1], 'validation.xlsx')
        elif isTraining == 'test':
            filePath = os.path.join(root, 'label', fold[0], fold[1], 'test.xlsx')
        else:
            raise ValueError('Invalid isTraining value: {}'.format(isTraining))

        csvFile = pd.read_excel(filePath, engine='openpyxl')

        for idx in range(len(csvFile)):
            path = os.path.join(root, 'dataAll',
                                csvFile.loc[idx, 'name'],
                                str(csvFile.loc[idx, 'Exam_Date']),
                                csvFile.loc[idx, 'OSOD'])
            label = csvFile.loc[idx, 'label']
            diseases = label.split(',')
            diseases = self.convertLabel(diseases)
            diseases_tensor = torch.tensor(
                [1 if key in diseases else 0 for key in self.disease_dict],
                dtype=torch.float32)

            # Skip samples with no valid disease label
            if torch.sum(diseases_tensor) == 0:
                continue

            # Ensure both modality directories exist and are non-empty
            modal0_path = os.path.join(path, self.modal[0])
            modal1_path = os.path.join(path, self.modal[1])
            if os.path.exists(modal0_path) and os.path.exists(modal1_path):
                if len(os.listdir(modal1_path)) > 0:
                    try:
                        if len(os.listdir(modal0_path)) > 0:
                            items.append([path, diseases_tensor, csvFile.loc[idx, 'key']])
                    except NotADirectoryError:
                        continue

        return items
