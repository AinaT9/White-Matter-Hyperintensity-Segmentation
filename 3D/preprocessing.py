import os
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
from skimage.transform import resize

def createDictionaryM(path:str, DICT: dict,location:str):
    subdirectories= os.listdir(path)
    for x in subdirectories:
        DICT["mask"].append(os.path.join(path,x, "wmh.nii.gz"))
        DICT["pathsT1"].append(os.path.join(path,x, "pre", "T1.nii.gz"))
        DICT["pathsFLAIR"].append(os.path.join(path,x, "pre", "FLAIR.nii.gz"))
        DICT["location"].append(location)
        DICT["ID"].append(x)

def createDictionary(path:str, DICT: dict):
    subdirectories= os.listdir(path)
    for x in subdirectories:
        for y in os.listdir(os.path.join(path, x)):
            t1_file = f"{x}_{y}_T1.nii.gz"
            t2_file = f"{x}_{y}_T2.nii.gz"
            flair_file = f"{x}_{y}_FLAIR.nii.gz"
            mask_file = f"{x}_{y}_MASK.nii.gz"
            DICT["mask"].append(os.path.join(path,x,y,mask_file))
            DICT["pathsT1"].append(os.path.join(path,x, y,t1_file))
            DICT["pathsFLAIR"].append(os.path.join(path,x, y,flair_file))
            DICT["pathsT2"].append(os.path.join(path,x, y,t2_file))

def getIDsICPR(dictio:dict):
    lista =["P2_T1","P2_T4","P3_T3","P4_T2","P5_T2","P8_T1","P9_T1","P9_T2","P10_T1","P10_T2","P11_T2",
            "P20_T3","P25_T1","P36_T1","P40_T1","P45_T1","P46_T1","P51_T2","P52_T2",]
    val={k:[] for k in dictio.keys()}
    train={k:[] for k in dictio.keys()}
    for x,i in enumerate(dictio["pathsFLAIR"]):

        filename = i.split("\\")[-1]
        identifier = filename.split('_')[0] + "_" + filename.split('_')[1]
        if identifier in lista:
            for key in dictio.keys():
                val[key].append(dictio[key][x])
        else:
            for key in dictio.keys():
                train[key].append(dictio[key][x])
    return train, val               


def getIDs(dictio:dict):
    Utrecht=["4","11","17","21","25","29"]
    Singapore=["50","51","55","62"]
    Amsterdam=["109","132"]
    val={k:[] for k in dictio.keys()}
    train={k:[] for k in dictio.keys()}
    for i in range(len(dictio["ID"])):
        k=dictio['ID'][i]
        if k in Utrecht or k in Singapore or k in Amsterdam:
            for key in dictio.keys():
                val[key].append(dictio[key][i])
        else:
            for key in dictio.keys():
                train[key].append(dictio[key][i])
    return train, val  

class ICPR(Dataset):
    def __init__(self, flair, t1,t2, labels, transform, transform_label):
        super().__init__()
        self.flair = flair
        self.t1 = t1
        self.t2 =t2
        self.labels = labels
        self.transform = transform
        self.transform_label = transform_label
        self.len = len(self.flair)
           
    def __len__(self): 
        return self.len          

    def __getitem__(self, index):
        image=self.flair[index]
        label = self.labels[index]  

        flair = nib.load(image)
        flair_im= flair.get_fdata()
        #flair_im = flair_im[39:-39,41:-41,39:-39] 

        mask = nib.load(label)
        mask_im= mask.get_fdata()
        #mask_im = mask_im[39:-39,41:-41,39:-39] 
        
        flair_im=resize(flair_im, (128, 128, 48), preserve_range=True)
        mask_im = resize(mask_im, (128, 128, 48), preserve_range=True)
        image = self.transform(flair_im)
        mask = self.transform_label(mask_im)
        
        return image,mask    
    

class MICAI(Dataset):
    def __init__(self, flair, t1, labels, transform, transform_label):
        super().__init__()
        self.flair = flair
        self.t1 = t1
        self.labels = labels
        self.transform = transform
        self.transform_label = transform_label
        self.len = len(self.flair)
           
    def __len__(self): 
        return self.len          

    def __getitem__(self, index):
        image=self.flair[index]
        label = self.labels[index]  

        flair = nib.load(image)
        flair_im= flair.get_fdata()

        mask = nib.load(label)
        mask_im= mask.get_fdata()
        
        flair_im= resize(flair_im, (128, 128, 24), preserve_range=True)
        mask_im = resize(mask_im, (128, 128, 24), preserve_range=True)

        image = self.transform(flair_im)
        mask = self.transform_label(mask_im)
        return image,mask        

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 0.0

    def forward(self, y_pred, y_true):
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc            