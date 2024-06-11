import os
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib

def createDictionary(path:str, DICT: dict,location:str):
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

class ICPR(Dataset):
    def __init__(self, flair, t1,t2, labels, transform, transform_label, onlyFLAIR):
        super().__init__()
        self.flair = flair
        self.t1 = t1
        self.t2 =t2
        self.labels = labels
        self.transform = transform
        self.transform_label = transform_label
        self.images, self.masks= self.__totalimages__(onlyFLAIR)
        self.len = len(self.images)
           
    def __len__(self): 
        return self.len
    
    def __totalimages__(self, onlyFLAIR):
        images = []
        masks = []
        for flair,t1,t2, label in zip(self.flair,self.t1,self.t2, self.labels):
            flair = nib.load(flair)
            flair_im= flair.get_fdata() 

            t1=nib.load(t1)
            t1_im=t1.get_fdata()

            t2=nib.load(t2)
            t2_im=t2.get_fdata()

            lab = nib.load(label)
            label_img =  lab.get_fdata()
            if(onlyFLAIR):
                images.append(flair_im[])
                masks.append(label_img)    
            else:
                images.append(t1_im)
                masks.append(label_img) 
                images.append(t2_im)
                masks.append(label_img) 
        return images, masks            

    def __getitem__(self, index):
        image=self.images[index]
        mask = self.masks[index]  
        image = self.transform(image)
        mask = self.transform_label(mask)
        return image,mask              