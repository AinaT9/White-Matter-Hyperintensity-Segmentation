import nibabel as nib
import os
import scipy
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random 
import cv2
import numpy
from skimage.filters import threshold_otsu

def createDictionary(path:str, DICT: dict,location:str):
    subdirectories= os.listdir(path)
    for x in subdirectories:
        DICT["mask"].append(os.path.join(path,x, "wmh.nii.gz"))
        DICT["pathsT1"].append(os.path.join(path,x, "pre", "T1.nii.gz"))
        DICT["pathsFLAIR"].append(os.path.join(path,x, "pre", "FLAIR.nii.gz"))
        DICT["location"].append(location)

def divideDataset(dictio: dict, per=0.8):
    num_values=len(dictio["pathsFLAIR"])
    idx= list(range(num_values))
    random.shuffle(idx)
    part= int(num_values*per)
    train ={k: [v[i] for i in idx[:part]] for k, v in dictio.items()}
    val ={k: [v[i] for i in idx[part:]] for k, v in dictio.items()}
    return train,val


def add_transformation(image,final_size:int,options:bool, isMask:bool):
    if(not torch.is_tensor(image)):
        image= transforms.ToTensor()(image)
    # IF ONLY RESIZE
    if(options): 
        if(isMask):resize=transforms.Resize((final_size,final_size), antialias=True,interpolation= transforms.InterpolationMode.NEAREST)
        else:resize = transforms.Resize((final_size,final_size), antialias=True)
        image=resize(image)
        
    # IF CROP 
    # ELSE PAD
    else:
        if (image.shape[1] < final_size or image.shape[2] < final_size):
            pad_width = max(final_size - image.shape[1], 0)
            pad_height = max(final_size -  image.shape[2], 0)    
            total=(pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2) 
            padding=transforms.Pad(total, fill=0)
            image=padding(image)
        
        if(image.shape[1]>final_size or image.shape[2]>final_size):
            crop = transforms.CenterCrop((final_size,final_size))
            image=crop(image)

        

    return image

def add_normalization(image,n:int):
    match n:
        case 1:
            image= transforms.ToTensor()(image)
            return gaussian_normalization(image)
        case 2:
            image= transforms.ToTensor()(image)
            return minmax_normalization(image)
        case 3:
            mask= brainMask(image)
            image= transforms.ToTensor()(image)
            mask= transforms.ToTensor()(mask)
            return gaussian_normalizationFILL(image, mask)
        case 4:
            mask= brainMask(image)
            image= transforms.ToTensor()(image)
            mask= transforms.ToTensor()(mask)
            return minmax_normalizationFILL(image, mask)
        case 5: 
            mask= brainfilling(brainMask(image))
            image= transforms.ToTensor()(image)
            mask= transforms.ToTensor()(mask)
            return gaussian_normalizationFILL(image, mask)
        case 6:    
            mask= brainfilling(brainMask(image))
            image= transforms.ToTensor()(image)
            mask= transforms.ToTensor()(mask)
            return minmax_normalizationFILL(image, mask)
        


#def removeSkull(image)     
    # IF REMOVE SKULL 
    ################



def brainMask(image):
# Aislar cerebro
    thr=threshold_otsu(image)
    return image > thr

# def brainfilling(image):   
#    image_np = image.cpu().numpy()   
#    image_f=scipy.ndimage.morphology.binary_fill_holes(image_np) 
#    return torch.tensor(image_f, dtype=torch.bool)

def brainfilling(image):
    return scipy.ndimage.morphology.binary_fill_holes(image) 


def gaussian_normalizationFILL(image, brain):
    mean= torch.mean(image[brain==1])
    std=torch.std(image[brain==1])
    norm =(image-mean)/std
    return norm

def minmax_normalizationFILL(image,brain):
    min= torch.min(image[brain==1])
    max= torch.max(image[brain==1])
    norm = (image - min) / (max - min)
    return norm

def gaussian_normalization(image):
    mean= torch.mean(image)
    std=torch.std(image)
    norm =(image-mean)/std
    return norm

def minmax_normalization(image):
    min= torch.min(image)
    max= torch.max(image)
    norm = (image - min) / (max - min)
    return norm


def dataAugmentation(image, deg,sca=None,she=None):
    if(type(image)==numpy.ndarray):image= transforms.ToTensor()(image)
    transform = transforms.RandomAffine(degrees=deg, translate=(0,0), scale=sca, shear=she)
    return transform(image)

#ONLY RESIZE OR CROP
def transform_setter(size: int, hasResize:True):
    transform=transforms.Compose([
        lambda x: add_transformation(x, size,hasResize, False),
    ])

    transform_label=transforms.Compose([
        lambda x: add_transformation(x, size,hasResize, True),
    ])
    return transform, transform_label

#RESIZE/CROP AND NORMALIZATION
def transform_normalization(size: int, hasResize:True, n:0):
    transform=transforms.Compose([
        lambda x: add_transformation(add_normalization(x,n), size,hasResize, False),
    ])

    transform_label=transforms.Compose([
        lambda x: add_transformation(x, size,hasResize, True)
    ])
    return transform, transform_label

class Slices(Dataset):
    def __init__(self, images, labels, transform, transform_label, slices_deletion):
        super().__init__()
        self.paths = images
        self.labels = labels
        self.slices_deletion=slices_deletion
        self.images, self.masks= self.__totalimages__()
        self.len = len(self.images)
        self.transform = transform
        self.transform_label = transform_label
        
    def __len__(self): 
        return self.len
    
    def __totalimages__(self):
        images = []
        masks = []
        for path, label in zip(self.paths, self.labels):
            img = nib.load(path)
            image= img.get_fdata() 

            lab = nib.load(label)
            label_img =  lab.get_fdata()
            label_img[label_img==2]=0

            n_slices=image.shape[2]
            lim_inf= int(n_slices*0.1)
            lim_sup=int(n_slices*0.9)
            if(self.slices_deletion):
                for ii in range(0,n_slices):
                    im = image[:, :, ii]
                    lab=label_img[:, :, ii]
                    if(ii>lim_inf and ii<lim_sup or cv2.countNonZero(lab)!=0):
                        images.append(im)
                        masks.append(lab)           
            else:
                for ii in range(0,n_slices):
                    im = image[:, :, ii]
                    lab=label_img[:, :, ii]
                    images.append(im)
                    masks.append(lab)

        return images, masks            

    def __getitem__(self, index):
            image=self.images[index]
            mask= self.masks[index]
            image = self.transform(image)
            label_img = self.transform_label(mask)    
            return image,label_img  

    

def dataLoaders(path:str,train:dict, val:dict,transform, transform_label, isShuffled=False, size=30,slices_deletion=False):    
    train_data = Slices(train.get(path), train.get("mask"), transform, transform_label,slices_deletion)
    print(train_data.len)
    val_data = Slices(val.get(path), val.get("mask"), transform, transform_label,slices_deletion)
    print(val_data.len)
    train_dl = DataLoader(train_data, batch_size=size, shuffle=isShuffled)
    val_dl = DataLoader(val_data, batch_size=size, shuffle=isShuffled)
    return train_data, val_data, train_dl,val_dl 

class Concatenate(Dataset):
    def __init__(self, flair, t1, labels, transform, transform_label):
        super().__init__()
        self.flair = flair
        self.t1 = t1
        self.labels = labels
        self.images, self.masks= self.__totalimages__()
        self.len = len(self.images)
        self.transform = transform
        self.transform_label = transform_label
           
    def __len__(self): 
        return self.len
    
    def __totalimages__(self):
        images = []
        masks = []
        for flair,t1, label in zip(self.flair,self.t1, self.labels):
            flair = nib.load(flair)
            flair_im= flair.get_fdata() 

            t1=nib.load(t1)
            t1_im=t1.get_fdata()

            lab = nib.load(label)
            label_img =  lab.get_fdata()
            label_img[label_img==2]=0

            n_slices=t1_im.shape[2]
            lim_inf= int(n_slices*0.1)
            lim_sup=int(n_slices*0.9)
            for ii in range(0,n_slices):
                t = t1_im[:, :, ii]
                fl=flair_im[:,:,ii]
                lab=label_img[:, :, ii]
                if(ii>lim_inf and ii<lim_sup or cv2.countNonZero(lab)!=0):
                    im=numpy.concatenate((t[...,numpy.newaxis],fl[...,numpy.newaxis]), axis=2)
                    images.append(im)
                    masks.append(lab)           
        return images, masks            

    def __getitem__(self, index):
        image=self.images[index]
        mask= self.masks[index]
        image = self.transform(image)
        label_img = self.transform_label(mask)    
        return image,label_img       

def dataLoadersConcatenate(flair:str,t1:str,train:dict, val:dict,transform, transform_label, isShuffled=False, size=30):    
    train_data = Concatenate(train.get(flair),train.get(t1), train.get("mask"), transform, transform_label)
    print(train_data.len)
    val_data = Slices(val.get(flair),val.get(t1), val.get("mask"), transform, transform_label)
    print(val_data.len)
    train_dl = DataLoader(train_data, batch_size=size, shuffle=isShuffled)
    val_dl = DataLoader(val_data, batch_size=size, shuffle=isShuffled)
    return train_data, val_data, train_dl,val_dl 
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